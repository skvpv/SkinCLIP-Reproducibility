from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..common import PKG_ROOT, load_paths, load_yaml, now_iso, sha256_file, slurm_context, software_versions, write_json


def check_allocation(expect_gpu: bool, min_free_gib: float) -> dict:
    import torch

    r = {"cuda_available": torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch.cuda.is_available():
        r["device_count"] = torch.cuda.device_count()
        free, total = torch.cuda.mem_get_info()
        r.update(gpu_name=torch.cuda.get_device_name(0), total_gib=total / 2**30, free_gib=free / 2**30,
                 capability=".".join(map(str, torch.cuda.get_device_capability(0))))
        x = torch.randn(256, 256, device="cuda")
        r["matmul_ok"] = bool(torch.isfinite(x @ x).all())
    ok = (not expect_gpu) or (r["cuda_available"] and r.get("device_count") == 1 and r.get("free_gib", 0) >= min_free_gib
                              and r.get("matmul_ok"))
    r["pass"] = bool(ok)
    return r


def check_weights(exp: dict) -> dict:
    out = {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "HF_HOME": os.environ.get("HF_HOME"),
           "TORCH_HOME": os.environ.get("TORCH_HOME")}
    tests = {
        "vit": lambda: __import__("timm").create_model(exp["model"]["image_backbone"], pretrained=True, num_classes=0),
        "bio_clinicalbert": lambda: (__import__("transformers").AutoTokenizer.from_pretrained(exp["model"]["text_encoder"]),
                                     __import__("transformers").AutoModel.from_pretrained(exp["model"]["text_encoder"])),
        "resnet50": lambda: __import__("torchvision").models.resnet50(weights="IMAGENET1K_V2"),
        "biomedclip": lambda: __import__("open_clip").create_model_from_pretrained(exp["model"]["biomedclip"]),
    }
    for k, fn in tests.items():
        try:
            fn()
            out[k] = "LOADED"
        except Exception as exc:
            out[k] = f"FAIL: {exc.__class__.__name__}: {str(exc)[:300]}"
    out["pass"] = all(v == "LOADED" for k, v in out.items() if k in tests)
    return out


def check_paths(paths: dict) -> dict:
    raw = paths.get("raw", {})
    res = {}
    for k, v in raw.items():
        if not v:
            continue
        vs = v if isinstance(v, list) else [v]
        res[k] = all(Path(x).exists() for x in vs)
    res["work_dir_writable"] = os.access(Path(paths["work_dir"]).parent, os.W_OK)
    res["pass"] = all(res.values())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-gpu", action="store_true")
    ap.add_argument("--check-weights", action="store_true")
    ap.add_argument("--min-free-gib", type=float, default=8.0)
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    paths = load_paths(a.paths)
    exp = load_yaml(Path(a.config_dir) / "experiment.yaml")
    rec = {"time": now_iso(), "slurm": slurm_context(), "versions": software_versions(),
           "allocation": check_allocation(a.expect_gpu, a.min_free_gib), "paths": check_paths(paths),
           "config_sha256": {p.name: sha256_file(p) for p in sorted(Path(a.config_dir).glob("*.yaml"))}}
    if not rec["slurm"]["inside_slurm"]:
        rec["inside_slurm_pass"] = False
    if a.check_weights:
        rec["weights"] = check_weights(exp)
    checks = [rec["allocation"]["pass"], rec["paths"]["pass"], rec["slurm"]["inside_slurm"]] + \
             ([rec["weights"]["pass"]] if a.check_weights else [])
    rec["PASS"] = all(checks)
    out = Path(paths["work_dir"]) / "preflight" / f"{rec['slurm']['SLURM_JOB_ID'] or 'nojob'}.json"
    write_json(out, rec)
    print(f"preflight {'PASS' if rec['PASS'] else 'FAIL'} -> {out}")
    return 0 if rec["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
