from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from ..common import PKG_ROOT, load_paths, load_yaml, now_iso, read_json, software_versions, slurm_context, write_json
from ..data.metadata import META_DIM
from ..models.factory import build_model, load_text_bank, prompts_from_config


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def measure_model(model, inputs_fn, device, n_warm=10, n_rep=100) -> dict:
    model.eval().to(device)
    res = {"params_total": sum(p.numel() for p in model.parameters()),
           "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    try:
        from fvcore.nn import FlopCountAnalysis

        fl = FlopCountAnalysis(model, inputs_fn(1, device))
        fl.unsupported_ops_warnings(False)
        fl.uncalled_modules_warnings(False)
        res["gflops_per_image"] = fl.total() / 1e9
    except Exception as exc:
        res["gflops_per_image"] = None
        res["gflops_error"] = str(exc)
    with torch.no_grad():
        for bs in (1, 64):
            x = inputs_fn(bs, device)
            for _ in range(n_warm):
                model(*x)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(n_rep):
                t0 = time.perf_counter()
                model(*x)
                _sync(device)
                ts.append(time.perf_counter() - t0)
            res[f"latency_ms_bs{bs}"] = 1000 * statistics.median(ts)
            if device.type == "cuda" and bs == 64:
                res["peak_infer_vram_mib_bs64"] = torch.cuda.max_memory_allocated() / 2**20
    return res


def training_cost(paths, arm: str) -> dict:
    checkpoint = read_json(Path(paths["work_dir"]) / "integrity" / "checkpoint_manifest.json")
    tt, ep, peak = [], [], []
    for r in checkpoint["runs"]:
        if r["arm"] != arm:
            continue
        log = Path(r["checkpoint"]).parent / "train_log.jsonl"
        recs = [json.loads(line) for line in log.read_text().splitlines()]
        tt.append(sum(x["t_train_s"] + x["t_val_s"] for x in recs))
        ep.append(len(recs))
        peak.append(max((x.get("peak_vram_mib") or 0) for x in recs))
    if not tt:
        return {}
    return {"train_time_s_mean": statistics.mean(tt), "epochs_mean": statistics.mean(ep), "peak_train_vram_mib_max": max(peak)}


def run_cost(paths: dict, config_dir: Path) -> pd.DataFrame:
    exp = load_yaml(config_dir / "experiment.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, psha = prompts_from_config(config_dir / "prompts.yaml")
    S = exp["model"]["img_size"]
    rows = []
    for arm in ("F11", "F01", "F10", "F00", "MB", "M0", "BCLP"):
        H = load_text_bank(paths["work_dir"], psha)[0] if exp["arms"][arm].get("protos") == "text" else None
        model, _, _ = build_model(arm, exp, H)
        kind = exp["arms"][arm]["kind"]
        if kind in ("prototype", "metablock"):
            fn = lambda b, d: (torch.randn(b, 3, S, S, device=d), torch.randn(b, META_DIM, device=d))
        elif kind == "meta_only":
            fn = lambda b, d: (None, torch.randn(b, META_DIM, device=d))
        else:
            fd = exp["model"].get("biomedclip_feat_dim", 512)
            fn = lambda b, d, fd=fd: (None, torch.randn(b, META_DIM, device=d), torch.randn(b, fd, device=d))
        r = {"arm": arm, **measure_model(model, fn, device), **training_cost(paths, arm)}
        if kind == "biomedclip_probe":
            r["note"] = "plus frozen BiomedCLIP image encoder at inference (row BiomedCLIP_visual)"
        rows.append(r)
    from ..models.biomedclip import load_biomedclip

    bc, _ = load_biomedclip(exp["model"]["biomedclip"], str(device))
    rows.append({"arm": "BiomedCLIP_visual", **measure_model(bc.visual, lambda b, d: (torch.randn(b, 3, 224, 224, device=d),), device)})
    df = pd.DataFrame(rows)
    rd = Path(paths["work_dir"]) / "results"
    rd.mkdir(parents=True, exist_ok=True)
    df.to_csv(rd / "table_cost.csv", index=False)
    write_json(rd / "cost_context.json", {"time": now_iso(), "slurm": slurm_context(), "versions": software_versions(),
                                         "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                                         "note": "Bio-ClinicalBERT prompt encoding is a one-off frozen cost (7 prompts)."})
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    print(run_cost(load_paths(a.paths), Path(a.config_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
