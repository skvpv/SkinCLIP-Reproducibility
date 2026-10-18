from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from ..common import PKG_ROOT, load_paths, load_yaml, now_iso, slurm_context, write_json
from .train import train_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    ap.add_argument("--safety", type=float, default=1.5)
    a = ap.parse_args(argv)
    paths, cd = load_paths(a.paths), Path(a.config_dir)
    exp = load_yaml(cd / "experiment.yaml")
    root = Path(paths["work_dir"]) / "pilot" / "runs"
    if root.exists():
        shutil.rmtree(root)
    rep = {"time": now_iso(), "slurm": slurm_context(),
           "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "arms": {}}
    for arm, lam in (("F11", 0.5), ("MB", 0.0)):
        t0 = time.time()
        train_run(arm, 0, lam, paths, cd, {"epochs": 1, "num_workers": exp["train"]["num_workers"]}, out_root=root)
        rdir = root / arm / f"lam{lam:g}" / "seed0"
        e = json.loads((rdir / "train_log.jsonl").read_text().splitlines()[0])
        t1 = time.time()
        torch.load(rdir / "last.pt", map_location="cpu", weights_only=False)
        rep["arms"][arm] = {"wall_s_incl_setup": time.time() - t0, "epoch_train_s": e["t_train_s"], "val_s": e["t_val_s"],
                            "s_per_iter": e["s_per_iter"], "peak_vram_mib": e["peak_vram_mib"],
                            "checkpoint_mib": (rdir / "last.pt").stat().st_size / 2**20, "resume_load_s": time.time() - t1}
    E = exp["train"]["max_epochs"]
    per_run_worst_h = {k: (v["epoch_train_s"] + v["val_s"]) * E / 3600 for k, v in rep["arms"].items()}
    n_vit, n_mb = 16, 3
    rep["projection"] = {"max_epochs": E, "per_run_worst_case_h": per_run_worst_h,
                         "campaign_worst_case_gpu_h": per_run_worst_h["F11"] * n_vit + per_run_worst_h["MB"] * n_mb,
                         "note": "worst case = no early stopping; ESTIMATE ONLY (queue time excluded)"}
    write_json(Path(paths["work_dir"]) / "pilot" / "pilot_report.json", rep)
    worst = max(per_run_worst_h.values()) * a.safety
    peak = max(v["peak_vram_mib"] or 0 for v in rep["arms"].values()) / 1024
    tpl = (cd / "integrity" / "resources.env.template").read_text()
    hours = int(worst) + 1
    sug = (tpl.replace("resources_STATUS=NOT_FROZEN", "resources_STATUS=SUGGESTED_NOT_FROZEN   # Review: review, then set FROZEN")
              .replace("resources_PILOT_JOB=", f"resources_PILOT_JOB={rep['slurm']['SLURM_JOB_ID']}")
              .replace("GPU_TIME=04:00:00", f"GPU_TIME={hours:02d}:00:00   # pilot worst case x {a.safety}"))
    sug += (f"\n# measured: peak VRAM {peak:.2f} GiB; per-run worst case (no early stop) {per_run_worst_h} h;"
            f" campaign worst case {rep['projection']['campaign_worst_case_gpu_h']:.1f} GPU-h (ESTIMATE)\n")
    (Path(paths["work_dir"]) / "pilot" / "resources.SUGGESTED.env").write_text(sug)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
