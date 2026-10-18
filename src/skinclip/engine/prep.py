from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import PKG_ROOT, load_paths, load_yaml, require_cohort, write_json
from ..data.cache import build_cache, load_cache
from ..models.factory import encode_prompts, prompts_from_config, save_text_bank, synthetic_text_bank
from .data_access import split_frame


def text_bank(paths, config_dir: Path) -> dict:
    cohort = require_cohort(paths)
    exp = load_yaml(config_dir / "experiment.yaml")
    prompts, sha = prompts_from_config(config_dir / "prompts.yaml")
    if sha not in cohort["artifacts"].values():
        raise RuntimeError("prompts.yaml is not the cohort-frozen version")
    if exp.get("synthetic"):
        H, info = synthetic_text_bank(), {"text_encoder": "SYNTHETIC (tests only)"}
    else:
        H, info = encode_prompts(exp["model"]["text_encoder"], prompts, exp["model"]["text_max_len"])
    p = save_text_bank(paths["work_dir"], H, info, sha)
    return {"text_bank": str(p), **info}


def cache_cohort(paths, config_dir: Path, cohort: str, workers: int) -> dict:
    require_cohort(paths)
    exp = load_yaml(config_dir / "experiment.yaml")
    df = pd.read_csv(Path(paths["work_dir"]) / "cohorts" / f"{cohort}.csv")
    npy = build_cache(df, paths["work_dir"], cohort, exp["image"], workers)
    return {"cache": str(npy), "n": int(len(df))}


def bc_features_trainval(paths, config_dir: Path, device: str) -> dict:
    from ..models.biomedclip import image_features, load_biomedclip

    exp = load_yaml(config_dir / "experiment.yaml")
    model, _ = load_biomedclip(exp["model"]["biomedclip"], device)
    full = pd.read_csv(Path(paths["work_dir"]) / "cohorts" / "ham.csv")["image_id"].tolist()
    mm = load_cache(paths["work_dir"], "ham", exp["image"]["cache_size"], full)
    out = Path(paths["work_dir"]) / "features"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for sp in ("train", "val"):
        df, _ = split_frame(paths, sp, "training")
        f = image_features(model, mm[df["row"].to_numpy()], device)
        np.save(out / f"biomedclip_{sp}.npy", f)
        res[sp] = list(f.shape)
    write_json(out / "biomedclip_trainval.json", {"model": exp["model"]["biomedclip"], "shapes": res})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["text_bank", "cache_ham", "bc_features"])
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    paths, cd = load_paths(a.paths), Path(a.config_dir)
    r = {"text_bank": lambda: text_bank(paths, cd), "cache_ham": lambda: cache_cohort(paths, cd, "ham", a.workers),
         "bc_features": lambda: bc_features_trainval(paths, cd, a.device)}[a.task]()
    print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
