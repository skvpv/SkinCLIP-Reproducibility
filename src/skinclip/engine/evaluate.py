from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import CLASSES
from ..common import (PKG_ROOT, JsonlLogger, load_paths, load_yaml, now_iso, require_checkpoint, sha256_file, software_versions,
                      write_json)
from ..data.cache import build_cache
from ..data.dataset import make_loader
from ..models.factory import build_model, load_text_bank, needs_features, needs_images, prompts_from_config
from .data_access import make_dataset, split_frame
from .train import predict


def eval_splits(exp) -> list[str]:
    return ["internal_test", "e1", "e1_oot"] + (["e2", "e2_oot"] if exp.get("include_e2") else [])


def _eval_dir(paths) -> Path:
    d = Path(paths["work_dir"]) / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_preds(path: Path, df: pd.DataFrame, probs: np.ndarray) -> None:
    out = df[["image_id", "cluster_id", "institution", "label", "label_idx", "site_group", "age", "sex_raw"]].copy()
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, i]
    out["pred_idx"] = probs.argmax(1)
    out.to_csv(path, index=False)


def _bc_eval_features(paths, exp, split: str, device: str) -> np.ndarray:
    from ..data.cache import load_cache
    from ..models.biomedclip import image_features, load_biomedclip

    f = Path(paths["work_dir"]) / "features" / f"biomedclip_{split}.npy"
    if f.exists():
        return np.load(f)
    df, cohort = split_frame(paths, split, "evaluation")
    full = pd.read_csv(Path(paths["work_dir"]) / "cohorts" / f"{cohort}.csv")["image_id"].tolist()
    mm = load_cache(paths["work_dir"], cohort, exp["image"]["cache_size"], full)
    model, _ = load_biomedclip(exp["model"]["biomedclip"], device)
    feats = image_features(model, mm[df["row"].to_numpy()], device)
    np.save(f, feats)
    return feats


def evaluate_all(paths: dict, config_dir: Path, rerun_reason: str | None = None, workers: int = 0) -> dict:
    checkpoint = require_checkpoint(paths)
    exp = load_yaml(config_dir / "experiment.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and exp["train"]["amp"] == "bf16"
    ed, log = _eval_dir(paths), JsonlLogger(_eval_dir(paths) / "eval_log.jsonl")
    for cohort in sorted({split_frame(paths, s, "evaluation")[1] for s in eval_splits(exp)} - {"ham"}):
        df = pd.read_csv(Path(paths["work_dir"]) / "cohorts" / f"{cohort}.csv")
        build_cache(df, paths["work_dir"], cohort, exp["image"], workers)
    prompts, psha = prompts_from_config(config_dir / "prompts.yaml")
    done = {}
    for run in checkpoint["runs"] + [{"arm": "BCZS", "seed": None, "lam": None, "checkpoint": None}]:
        arm, seed = run["arm"], run["seed"]
        tag = f"{arm}" + (f"_s{seed}" if seed is not None else "")
        odir = ed / tag
        if (odir / "eval_record.json").exists() and not rerun_reason:
            done[tag] = "exists (skipped)"
            continue
        odir.mkdir(parents=True, exist_ok=True)
        if arm == "BCZS":
            from ..models.biomedclip import load_biomedclip, zero_shot_logits

            model, tok = load_biomedclip(exp["model"]["biomedclip"], str(device))
            for sp in eval_splits(exp):
                df, _ = split_frame(paths, sp, "evaluation")
                logits = zero_shot_logits(model, tok, _bc_eval_features(paths, exp, sp, str(device)), prompts, str(device))
                _save_preds(odir / f"{sp}.csv", df, torch.softmax(torch.from_numpy(logits), 1).numpy())
        else:
            text_H = load_text_bank(paths["work_dir"], psha)[0] if exp["arms"][arm].get("protos") == "text" else None
            model, (mean, std), _ = build_model(arm, exp, text_H)
            ck = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            model.to(device)
            for sp in eval_splits(exp):
                feats = _bc_eval_features(paths, exp, sp, str(device)) if needs_features(arm, exp) else None
                ds, df = make_dataset(paths, exp, sp, "evaluation", mean, std, needs_images(arm, exp), False, feats)
                res = predict(model, make_loader(ds, 128, False, 0, 0, workers), device, amp,
                              return_q=(arm == exp["interpret"]["arm"] and seed == exp["interpret"]["seed"]))
                _save_preds(odir / f"{sp}.csv", df, res["probs"])
                if "q" in res:
                    np.save(odir / f"{sp}_q.npy", res["q"])
        rec = {"arm": arm, "seed": seed, "lam": run["lam"], "time": now_iso(), "rerun_reason": rerun_reason,
               "checkpoint_sha256": sha256_file(run["checkpoint"]) if run["checkpoint"] else None,
               "outputs": {p.name: sha256_file(p) for p in sorted(odir.glob("*.csv"))}, "versions": software_versions()}
        write_json(odir / "eval_record.json", rec)
        log.log(**{k: v for k, v in rec.items() if k != "outputs"})
        done[tag] = "evaluated"
    write_json(ed / "evaluation_summary.json", {"time": now_iso(), "runs": done})
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    ap.add_argument("--rerun-reason", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    print(evaluate_all(load_paths(a.paths), Path(a.config_dir), a.rerun_reason, a.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
