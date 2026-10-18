from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .. import NUM_CLASSES
from ..common import PKG_ROOT, load_paths, load_yaml, now_iso, slurm_context, write_json
from ..losses import class_weights, total_loss
from ..models.factory import build_model, load_text_bank, needs_images, prompts_from_config
from .data_access import make_dataset
from .train import Preempted, model_inputs, train_run


def overfit_check(paths, cd: Path, device) -> dict:
    exp = load_yaml(cd / "experiment.yaml")
    _, psha = prompts_from_config(cd / "prompts.yaml")
    model, (mean, std), _ = build_model("F11", exp, load_text_bank(paths["work_dir"], psha)[0])
    model.to(device).train()
    ds, df = make_dataset(paths, exp, "train", "training", mean, std, True, False)
    idx = np.random.default_rng(0).choice(len(ds), 16, replace=False)
    batch = torch.utils.data.default_collate([ds[int(i)] for i in idx])
    w = class_weights(df["label_idx"].to_numpy(), NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(40):
        out = model(**model_inputs(batch, device))
        loss, _ = total_loss(out["logits"], batch["label"].to(device), w, 0.5, 2.0, 1.0, 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return {"loss_first": losses[0], "loss_last": losses[-1], "pass": losses[-1] < 0.5 * losses[0]}


def resume_check(paths, cd: Path, root: Path) -> dict:
    smoke = {"subset": 256, "epochs": 2, "max_batches": 8, "preempt_after_epoch": 0, "num_workers": 2}
    rdir = root / "F11" / "lam0.5" / "seed0"
    try:
        train_run("F11", 0, 0.5, paths, cd, smoke, out_root=root)
        return {"pass": False, "error": "pre-emption was not simulated"}
    except Preempted:
        pass
    rec = train_run("F11", 0, 0.5, paths, cd, smoke, out_root=root)
    log = [json.loads(line) for line in (rdir / "train_log.jsonl").read_text().splitlines()]
    resumed = [h for h in rec["history"] if h.get("resumed_from_epoch") is not None]
    ok = (rec["state"] == "TRAINING_COMPLETED" and [e["epoch"] for e in log] == [0, 1] and resumed
          and resumed[0]["resumed_from_epoch"] == 0 and (rdir / "val_predictions_best.csv").exists())
    return {"pass": bool(ok), "epochs_logged": [e["epoch"] for e in log], "state": rec["state"],
            "result_file": str(rdir / "val_predictions_best.csv")}


def arms_check(paths, cd: Path, root: Path) -> dict:
    exp = load_yaml(cd / "experiment.yaml")
    res = {}
    for arm in ("F01", "F10", "F00", "MB", "M0") + (("BCLP",) if (Path(paths["work_dir"]) / "features" / "biomedclip_train.npy").exists() else ()):
        try:
            r = train_run(arm, 0, 0.5 if exp["arms"][arm]["kind"] == "prototype" else 0.0, paths, cd,
                          {"subset": 128, "epochs": 1, "max_batches": 3, "num_workers": 2}, out_root=root)
            res[arm] = r["state"]
        except Exception as exc:
            res[arm] = f"FAIL {exc.__class__.__name__}: {exc}"
    res["pass"] = all(v == "TRAINING_COMPLETED" for k, v in res.items())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    paths, cd = load_paths(a.paths), Path(a.config_dir)
    root = Path(paths["work_dir"]) / "smoke" / "runs"
    if root.exists():
        shutil.rmtree(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = {"time": now_iso(), "slurm": slurm_context(), "device": str(device)}
    for name, fn in (("overfit_16", lambda: overfit_check(paths, cd, device)), ("train_ckpt_resume_eval", lambda: resume_check(paths, cd, root)),
                     ("all_arm_kinds", lambda: arms_check(paths, cd, root))):
        try:
            rep[name] = fn()
        except Exception as exc:
            rep[name] = {"pass": False, "error": f"{exc.__class__.__name__}: {exc}"}
    rep["PASS"] = all(rep[k]["pass"] for k in ("overfit_16", "train_ckpt_resume_eval", "all_arm_kinds"))
    write_json(Path(paths["work_dir"]) / "smoke" / "smoke_report.json", rep)
    print(json.dumps(rep, indent=1, default=str)[:3000])
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
