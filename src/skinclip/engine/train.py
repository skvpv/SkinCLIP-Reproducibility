from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import CLASSES, NUM_CLASSES
from ..analysis.metrics import classification_metrics
from ..common import (PKG_ROOT, JsonlLogger, RunRecord, load_paths, load_yaml, require_cohort, seed_everything, sha256_file,
                      sha256_obj, software_versions)
from ..data.dataset import make_loader
from ..losses import class_weights, total_loss
from ..models.factory import build_model, load_text_bank, needs_features, needs_images, prompts_from_config
from .data_access import make_dataset

INPUT_KEYS = ("image", "meta", "features")


class Preempted(SystemExit):
    pass


def validate_experiment(exp: dict) -> None:
    if exp.get("synthetic") and os.environ.get("SKINCLIP_ALLOW_SYNTHETIC") != "1":
        raise RuntimeError("experiment.yaml has synthetic: true; refusing to run outside the test suite")


def model_inputs(batch: dict, device) -> dict:

    return {k: batch[k].to(device, non_blocking=True) for k in INPUT_KEYS if k in batch}


@torch.no_grad()
def predict(model, loader, device, amp: bool, return_q: bool = False) -> dict:
    model.eval()
    logits, labels, qs = [], [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(**model_inputs(batch, device))
        logits.append(out["logits"].float().cpu())
        labels.append(batch["label"])
        if return_q and "q" in out:
            qs.append(out["q"].float().cpu())
    res = {"logits": torch.cat(logits).numpy(), "labels": torch.cat(labels).numpy()}
    res["probs"] = torch.softmax(torch.from_numpy(res["logits"]), 1).numpy()
    if qs:
        res["q"] = torch.cat(qs).numpy()
    return res


def run_dir_for(work: Path, arm: str, lam: float, seed: int) -> Path:
    return work / "runs" / arm / f"lam{lam:g}" / f"seed{seed}"


def _param_groups(model, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n == "theta" else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def train_run(arm: str, seed: int, lam: float, paths: dict, config_dir: Path, smoke: dict | None = None,
              out_root: Path | None = None) -> dict:
    exp = load_yaml(config_dir / "experiment.yaml")
    validate_experiment(exp)
    cohort = require_cohort(paths)
    tr = exp["train"] if exp["arms"][arm]["kind"] != "biomedclip_probe" else {**exp["train"], **exp["probe"]}
    work = Path(paths["work_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and tr["amp"] == "bf16"
    seed_everything(seed, tr["deterministic"])

    text_H, text_sha = None, None
    prompts, prompts_sha = prompts_from_config(config_dir / "prompts.yaml")
    if prompts_sha not in cohort["artifacts"].values():
        raise RuntimeError("prompts.yaml differs from the cohort-frozen version")
    if exp["arms"][arm].get("protos") == "text":
        text_H, _ = load_text_bank(work, prompts_sha)
        text_sha = sha256_obj(text_H.numpy().round(6).tolist())
    model, (mean, std), minfo = build_model(arm, exp, text_H)
    model.to(device)

    feats = {}
    if needs_features(arm, exp):
        for sp in ("train", "val"):
            feats[sp] = np.load(work / "features" / f"biomedclip_{sp}.npy")
    imgs = needs_images(arm, exp)
    train_ds, train_df = make_dataset(paths, exp, "train", "training", mean, std, imgs, True, feats.get("train"))
    val_ds, val_df = make_dataset(paths, exp, "val", "training", mean, std, imgs, False, feats.get("val"))
    if smoke and smoke.get("subset"):
        idx = np.random.default_rng(0).choice(len(train_ds), size=min(smoke["subset"], len(train_ds)), replace=False)
        train_ds = torch.utils.data.Subset(train_ds, sorted(idx.tolist()))
    w = class_weights(train_df["label_idx"].to_numpy(), NUM_CLASSES).to(device)

    cfg_sig = sha256_obj({"arm": exp["arms"][arm], "model": exp["model"], "train": tr, "image": exp["image"],
                          "augment": exp["augment"], "lam": lam, "seed": seed, "cohort": sorted(cohort["artifacts"].values()),
                          "text_bank": text_sha, "smoke": smoke})
    rdir = (out_root or work / "runs") / arm / f"lam{lam:g}" / f"seed{seed}"
    rdir.mkdir(parents=True, exist_ok=True)
    run_id = f"{arm}_lam{lam:g}_s{seed}_{cfg_sig[:10]}"
    rec = RunRecord(rdir, run_id, {"arm": arm, "lam": lam, "seed": seed, "config_signature": cfg_sig,
                                   "model_info": minfo, "versions": software_versions(),
                                   "class_weights": w.cpu().tolist(), "n_train": len(train_ds), "n_val": len(val_ds),
                                   "train_ids_sha256": sha256_obj(train_df["image_id"].tolist()),
                                   "val_ids_sha256": sha256_obj(val_df["image_id"].tolist()), "smoke": smoke})
    log = JsonlLogger(rdir / "train_log.jsonl")
    opt = torch.optim.Adam(_param_groups(model, tr["weight_decay"]), lr=tr["lr"])
    start, best, best_epoch, bad = 0, -math.inf, -1, 0
    last = rdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["cfg_sig"] != cfg_sig:
            raise RuntimeError(f"existing checkpoint in {rdir} has a different config signature; refusing to resume")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best, best_epoch, bad = ck["epoch"] + 1, ck["best"], ck["best_epoch"], ck["bad"]
        if ck.get("finished"):
            if rec.data.get("state") != "TRAINING_COMPLETED":
                rec.event("TRAINING_COMPLETED", best_epoch=best_epoch, best_val_bma=best, repaired=True,
                          checksums={"best.pt": sha256_file(rdir / "best.pt"), "last.pt": sha256_file(last),
                                     "val_predictions_best.csv": sha256_file(rdir / "val_predictions_best.csv")})
            return rec.data
        rec.event("RUNNING", resumed_from_epoch=ck["epoch"], resume_checkpoint=str(last))
    else:
        rec.event("RUNNING", resumed_from_epoch=None)
    max_epochs = smoke.get("epochs", tr["max_epochs"]) if smoke else tr["max_epochs"]
    nw = smoke.get("num_workers", tr["num_workers"]) if smoke else tr["num_workers"]
    finished = False
    try:
        for epoch in range(start, max_epochs):
            torch.manual_seed(seed * 100_003 + epoch + 1)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model.train()
            loader = make_loader(train_ds, tr["batch_size"], True, seed, epoch, nw)
            t0, n_it, run_loss, parts = time.time(), 0, 0.0, {"loss_cls": 0.0, "loss_clip": 0.0}
            for batch in loader:
                y = batch["label"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    out = model(**model_inputs(batch, device))
                loss, comp = total_loss(out["logits"], y, w, lam, tr["focal_gamma"], tr["focal_alpha"], tr["label_smoothing"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch} iter {n_it}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                opt.step()
                run_loss += float(loss.detach())
                for k in parts:
                    parts[k] += comp[k] if not math.isnan(comp[k]) else 0.0
                n_it += 1
                if smoke and smoke.get("max_batches") and n_it >= smoke["max_batches"]:
                    break
            t_train = time.time() - t0
            t1 = time.time()
            vl = make_loader(val_ds, max(tr["batch_size"], 64), False, seed, 0, nw)
            vp = predict(model, vl, device, amp)
            vm = classification_metrics(vp["labels"], vp["probs"], auroc=False)
            t_val = time.time() - t1
            improved = vm["bma"] > best + 1e-12
            if improved:
                best, best_epoch, bad = vm["bma"], epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "cfg_sig": cfg_sig, "val_bma": best}, rdir / "best.pt")
                pd.DataFrame({"image_id": val_df["image_id"], "label_idx": vp["labels"],
                              **{f"p_{c}": vp["probs"][:, i] for i, c in enumerate(CLASSES)}}).to_csv(
                    rdir / "val_predictions_best.csv", index=False)
            else:
                bad += 1
            stop = bad >= tr["patience"]
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "best": best,
                        "best_epoch": best_epoch, "bad": bad, "cfg_sig": cfg_sig, "finished": stop or epoch == max_epochs - 1},
                       last)
            peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
            log.log(epoch=epoch, iters=n_it, train_loss=run_loss / max(n_it, 1),
                    **{k: v / max(n_it, 1) for k, v in parts.items()}, val_bma=vm["bma"], val_macro_f1=vm["macro_f1"],
                    val_acc=vm["accuracy"], improved=improved, best_epoch=best_epoch, t_train_s=t_train, t_val_s=t_val,
                    s_per_iter=t_train / max(n_it, 1), peak_vram_mib=peak,
                    logit_scale=float(model.scale().detach()) if hasattr(model, "scale") else None)
            if smoke and smoke.get("preempt_after_epoch") == epoch and not (rdir / ".preempted_once").exists():
                (rdir / ".preempted_once").write_text("simulated pre-emption for the resume test\n")
                raise Preempted(99)
            if stop:
                break
        finished = True
    except Preempted:
        rec.event("RUNNING", note="simulated pre-emption; resubmit to resume")
        raise
    except Exception as exc:
        rec.event("FAILED", error=f"{exc.__class__.__name__}: {exc}")
        raise
    if finished:
        rec.event("TRAINING_COMPLETED", best_epoch=best_epoch, best_val_bma=best,
                  checksums={"best.pt": sha256_file(rdir / "best.pt"), "last.pt": sha256_file(last),
                             "val_predictions_best.csv": sha256_file(rdir / "val_predictions_best.csv")})
    return rec.data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    try:
        train_run(a.arm, a.seed, a.lam, load_paths(a.paths), Path(a.config_dir))
    except Preempted:
        return 99
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
