from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .. import CLASSES
from ..common import PKG_ROOT, load_paths, load_yaml, write_json
from ..data.metadata import encode_metadata
from ..models.factory import build_model, load_text_bank, prompts_from_config


class AttentionRecorder:


    def __init__(self, vit):
        self.maps, self.hooks = [], []
        for blk in vit.blocks:
            blk.attn.fused_attn = False
            self.hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: self.maps.append(i[0].detach())))

    def clear(self):
        self.maps = []

    def remove(self):
        for h in self.hooks:
            h.remove()


def attention_rollout(maps: list[torch.Tensor], n_prefix: int = 1) -> torch.Tensor:

    A = None
    for a in maps:
        a = a.mean(1)
        a = a + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        A = a if A is None else a @ A
    cls = A[:, 0, n_prefix:]
    g = int(cls.shape[-1] ** 0.5)
    return cls.reshape(-1, g, g)


def integrated_gradients(model, x, meta, target, baseline, steps=50) -> torch.Tensor:
    alphas = torch.linspace(0, 1, steps + 1, device=x.device)
    grads = []
    for a in alphas:
        xi = (baseline + a * (x - baseline)).requires_grad_(True)
        out = model(xi, meta)["logits"]
        g, = torch.autograd.grad(out[torch.arange(len(target)), target].sum(), xi)
        grads.append(g)
    G = torch.stack(grads)
    avg = ((G[:-1] + G[1:]) / 2).mean(0)
    return (x - baseline) * avg


def blur(x, k=31, sigma=10.0):
    ax = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    xc = x.reshape(-1, 1, *x.shape[-2:])
    xc = F.conv2d(F.pad(xc, (k // 2, k // 2, 0, 0), mode="reflect"), g)
    xc = F.conv2d(F.pad(xc, (0, 0, k // 2, k // 2), mode="reflect"), g.transpose(-1, -2))
    return xc.reshape(x.shape)


def deletion_curve(model, x, meta, target, patch_scores, fractions, blurred, patch=16) -> np.ndarray:

    g = patch_scores.shape[-1]
    order = torch.argsort(patch_scores.flatten(), descending=True)
    probs = []
    for f in fractions:
        mask = torch.zeros(g * g, device=x.device)
        mask[order[: int(round(f * g * g))]] = 1
        m = mask.view(1, 1, g, g)
        m = F.interpolate(m, scale_factor=patch, mode="nearest")
        xd = x * (1 - m) + blurred * m
        with torch.no_grad():
            probs.append(torch.softmax(model(xd, meta)["logits"], -1)[0, target].item())
    return np.array(probs)


def select_samples(pred: pd.DataFrame, n_ok: int, n_bad: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pred = pred.sort_values("image_id").reset_index(drop=True)
    pred["pred_ok"] = pred["pred_idx"] == pred["label_idx"]
    out = []
    for c in range(len(CLASSES)):
        for ok, n in ((True, n_ok), (False, n_bad)):
            pool = pred.index[(pred.label_idx == c) & (pred.pred_ok == ok)].to_numpy()
            if len(pool):
                out += rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()
    return pred.loc[sorted(out)]


def run_interpret(paths: dict, config_dir: Path) -> dict:
    from ..common import require_checkpoint
    from ..data.cache import load_cache
    from ..engine.data_access import split_frame

    checkpoint = require_checkpoint(paths)
    exp = load_yaml(config_dir / "experiment.yaml")
    ic = exp["interpret"]
    run = next(r for r in checkpoint["runs"] if r["arm"] == ic["arm"] and r["seed"] == ic["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, psha = prompts_from_config(config_dir / "prompts.yaml")
    H = load_text_bank(paths["work_dir"], psha)[0]
    model, (mean, std), _ = build_model(ic["arm"], exp, H)
    model.load_state_dict(torch.load(run["checkpoint"], map_location="cpu", weights_only=False)["model"])
    model.eval().to(device)
    rec = AttentionRecorder(model.image_encoder)
    od = Path(paths["work_dir"]) / "results" / "interpretability"
    od.mkdir(parents=True, exist_ok=True)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    summary = {"arm": ic["arm"], "seed": ic["seed"], "splits": {}}
    for split in ("internal_test", "e1"):
        pred = pd.read_csv(Path(paths["work_dir"]) / "eval" / f"{ic['arm']}_s{ic['seed']}" / f"{split}.csv")
        sel = select_samples(pred, ic["n_correct_per_class"], ic["n_incorrect_per_class"], ic["sample_seed"])
        df, cohort = split_frame(paths, split, "evaluation")
        df = df.set_index("image_id").loc[sel["image_id"]].reset_index()
        full = pd.read_csv(Path(paths["work_dir"]) / "cohorts" / f"{cohort}.csv")["image_id"].tolist()
        mm = load_cache(paths["work_dir"], cohort, exp["image"]["cache_size"], full)
        meta_all = torch.from_numpy(encode_metadata(df)).to(device)
        aucs_ig, aucs_rand, items = [], [], []
        for i, r in df.iterrows():
            raw = torch.from_numpy(np.array(mm[r["row"]])).permute(2, 0, 1)[None].float().div(255).to(device)
            if raw.shape[-1] != exp["model"]["img_size"]:
                raw = F.interpolate(raw, size=(exp["model"]["img_size"],) * 2, mode="bilinear", align_corners=False)
            x = (raw - mean_t) / std_t
            meta = meta_all[i:i + 1]
            rec.clear()
            with torch.no_grad():
                logits = model(x, meta)["logits"]
            tgt = logits.argmax(-1)
            roll = attention_rollout(rec.maps)[0].cpu().numpy()
            xb = (blur(raw) - mean_t) / std_t
            ig = sum(integrated_gradients(model, x, meta, tgt, b, ic["ig_steps"]) for b in (torch.zeros_like(x) - mean_t / std_t, xb)) / 2
            ig_map = ig.abs().sum(1)[0]
            g = roll.shape[-1]
            patch = x.shape[-1] // g
            scores = F.avg_pool2d(ig_map[None, None], patch)[0, 0]
            fr = ic["deletion_fractions"]
            d_ig = deletion_curve(model, x, meta, int(tgt), scores, fr, xb, patch)
            rng = np.random.default_rng(ic["sample_seed"] + i)
            d_rand = np.mean([deletion_curve(model, x, meta, int(tgt), torch.from_numpy(rng.random((g, g))).to(device), fr, xb, patch)
                              for _ in range(5)], axis=0)
            aucs_ig.append(float(np.trapezoid(d_ig, fr)))
            aucs_rand.append(float(np.trapezoid(d_rand, fr)))
            np.savez_compressed(od / f"{split}_{r['image_id']}.npz", rollout=roll, ig=ig_map.cpu().numpy(), image=np.array(mm[r["row"]]),
                                deletion_ig=d_ig, deletion_random=d_rand)
            items.append({"image_id": r["image_id"], "label": r["label"], "pred": CLASSES[int(tgt)]})
        diff = np.array(aucs_rand) - np.array(aucs_ig)
        bs = np.random.default_rng(0).choice(len(diff), size=(2000, len(diff)))
        summary["splits"][split] = {"items": items, "deletion_auc_ig_mean": float(np.mean(aucs_ig)),
                                    "deletion_auc_random_mean": float(np.mean(aucs_rand)),
                                    "auc_diff_random_minus_ig_mean": float(diff.mean()),
                                    "auc_diff_ci95": np.percentile(diff[bs].mean(1), [2.5, 97.5]).tolist()}
    rec.remove()

    Hn = F.normalize(H.float(), dim=-1)
    P = model.prototypes().detach().cpu()
    summary["text_prototype_cosine_before"] = (Hn @ Hn.t()).numpy().round(4).tolist()
    summary["text_prototype_cosine_after"] = (P @ P.t()).numpy().round(4).tolist()

    q = np.load(Path(paths["work_dir"]) / "eval" / f"{ic['arm']}_s{ic['seed']}" / "e1_q.npy")
    e1 = pd.read_csv(Path(paths["work_dir"]) / "eval" / f"{ic['arm']}_s{ic['seed']}" / "e1.csv")
    rng = np.random.default_rng(ic["sample_seed"])
    n = min(ic["tsne_max_points"], len(e1))
    pick = np.sort(rng.choice(len(e1), size=n, replace=False))
    from sklearn.manifold import TSNE

    emb = TSNE(n_components=2, random_state=ic["sample_seed"], init="pca").fit_transform(q[pick])
    pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "label": e1["label"].to_numpy()[pick]}).to_csv(od / "tsne_e1.csv", index=False)
    write_json(od / "interpretability_summary.json", summary)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    run_interpret(load_paths(a.paths), Path(a.config_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
