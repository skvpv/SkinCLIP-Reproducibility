from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .. import CLASSES
from ..common import PKG_ROOT, load_paths, read_json

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
BLUES = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
INK, MUTED = "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def confusion_figure(paths, out, arm="F11", split="e1"):
    files = sorted(glob.glob(str(Path(paths["work_dir"]) / "eval" / f"{arm}_s*" / f"{split}.csv")))
    Cs = []
    for f in files:
        d = pd.read_csv(f)
        y, p = d["label_idx"].to_numpy(), d[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)
        C = np.bincount(y * 7 + p, minlength=49).reshape(7, 7).astype(float)
        Cs.append(C / np.maximum(C.sum(1, keepdims=True), 1))
    C = np.mean(Cs, 0)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(C, cmap=BLUES, vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if C[i, j] > 0.55 else INK)
    ax.set_xticks(range(7), CLASSES, rotation=45)
    ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{arm} on {split.upper()} (row-normalised, mean of {len(Cs)} seeds)", fontsize=9, color=INK)
    _save(fig, out, f"fig_confusion_{arm}_{split}")


def oot_figure(paths, out):
    st = read_json(Path(paths["work_dir"]) / "results" / "statistics.json")["oot"]
    tags = [t for t in st if t.startswith("F11")]
    if not tags:
        return
    dist = pd.DataFrame([st[t]["predicted_class_distribution"] for t in tags]).reindex(columns=CLASSES).fillna(0).mean()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.bar(CLASSES, dist.values, color=CAT[0], width=0.6)
    for i, v in enumerate(dist.values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_ylabel("Share of SCC images")
    ax.set_title("Predicted class of out-of-taxonomy SCC images (F11)", fontsize=9, color=INK)
    _save(fig, out, "fig_oot_scc")


def tradeoff_figure(paths, out):
    rd = Path(paths["work_dir"]) / "results"
    cost = pd.read_csv(rd / "table_cost.csv").set_index("arm")
    res = pd.read_csv(rd / "table_main_results.csv")
    r = res[(res.split == "e1") & (res.metric == "bma")].set_index("arm")
    arms = [a for a in r.index if a in cost.index]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for a in arms:
        lat = cost.loc[a, "latency_ms_bs1"] + (cost.loc["BiomedCLIP_visual", "latency_ms_bs1"] if a == "BCLP" else 0)
        ax.errorbar(lat, r.loc[a, "mean"], yerr=[[r.loc[a, "mean"] - r.loc[a, "ci_lo"]], [r.loc[a, "ci_hi"] - r.loc[a, "mean"]]],
                    fmt="o", color=CAT[0], ms=6, capsize=2, lw=1)
        ax.annotate(a, (lat, r.loc[a, "mean"]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("Inference latency, batch 1 (ms)")
    ax.set_ylabel("E1 balanced accuracy (95% CI)")
    _save(fig, out, "fig_tradeoff")


def tsne_figure(paths, out):
    f = Path(paths["work_dir"]) / "results" / "interpretability" / "tsne_e1.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for i, c in enumerate(CLASSES):
        s = d[d.label == c]
        ax.scatter(s.x, s.y, s=8, color=CAT[i], marker=MARKERS[i], label=c, linewidths=0, alpha=0.8)
    ax.legend(frameon=False, fontsize=7, markerscale=1.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE of F11 fused embeddings (E1 subsample)", fontsize=9, color=INK)
    _save(fig, out, "fig_tsne_e1")


def prototype_figure(paths, out):
    s = read_json(Path(paths["work_dir"]) / "results" / "interpretability" / "interpretability_summary.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, key, title in zip(axs, ("text_prototype_cosine_before", "text_prototype_cosine_after"),
                              ("Frozen Bio-ClinicalBERT prompts", "Trained text prototypes")):
        M = np.array(s[key])
        ax.imshow(M, cmap=BLUES, vmin=min(0, M.min()), vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(7), CLASSES, rotation=45)
        ax.set_yticks(range(7), CLASSES)
        ax.set_title(f"{title} (cosine)", fontsize=9, color=INK)
    _save(fig, out, "fig_text_prototypes")


def interpret_grid(paths, out, split="e1", max_rows=14):
    d = Path(paths["work_dir"]) / "results" / "interpretability"
    s = read_json(d / "interpretability_summary.json")["splits"].get(split, {})
    items = s.get("items", [])[:max_rows]
    if not items:
        return
    fig, axs = plt.subplots(len(items), 3, figsize=(5.4, 1.8 * len(items)))
    axs = np.atleast_2d(axs)
    for r, it in enumerate(items):
        z = np.load(d / f"{split}_{it['image_id']}.npz")
        img = z["image"]
        H = img.shape[0]
        roll = np.kron(z["rollout"], np.ones((H // z["rollout"].shape[0],) * 2))
        ig = z["ig"]
        ig = np.kron(ig, np.ones((H // ig.shape[0],) * 2)) if ig.shape[0] != H else ig
        for c, (title, ov) in enumerate((("Image", None), ("Attention rollout", roll), ("Integrated Gradients", ig))):
            ax = axs[r, c]
            ax.imshow(img)
            if ov is not None:
                ax.imshow(ov / (ov.max() + 1e-8), cmap=BLUES, alpha=0.55)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8, color=INK)
        axs[r, 0].set_ylabel(f"true {it['label']}\npred {it['pred']}", fontsize=7, color=INK)
    _save(fig, out, f"fig_interpretability_{split}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    paths = load_paths(a.paths)
    out = Path(paths["work_dir"]) / "results" / "figures"
    for fn in (confusion_figure, oot_figure, tradeoff_figure, tsne_figure, prototype_figure, interpret_grid):
        try:
            fn(paths, out)
            print(f"OK   {fn.__name__}")
        except FileNotFoundError as exc:
            print(f"SKIP {fn.__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
