from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import CLASSES
from ..common import PKG_ROOT, load_paths, load_yaml, now_iso, read_json, write_json
from .metrics import classification_metrics, confusion, from_confusion

K = len(CLASSES)
PCOLS = [f"p_{c}" for c in CLASSES]


def load_preds(paths, tag: str, split: str) -> pd.DataFrame:
    return pd.read_csv(Path(paths["work_dir"]) / "eval" / tag / f"{split}.csv", dtype={"cluster_id": str})


def run_tags(paths, arm: str) -> list[str]:
    checkpoint = read_json(Path(paths["work_dir"]) / "integrity" / "checkpoint_manifest.json")
    if arm == "BCZS":
        return ["BCZS"]
    return [f"{r['arm']}_s{r['seed']}" for r in checkpoint["runs"] if r["arm"] == arm]


def cluster_weights(df: pd.DataFrame, B: int, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(df["cluster_id"])
    clabel = df.groupby(codes)["label_idx"].agg(lambda s: s.value_counts().index[0]).to_numpy()
    W = np.zeros((B, len(df)), dtype=np.float32)
    for c in np.unique(clabel):
        cl = np.flatnonzero(clabel == c)
        draws = rng.integers(0, len(cl), size=(B, len(cl)))
        cnt = np.zeros((B, len(uniq)), dtype=np.float32)
        np.add.at(cnt, (np.repeat(np.arange(B), len(cl)), cl[draws].ravel()), 1.0)
        W += cnt[:, codes]
    return W


def _metric(y, pred, w, name):
    return from_confusion(confusion(y, pred, K, w))[name]


def summarise_arm(preds: list[pd.DataFrame], W: np.ndarray, metrics=("bma", "macro_f1", "accuracy"), auroc_ci=False) -> dict:
    y = preds[0]["label_idx"].to_numpy()
    out = {}
    per_seed = [classification_metrics(y, p[PCOLS].to_numpy()) for p in preds]
    for m in list(metrics) + ["macro_auroc"]:
        vals = np.array([s[m] for s in per_seed])
        rec = {"per_seed": vals.tolist(), "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
        if m != "macro_auroc" or auroc_ci:
            boots = np.array([np.mean([_boot(y, p, w, m) for p in preds]) for w in W])
            rec["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out[m] = rec
    out["per_class_mean"] = {k: np.mean([s[k] for s in per_seed], axis=0).tolist() for k in ("recall", "specificity", "f1")}
    return out


def _boot(y, p, w, m):
    if m == "macro_auroc":
        return classification_metrics(y, p[PCOLS].to_numpy(), w, auroc=True)["macro_auroc"]
    return _metric(y, p[PCOLS].to_numpy().argmax(1), w, m)


def assert_prediction_alignment(reference, candidate):
    for column in ("image_id", "cluster_id", "label_idx"):
        if column not in reference or column not in candidate:
            raise ValueError(f"prediction table missing {column}")
        if not np.array_equal(reference[column].astype(str).to_numpy(), candidate[column].astype(str).to_numpy()):
            raise ValueError(f"prediction tables are not aligned by {column}")
    if reference["image_id"].duplicated().any() or candidate["image_id"].duplicated().any():
        raise ValueError("prediction tables contain duplicate image identifiers")


def paired(predsA, predsB, W, metric="bma") -> dict:
    y = predsA[0]["label_idx"].to_numpy()
    for p in predsA + predsB:
        assert_prediction_alignment(predsA[0], p)
        if not np.array_equal(p["label_idx"].to_numpy(), y):
            raise ValueError("paired comparison requires identical image order")
    pA = [p[PCOLS].to_numpy().argmax(1) for p in predsA]
    pB = [p[PCOLS].to_numpy().argmax(1) for p in predsB]
    point = np.mean([_metric(y, a, None, metric) for a in pA]) - np.mean([_metric(y, b, None, metric) for b in pB])
    d = np.array([np.mean([_metric(y, a, w, metric) for a in pA]) - np.mean([_metric(y, b, w, metric) for b in pB]) for w in W])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"delta": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_boot": p}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def mcnemar_one_per_cluster(predA: pd.DataFrame, predB: pd.DataFrame, seed: int) -> dict:
    from scipy.stats import binomtest

    assert_prediction_alignment(predA, predB)
    rng = np.random.default_rng(seed)
    idx = predA.groupby("cluster_id").indices
    pick = np.array([rng.choice(v) for _, v in sorted(idx.items())])
    y = predA["label_idx"].to_numpy()[pick]
    a = predA[PCOLS].to_numpy()[pick].argmax(1) == y
    b = predB[PCOLS].to_numpy()[pick].argmax(1) == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"n_clusters": int(len(pick)), "A_correct_B_wrong": n01, "A_wrong_B_correct": n10, "p_exact": float(p)}


def oot_analysis(paths, tag_list: list[str]) -> dict:
    from sklearn.metrics import roc_auc_score

    res = {}
    for tag in tag_list:
        oot = load_preds(paths, tag, "e1_oot")
        ind = load_preds(paths, tag, "e1")
        if len(oot) == 0:
            continue
        dist = pd.Series(np.array(CLASSES)[oot[PCOLS].to_numpy().argmax(1)]).value_counts(normalize=True).to_dict()
        msp = np.r_[ind[PCOLS].max(axis=1).to_numpy(), oot[PCOLS].max(axis=1).to_numpy()]
        lab = np.r_[np.ones(len(ind)), np.zeros(len(oot))]
        res[tag] = {"n_oot": int(len(oot)), "oot_labels": oot["label"].value_counts().to_dict(),
                    "predicted_class_distribution": dist, "msp_auroc_in_vs_oot": float(roc_auc_score(lab, msp))}
    return res


def run_stats(paths: dict, config_dir: Path) -> dict:
    exp = load_yaml(config_dir / "experiment.yaml")
    st = exp["stats"]
    arms = ["F11", "F01", "F10", "F00", "MB", "BCLP", "BCZS", "M0"]
    out = {"time": now_iso(), "splits": {}, "design": st}
    for split in ("internal_test", "e1"):
        base = load_preds(paths, run_tags(paths, "F11")[0], split)
        W = cluster_weights(base, st["n_boot"], st["boot_seed"])
        preds = {a: [load_preds(paths, t, split) for t in run_tags(paths, a)] for a in arms}
        for arm_predictions in preds.values():
            for prediction in arm_predictions:
                assert_prediction_alignment(base, prediction)
        S = {"n_images": int(len(base)), "n_clusters": int(base["cluster_id"].nunique()),
             "arms": {a: summarise_arm(p, W, auroc_ci=st.get("auroc_ci", False)) for a, p in preds.items() if p}}
        conf = {h: paired(preds[a], preds[b], W, st["primary_metric"]) for h, (a, b) in st["confirmatory"].items()}
        adj = holm({h: v["p_boot"] for h, v in conf.items()})
        for h in conf:
            conf[h]["p_holm"] = adj[h]
        S["confirmatory"] = conf
        S["supporting"] = {h: paired(preds[a], preds[b], W, st["primary_metric"]) for h, (a, b) in st["supporting"].items()}
        S["descriptive"] = {f"{a}_vs_{b}": paired(preds[a], preds[b], W, st["primary_metric"]) for a, b in st["descriptive"]}
        S["mcnemar"] = {h: [mcnemar_one_per_cluster(pa, pb, st["mcnemar_seed"]) for pa, pb in zip(preds[a], preds[b])]
                        for h, (a, b) in st["confirmatory"].items()}
        if split == "e1":
            S["subgroups"] = subgroups(preds, ["F11", "F00", "F01"])
        out["splits"][split] = S
    out["oot"] = oot_analysis(paths, run_tags(paths, "F11") + run_tags(paths, "F00"))
    rd = Path(paths["work_dir"]) / "results"
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "statistics.json", out)
    tables(out, rd)
    return out


def subgroups(preds: dict, arms: list[str]) -> dict:
    res = {}
    base = preds[arms[0]][0]
    groups = {"institution": base["institution"].astype(str),
              "metadata_complete": (~base["age"].isna() & base["sex_raw"].isin(["male", "female"]) &
                                    (base["site_group"] != "unknown")).map({True: "complete", False: "any_missing"})}
    for gname, g in groups.items():
        for val in sorted(g.unique()):
            m = (g == val).to_numpy()
            if m.sum() < 30:
                continue
            res[f"{gname}={val}"] = {a: float(np.mean([classification_metrics(p["label_idx"].to_numpy()[m],
                                                                               p[PCOLS].to_numpy()[m], auroc=False)["bma"]
                                                        for p in preds[a]])) for a in arms}
            res[f"{gname}={val}"]["n"] = int(m.sum())
    return res


def tables(out: dict, rd: Path) -> None:
    rows = []
    for split, S in out["splits"].items():
        for arm, A in S["arms"].items():
            for m in ("bma", "macro_f1", "accuracy", "macro_auroc"):
                r = A[m]
                rows.append({"split": split, "arm": arm, "metric": m, "mean": r["mean"], "sd": r["sd"],
                             "ci_lo": r.get("ci95", [np.nan, np.nan])[0], "ci_hi": r.get("ci95", [np.nan, np.nan])[1]})
    pd.DataFrame(rows).to_csv(rd / "table_main_results.csv", index=False)
    hy = [{"split": s, "contrast": h, **{k: v for k, v in d.items() if k != "ci95"}, "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1]}
          for s, S in out["splits"].items() for grp in ("confirmatory", "supporting", "descriptive") for h, d in S[grp].items()]
    pd.DataFrame(hy).to_csv(rd / "table_contrasts.csv", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    run_stats(load_paths(a.paths), Path(a.config_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
