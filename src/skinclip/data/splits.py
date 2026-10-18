from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "internal_test")


class _UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def split_groups(df: pd.DataFrame, links: pd.DataFrame | None = None) -> pd.Series:
    uf = _UF(df["lesion_id"].unique().tolist())
    if links is not None and len(links):
        les = dict(zip(df["image_id"], df["lesion_id"]))
        for a, b in zip(links["image_a"], links["image_b"]):
            if a in les and b in les:
                uf.union(les[a], les[b])
    return df["lesion_id"].map(uf.find).rename("split_group")


def stratified_group_split(df: pd.DataFrame, label_col: str, group_col: str, ratios=(0.70, 0.15, 0.15),
                           seed: int = 20261013) -> tuple[pd.Series, dict]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1")
    rng = np.random.default_rng(seed)
    g_label = df.groupby(group_col)[label_col].agg(lambda s: s.value_counts().index[0])
    g_mixed = df.groupby(group_col)[label_col].nunique()
    g_size = df.groupby(group_col).size()
    assign: dict = {}
    for cls in sorted(g_label.unique()):
        groups = np.array(sorted(g_label.index[g_label == cls]))
        rng.shuffle(groups)
        total = g_size[groups].sum()
        target = np.array(ratios) * total
        cur = np.zeros(3)
        for g in groups:
            k = int(np.argmax(target - cur))
            assign[g] = SPLIT_NAMES[k]
            cur[k] += g_size[g]
    split = df[group_col].map(assign).rename("split")
    report = {"mixed_label_groups": int((g_mixed > 1).sum()),
              "images_per_split_class": df.assign(split=split).groupby(["split", label_col]).size().unstack(fill_value=0)
              .to_dict(orient="index")}
    return split, report


def class_presence_probability(labels: pd.Series, batch_size: int) -> dict:

    pi = labels.value_counts(normalize=True)
    return {k: float(1 - (1 - v) ** batch_size) for k, v in pi.items()}
