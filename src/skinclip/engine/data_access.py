from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common import EVAL_SPLITS, TRAIN_SPLITS, assert_split_access, require_cohort
from ..data.cache import load_cache
from ..data.dataset import LesionDataset, build_augment
from ..data.metadata import encode_metadata

_SPLIT_TO_COHORT = {"train": "ham", "val": "ham", "internal_test": "ham", "e1": "e1", "e1_oot": "e1_oot", "e2": "e2",
                    "e2_oot": "e2_oot"}


def split_frame(paths: dict, split: str, purpose: str) -> tuple[pd.DataFrame, str]:

    assert_split_access(split, purpose, paths)
    require_cohort(paths)
    cohort = _SPLIT_TO_COHORT[split]
    df = pd.read_csv(Path(paths["work_dir"]) / "cohorts" / f"{cohort}.csv", dtype={"lesion_id": str})
    df["row"] = np.arange(len(df))
    if cohort == "ham":
        df = df[df["split"] == split].copy()
    return df.reset_index(drop=True), cohort


def make_dataset(paths: dict, exp: dict, split: str, purpose: str, mean, std, with_images: bool, train_aug: bool,
                 features: np.ndarray | None = None) -> tuple[LesionDataset, pd.DataFrame]:
    df, cohort = split_frame(paths, split, purpose)
    images = None
    if with_images:
        full_ids = pd.read_csv(Path(paths["work_dir"]) / "cohorts" / f"{cohort}.csv")["image_id"].tolist()
        mm = load_cache(paths["work_dir"], cohort, exp["image"]["cache_size"], full_ids)
        images = _RowView(mm, df["row"].to_numpy())
    labels = df["label_idx"].to_numpy()
    meta = encode_metadata(df)
    aug = build_augment(exp["augment"]) if train_aug else None
    out_size = exp["model"]["img_size"] if exp["model"]["img_size"] != exp["image"]["cache_size"] else None
    return LesionDataset(images, meta, labels, mean, std, aug, out_size, features), df


class _RowView:


    def __init__(self, mm, rows):
        self.mm, self.rows = mm, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.mm[self.rows[i]]
