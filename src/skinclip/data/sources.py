from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ISIC_LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
IMG_EXT = (".jpg", ".jpeg", ".png")


class SchemaError(ValueError):
    pass


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}; found {list(df.columns)}")


def index_images(dirs: list[str | Path]) -> dict[str, Path]:

    out: dict[str, Path] = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            raise FileNotFoundError(f"image directory not found: {d}")
        for p in d.rglob("*"):
            if p.suffix.lower() in IMG_EXT and p.is_file():
                if p.stem in out and out[p.stem] != p:
                    raise SchemaError(f"duplicate image stem {p.stem}: {out[p.stem]} and {p}")
                out[p.stem] = p
    return out


def _attach_paths(df: pd.DataFrame, img_index: dict[str, Path], name: str) -> pd.DataFrame:
    df = df.copy()
    df["path"] = df["image_id"].map(lambda s: str(img_index[s]) if s in img_index else None)
    n_missing = int(df["path"].isna().sum())
    df.attrs["n_missing_images"] = n_missing
    df.attrs["source_name"] = name
    return df


def load_ham(metadata_csv: str | Path, image_dirs: list[str | Path]) -> pd.DataFrame:
    raw = pd.read_csv(metadata_csv)
    _require(raw, ["lesion_id", "image_id", "dx", "age", "sex", "localization"], "HAM10000 metadata")
    df = pd.DataFrame({
        "image_id": raw["image_id"].astype(str),
        "source": "HAM",
        "lesion_id": raw["lesion_id"].astype(str),
        "label_raw": raw["dx"].astype(str).str.lower(),
        "age": pd.to_numeric(raw["age"], errors="coerce"),
        "sex_raw": raw["sex"].astype(str).str.lower(),
        "site_raw": raw["localization"].astype(str).str.lower(),
        "dx_type": raw.get("dx_type", pd.Series([None] * len(raw))),
    })
    if df["image_id"].duplicated().any():
        raise SchemaError("HAM10000: duplicated image_id")
    return _attach_paths(df, index_images(image_dirs), "HAM")


def _onehot_to_label(gt: pd.DataFrame, name: str) -> pd.Series:
    _require(gt, ["image"] + ISIC_LABEL_COLS, name)
    m = gt[ISIC_LABEL_COLS].to_numpy(dtype=float)
    rowsum = m.sum(axis=1)
    if not np.allclose(rowsum, 1.0):
        bad = gt.loc[~np.isclose(rowsum, 1.0), "image"].head(5).tolist()
        raise SchemaError(f"{name}: ground truth rows not one-hot, e.g. {bad}")
    return pd.Series(np.array(ISIC_LABEL_COLS)[m.argmax(axis=1)], index=gt.index)


def load_isic2019(gt_csv: str | Path, meta_csv: str | Path, image_dirs: list[str | Path],
                  source: str = "ISIC2019_train") -> pd.DataFrame:
    gt = pd.read_csv(gt_csv)
    meta = pd.read_csv(meta_csv)
    name = f"{source} ground truth"
    gt = gt.assign(label_raw=_onehot_to_label(gt, name))
    _require(meta, ["image", "age_approx", "anatom_site_general", "sex"], f"{source} metadata")
    if "lesion_id" not in meta.columns:
        meta = meta.assign(lesion_id=pd.NA)
    merged = gt[["image", "label_raw"]].merge(meta, on="image", how="left", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise SchemaError(f"{source}: {(merged['_merge'] != 'both').sum()} ground-truth rows lack metadata")
    df = pd.DataFrame({
        "image_id": merged["image"].astype(str),
        "source": source,
        "lesion_id": merged["lesion_id"].astype("string"),
        "label_raw": merged["label_raw"].astype(str),
        "age": pd.to_numeric(merged["age_approx"], errors="coerce"),
        "sex_raw": merged["sex"].astype("string").str.lower().fillna("nan"),
        "site_raw": merged["anatom_site_general"].astype("string").str.lower().fillna("nan"),
    })
    return _attach_paths(df, index_images(image_dirs), source)


def load_ham_lesion_groupings(csv: str | Path | None) -> pd.DataFrame | None:

    if csv is None or not Path(csv).exists():
        return None
    g = pd.read_csv(csv)
    _require(g, ["image", "lesion_id"], "ISIC2018 lesion groupings")
    return g


def canonical_image_id(image_id: str) -> str:

    return image_id[:-len("_downsampled")] if image_id.endswith("_downsampled") else image_id
