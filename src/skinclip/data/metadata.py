from __future__ import annotations

import numpy as np
import pandas as pd

SITE_GROUPS = ["head_neck", "torso", "upper_extremity", "lower_extremity", "palms_soles", "oral_genital", "unknown"]
META_DIM = 5 + len(SITE_GROUPS)


class UnmappedSiteError(ValueError):
    pass


def harmonise_sites(df: pd.DataFrame, site_table: dict) -> pd.Series:

    if site_table.get("groups") != SITE_GROUPS:
        raise ValueError(f"site table groups {site_table.get('groups')} != {SITE_GROUPS}")
    out = pd.Series(index=df.index, dtype=object)
    for src, g in df.groupby("source"):
        key = "ISIC2019_train" if src == "ISIC2019_test" else src
        mapping = site_table["sources"][key]
        raw = g["site_raw"].fillna("nan").astype(str)
        bad = sorted(set(raw) - set(mapping))
        if bad:
            raise UnmappedSiteError(f"{src}: unmapped site values {bad} (update config/site_harmonisation.yaml at cohort)")
        mapped = raw.map(mapping)
        if not mapped.isin(SITE_GROUPS).all():
            raise ValueError(f"{src}: site table maps to unknown group(s) {sorted(set(mapped) - set(SITE_GROUPS))}")
        out.loc[g.index] = mapped
    return out


def encode_metadata(df: pd.DataFrame) -> np.ndarray:

    n = len(df)
    x = np.zeros((n, META_DIM), dtype=np.float32)
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype=float)
    miss = np.isnan(age)
    x[:, 0] = np.where(miss, 0.0, age / 100.0)
    x[:, 1] = miss.astype(np.float32)
    sex = df["sex_raw"].astype(str).str.lower().to_numpy()
    x[:, 2] = sex == "male"
    x[:, 3] = sex == "female"
    x[:, 4] = ~np.isin(sex, ["male", "female"])
    site_idx = df["site_group"].map({g: i for i, g in enumerate(SITE_GROUPS)}).to_numpy()
    if pd.isna(site_idx).any():
        raise ValueError("site_group missing or invalid; run harmonise_sites first")
    x[np.arange(n), 5 + site_idx.astype(int)] = 1.0
    return x
