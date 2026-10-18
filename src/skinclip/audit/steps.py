from __future__ import annotations

import hashlib
import io
import itertools
import multiprocessing as _mp
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..common import load_yaml, write_json
from ..data.sources import canonical_image_id, load_ham, load_isic2019


def load_all_sources(paths: dict) -> dict[str, pd.DataFrame]:
    raw = paths["raw"]
    out = {"HAM": load_ham(raw["ham_metadata_csv"], raw["ham_image_dirs"]),
           "ISIC2019_train": load_isic2019(raw["isic2019_train_gt_csv"], raw["isic2019_train_meta_csv"],
                                           raw["isic2019_train_image_dirs"], "ISIC2019_train")}
    if raw.get("isic2019_test_gt_csv"):
        out["ISIC2019_test"] = load_isic2019(raw["isic2019_test_gt_csv"], raw["isic2019_test_meta_csv"],
                                             raw["isic2019_test_image_dirs"], "ISIC2019_test")
    return out


def audit_dir(paths: dict) -> Path:
    d = Path(paths["work_dir"]) / "audit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _inspect(path: str | None) -> dict:
    if path is None:
        return {"readable": False, "error": "file not found"}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
        return {"readable": True, "width": w, "height": h, "mode": mode, "bytes": Path(path).stat().st_size}
    except Exception as exc:
        return {"readable": False, "error": f"{exc.__class__.__name__}: {exc}"}


def inventory_inventory(paths: dict, workers: int = 8) -> dict:
    srcs = load_all_sources(paths)
    rows = []
    for name, df in srcs.items():
        with _mp.get_context("spawn").Pool(max(1, int(workers))) as pool:
            info = pool.map(_inspect, df["path"].tolist(), chunksize=64)
        rows.append(pd.concat([df[["image_id", "source", "path"]].reset_index(drop=True), pd.DataFrame(info)], axis=1))
    inv = pd.concat(rows, ignore_index=True)
    inv.to_csv(audit_dir(paths) / "inventory_inventory.csv", index=False)
    summ = {}
    for name, g in inv.groupby("source"):
        ok = g.loc[g["readable"].astype(bool)]
        res = ok["width"].astype(int).astype(str) + "x" + ok["height"].astype(int).astype(str)
        summ[name] = {"n_rows": int(len(g)), "n_unreadable": int((~g["readable"].astype(bool)).sum()),
                      "resolution_counts_toinventory0": {k: int(v) for k, v in res.value_counts().head(10).items()}}
    write_json(audit_dir(paths) / "inventory_summary.json", summ)
    return summ


def lesion_prefix(lid) -> str:
    if lid is None or pd.isna(lid) or str(lid) in ("", "nan", "<NA>"):
        return "<none>"
    s = str(lid)
    pref = "".join(itertools.takewhile(lambda ch: ch.isalpha(), s))
    return pref or "<numeric>"


def provenance_id_lesion_overlap(paths: dict) -> dict:
    srcs = load_all_sources(paths)
    ham = srcs["HAM"]
    ham_ids = set(ham["image_id"])
    ham_lesions = set(ham["lesion_id"])
    out, summ = [], {}
    for name in [k for k in srcs if k != "HAM"]:
        df = srcs[name].copy()
        df["canonical_id"] = df["image_id"].map(canonical_image_id)
        df["id_match"] = df["canonical_id"].isin(ham_ids)
        df["lesion_match"] = df["lesion_id"].astype("string").isin(ham_lesions).fillna(False).astype(bool)
        df["lesion_prefix"] = df["lesion_id"].map(lesion_prefix)
        out.append(df[["image_id", "canonical_id", "source", "lesion_id", "lesion_prefix", "label_raw",
                       "id_match", "lesion_match"]])

        ct = pd.crosstab(df["lesion_prefix"], df["id_match"]).rename(columns={True: "id_match", False: "no_id_match"})
        summ[name] = {"n": int(len(df)), "n_id_match": int(df["id_match"].sum()),
                      "n_lesion_match": int(df["lesion_match"].sum()),
                      "n_lesion_match_without_id_match": int((df["lesion_match"] & ~df["id_match"]).sum()),
                      "prefix_vs_id_match": ct.to_dict(orient="index")}
    res = pd.concat(out, ignore_index=True)
    res.to_csv(audit_dir(paths) / "provenance_id_lesion_overlap.csv", index=False)
    write_json(audit_dir(paths) / "provenance_summary.json", summ)
    return summ


def _hash_one(args) -> dict:
    image_id, source, path = args
    import imagehash

    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb)
            md5 = hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()
            ph = str(imagehash.phash(rgb, hash_size=8))
            dh = str(imagehash.dhash(rgb, hash_size=8))
        return {"image_id": image_id, "source": source, "md5_pixels": md5, "phash": ph, "dhash": dh, "ok": True}
    except Exception as exc:
        return {"image_id": image_id, "source": source, "ok": False, "error": str(exc)}


def exact_overlap_hashes(paths: dict, workers: int = 8) -> dict:
    srcs = load_all_sources(paths)
    jobs = [(r.image_id, r.source, r.path) for df in srcs.values() for r in df.itertuples() if r.path]
    with _mp.get_context("spawn").Pool(max(1, int(workers))) as pool:
        recs = pool.map(_hash_one, jobs, chunksize=64)
    h = pd.DataFrame(recs)
    h.to_csv(audit_dir(paths) / "exact_overlap_hashes.csv", index=False)
    ham_md5 = set(h.loc[h.source == "HAM", "md5_pixels"])
    summ = {s: {"n": int((h.source == s).sum()), "n_failed": int(((h.source == s) & ~h.ok).sum()),
                "n_exact_pixel_match_HAM": int(h.loc[h.source == s, "md5_pixels"].isin(ham_md5).sum())}
            for s in h.source.unique() if s != "HAM"}

    hm = h[h.source == "HAM"].merge(srcs["HAM"][["image_id", "lesion_id"]], on="image_id")
    dup = hm.groupby("md5_pixels")["lesion_id"].nunique()
    summ["HAM_internal_exact_dups_across_lesions"] = int((dup > 1).sum())
    write_json(audit_dir(paths) / "exact_overlap_summary.json", summ)
    return summ


def class_counts_label_crosstab(paths: dict) -> dict:
    srcs = load_all_sources(paths)
    ham = srcs["HAM"][["image_id", "label_raw"]].rename(columns={"label_raw": "ham_dx"})
    summ = {}
    for name in [k for k in srcs if k != "HAM"]:
        df = srcs[name].assign(canonical_id=lambda d: d["image_id"].map(canonical_image_id))
        m = df.merge(ham, left_on="canonical_id", right_on="image_id", suffixes=("", "_ham"))
        ct = pd.crosstab(m["ham_dx"], m["label_raw"])
        ct.to_csv(audit_dir(paths) / f"class_counts_crosstab_HAMdx_vs_{name}.csv")
        counts = df["label_raw"].value_counts().to_dict()

        lid = df.dropna(subset=["lesion_id"])
        n_conflict = int((lid.groupby("lesion_id")["label_raw"].nunique() > 1).sum())
        summ[name] = {"class_counts": {k: int(v) for k, v in counts.items()},
                      "crosstab_HAMdx_vs_label": {r: {c: int(v) for c, v in row.items()} for r, row in ct.iterrows()},
                      "n_lesions_with_conflicting_labels": n_conflict}
    hl = srcs["HAM"].groupby("lesion_id")["label_raw"].nunique()
    summ["HAM_lesions_with_conflicting_labels"] = int((hl > 1).sum())
    write_json(audit_dir(paths) / "class_counts_summary.json", summ)
    return summ


def metadata_audit_lesion_structure(paths: dict) -> dict:
    srcs = load_all_sources(paths)
    ham = srcs["HAM"]
    per_les = ham.groupby("lesion_id").size()
    summ = {"HAM": {"n_images": int(len(ham)), "n_lesions": int(ham["lesion_id"].nunique()),
                    "images_per_lesion_counts": {int(k): int(v) for k, v in per_les.value_counts().sort_index().items()},
                    "lesions_per_class": {k: int(v) for k, v in ham.groupby("label_raw")["lesion_id"].nunique().items()},
                    "images_per_class": {k: int(v) for k, v in ham["label_raw"].value_counts().items()}}}
    for name in [k for k in srcs if k != "HAM"]:
        df = srcs[name]
        has = df["lesion_id"].notna() & ~df["lesion_id"].astype(str).isin(["", "nan", "<NA>"])
        summ[name] = {"n_images": int(len(df)), "n_with_lesion_id": int(has.sum()),
                      "n_lesions": int(df.loc[has, "lesion_id"].nunique()),
                      "lesion_id_coverage_by_prefix": df.assign(p=df["lesion_id"].map(lesion_prefix))
                      .groupby("p").size().astype(int).to_dict()}
    write_json(audit_dir(paths) / "metadata_audit_summary.json", summ)
    return summ


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def site_mapping_metadata(paths: dict, site_table_path: str | Path) -> dict:
    srcs = load_all_sources(paths)
    table = load_yaml(site_table_path)
    summ = {"site_table_status": table.get("status")}
    for name, df in srcs.items():
        mapping = table["sources"].get(name if name != "ISIC2019_test" else "ISIC2019_train", {})
        observed = df["site_raw"].fillna("nan").value_counts().to_dict()
        unmapped = sorted(set(observed) - set(mapping))
        summ[name] = {"site_values": {k: int(v) for k, v in observed.items()},
                      "unmapped_site_values": unmapped,
                      "sex_values": {k: int(v) for k, v in df["sex_raw"].fillna("nan").value_counts().items()},
                      "age_missing": int(df["age"].isna().sum()),
                      "cramers_v_site_x_label": cramers_v(df["site_raw"].fillna("nan"), df["label_raw"]),
                      "cramers_v_sex_x_label": cramers_v(df["sex_raw"].fillna("nan"), df["label_raw"]),
                      "cramers_v_age_missing_x_label": cramers_v(df["age"].isna(), df["label_raw"])}
    write_json(audit_dir(paths) / "site_mapping_summary.json", summ)
    return summ
