from __future__ import annotations

import multiprocessing as _mp
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..common import sha256_obj, write_json

_INTERP = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def preprocess_image(path: str, size: int, clip: float, tile: int, interp: str = "area") -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    bgr = cv2.resize(bgr, (size, size), interpolation=_INTERP[interp])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _work(args):
    mm_path, shape, rows, paths, size, clip, tile, interp = args
    mm = np.lib.format.open_memmap(mm_path, mode="r+")
    for r, p in zip(rows, paths):
        mm[r] = preprocess_image(p, size, clip, tile, interp)
    mm.flush()
    return len(rows)


def cache_paths(work_dir: str | Path, cohort: str, size: int) -> tuple[Path, Path, Path]:
    d = Path(work_dir) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{cohort}_{size}.npy", d / f"{cohort}_{size}.index.csv", d / f"{cohort}_{size}.meta.json"


def build_cache(df: pd.DataFrame, work_dir: str | Path, cohort: str, image_cfg: dict, workers: int = 8,
                chunk: int = 256) -> Path:
    size = int(image_cfg["cache_size"])
    npy, idx, meta = cache_paths(work_dir, cohort, size)
    params = {"size": size, "clip": image_cfg["clahe_clip_limit"], "tile": image_cfg["clahe_tile_grid"],
              "interp": image_cfg["resize_interpolation"], "image_ids": df["image_id"].tolist()}
    sig = sha256_obj(params)
    if meta.exists() and npy.exists():
        import json

        if json.loads(meta.read_text()).get("signature") == sig:
            return npy
    mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(len(df), size, size, 3))
    del mm
    jobs = [(str(npy), None, list(range(s, min(s + chunk, len(df)))), df["path"].iloc[s:s + chunk].tolist(), size,
             params["clip"], params["tile"], params["interp"]) for s in range(0, len(df), chunk)]
    with _mp.get_context("spawn").Pool(max(1, int(workers))) as pool:
        done = sum(pool.map(_work, jobs))
    assert done == len(df)
    df[["image_id"]].to_csv(idx, index=False)
    write_json(meta, {"signature": sig, "n": len(df), **{k: v for k, v in params.items() if k != "image_ids"}})
    return npy


def load_cache(work_dir: str | Path, cohort: str, size: int, expected_ids: list[str]) -> np.ndarray:
    npy, idx, _ = cache_paths(work_dir, cohort, size)
    ids = pd.read_csv(idx)["image_id"].tolist()
    if ids != list(expected_ids):
        raise RuntimeError(f"cache row order for {cohort} does not match the cohort file; rebuild the cache")
    return np.load(npy, mmap_mode="r")
