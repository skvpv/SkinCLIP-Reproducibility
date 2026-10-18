from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..common import read_json, write_json
from ..data.sources import canonical_image_id
from .steps import audit_dir, load_all_sources

_POcohort_build = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hex_to_u64(s: pd.Series) -> np.ndarray:
    return np.array([int(v, 16) for v in s], dtype=np.uint64)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    x = np.bitwise_xor(a[:, None].astype(np.uint64), b[None, :].astype(np.uint64))
    by = x.view(np.uint8).reshape(x.shape + (8,))
    return _POcohort_build[by].sum(axis=-1, dtype=np.uint16).astype(np.uint8)


def _threshold(neg: np.ndarray, fpr: float, higher_is_similar: bool) -> float:

    neg = np.asarray(neg, dtype=float)
    if higher_is_similar:
        c = float(np.quantile(neg, 1.0 - fpr, method="higher"))
        while (neg >= c).mean() > fpr:
            c = float(np.nextafter(c, np.inf))
        return c
    t = float(np.quantile(neg, fpr, method="lower"))
    while t >= 0 and (neg <= t).mean() > fpr:
        t -= 1.0
    return t


def compute_embeddings(paths: dict, model_name: str, device: str = "cuda", batch: int = 128,
                       pretrained: bool = True, img_size: int = 224) -> dict:

    import timm
    import torch

    srcs = load_all_sources(paths)
    df = pd.concat([s[["image_id", "source", "path"]] for s in srcs.values()], ignore_index=True).dropna(subset=["path"])
    kw = {"img_size": img_size} if img_size != 224 else {}
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **kw).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(df), batch):
            ims = [np.asarray(Image.open(p).convert("RGB").resize((img_size, img_size), Image.BILINEAR))
                   for p in df["path"].iloc[i:i + batch]]
            x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().div(255).to(device)
            f = model((x - mean) / std)
            feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype(np.float16))
    emb = np.concatenate(feats)
    d = audit_dir(paths)
    np.save(d / "image_embeddings_embeddings.npy", emb)
    df[["image_id", "source"]].to_csv(d / "image_embeddings_embedding_index.csv", index=False)
    summ = {"n": int(len(df)), "model": model_name, "pretrained": pretrained, "dim": int(emb.shape[1])}
    write_json(d / "image_embeddings_summary.json", summ)
    return summ


NULL_CLASS = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc",
              "SCC": "akiec", "UNK": None}


def similarity_screen_near_duplicates(paths: dict, fpr_strict: float = 5e-4, fpr_review: float = 5e-3, n_negative_pairs: int = 0,
                        seed: int = 20261013, chunk: int = 1024, max_review_rows: int = 3000,
                        use_embeddings: bool = True) -> dict:


    d = audit_dir(paths)
    srcs = load_all_sources(paths)
    hashes = pd.read_csv(d / "exact_overlap_hashes.csv", dtype={"phash": str, "dhash": str})
    hashes = hashes[hashes["ok"].astype(bool)].copy()
    for c in ("phash", "dhash"):
        hashes[c] = hex_to_u64(hashes[c])
    ov = pd.read_csv(d / "provenance_id_lesion_overlap.csv")
    ham = hashes[hashes.source == "HAM"].merge(srcs["HAM"][["image_id", "lesion_id", "label_raw"]], on="image_id").reset_index(drop=True)
    ham_md5 = set(ham["md5_pixels"])
    ham_cls = ham["label_raw"].to_numpy()
    emb = idx = None
    if use_embeddings and (d / "image_embeddings_embeddings.npy").exists():
        emb = np.load(d / "image_embeddings_embeddings.npy").astype(np.float32)
        idx = pd.read_csv(d / "image_embeddings_embedding_index.csv")
        idx["row"] = np.arange(len(idx))

    def emb_rows(frame, source):
        if emb is None:
            return None
        m = frame[["image_id"]].merge(idx[idx.source == source][["image_id", "row"]], on="image_id", how="left")
        if m["row"].isna().any():
            raise RuntimeError(f"embedding missing for some {source} images; rerun image_embeddings")
        return emb[m["row"].astype(int).to_numpy()]

    E_ham = emb_rows(ham, "HAM")
    summary, candidates = {"method": "per-image nearest-neighbour calibration; null = NN to different-diagnosis HAM images",
                           "fpr_strict_per_image": fpr_strict, "fpr_review_per_image": fpr_review}, []
    metrics = [("phash", False), ("dhash", False)] + ([("cosine", True)] if emb is not None else [])
    for name in [k for k in srcs if k != "HAM"]:
        t = hashes[hashes.source == name].merge(
            ov[ov.source == name][["image_id", "canonical_id", "id_match", "lesion_match", "label_raw"]], on="image_id").reset_index(drop=True)
        E_t = emb_rows(t, name)
        ham_pos = {iid: i for i, iid in enumerate(ham["image_id"])}

        pos_t = np.flatnonzero(t["id_match"].to_numpy())
        pos_h = np.array([ham_pos[c] for c in t["canonical_id"].iloc[pos_t]], dtype=int)

        def pair_vals(metric, ti, hi):
            if metric == "cosine":
                return (E_t[ti] * E_ham[hi]).sum(1)
            x = np.bitwise_xor(t[metric].to_numpy()[ti].astype(np.uint64), ham[metric].to_numpy()[hi].astype(np.uint64))
            return _POcohort_build[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


        elig = np.flatnonzero(~t["id_match"].to_numpy() & ~t["lesion_match"].to_numpy() & ~t["md5_pixels"].isin(ham_md5).to_numpy())
        nn_all = {m: np.empty(len(elig)) for m, _ in metrics}
        nn_arg = {m: np.empty(len(elig), dtype=int) for m, _ in metrics}
        nn_null = {m: np.empty(len(elig)) for m, _ in metrics}
        for s in range(0, len(elig), chunk):
            r = elig[s:s + chunk]
            null_cls = t["label_raw"].iloc[r].map(NULL_CLASS).to_numpy()
            diff = ham_cls[None, :] != null_cls[:, None]
            for m, hi in metrics:
                if m == "cosine":
                    M = E_t[r] @ E_ham.T
                else:
                    M = hamming_matrix(t[m].to_numpy()[r], ham[m].to_numpy()).astype(float)
                best = M.argmax(1) if hi else M.argmin(1)
                nn_arg[m][s:s + len(r)] = best
                nn_all[m][s:s + len(r)] = M[np.arange(len(r)), best]
                Mn = np.where(diff, M, -np.inf if hi else np.inf)
                nn_null[m][s:s + len(r)] = Mn.max(1) if hi else Mn.min(1)
        cal, strict, review = {}, np.zeros(len(elig), bool), np.zeros(len(elig), bool)
        null_strict, null_review = np.zeros(len(elig), bool), np.zeros(len(elig), bool)
        for m, hi in metrics:
            ts, tr = _threshold(nn_null[m], fpr_strict, hi), _threshold(nn_null[m], fpr_review, hi)
            p = pair_vals(m, pos_t, pos_h) if len(pos_t) else np.array([])
            ok_s = (nn_all[m] >= ts) if hi else (nn_all[m] <= ts)
            ok_r = (nn_all[m] >= tr) if hi else (nn_all[m] <= tr)
            strict |= ok_s
            review |= ok_r
            null_strict |= (nn_null[m] >= ts) if hi else (nn_null[m] <= ts)
            null_review |= (nn_null[m] >= tr) if hi else (nn_null[m] <= tr)
            cal[m] = {"strict": ts, "review": tr, "direction": ">=" if hi else "<=",
                      "sensitivity_strict_on_ID_pairs": float(((p >= ts) if hi else (p <= ts)).mean()) if len(p) else None,
                      "sensitivity_review_on_ID_pairs": float(((p >= tr) if hi else (p <= tr)).mean()) if len(p) else None,
                      "ID_pair_value_quantiles": np.quantile(p, [0, .5, .99, 1]).tolist() if len(p) else None}
        for k, i in enumerate(elig):
            if not review[k]:
                continue
            for m, _ in metrics:
                j = nn_arg[m][k]
                candidates.append({"source": name, "image_id": t["image_id"].iloc[i], "ham_image_id": ham["image_id"].iloc[j],
                                   "ham_lesion_id": ham["lesion_id"].iloc[j], "metric": m, "value": float(nn_all[m][k]),
                                   "tier": "AUTO_EXCLUDE" if strict[k] else "REVIEW"})
        summary[name] = {"calibration": cal, "n_ID_pairs": int(len(pos_t)), "n_eligible_searched": int(len(elig)),
                         "null_union_flag_rate_strict": float(null_strict.mean()) if len(elig) else None,
                         "null_union_flag_rate_review": float(null_review.mean()) if len(elig) else None,
                         "n_images_auto_exclude": int(strict.sum()), "n_images_review": int((review & ~strict).sum())}
    cand = pd.DataFrame(candidates, columns=["source", "image_id", "ham_image_id", "ham_lesion_id", "metric", "value", "tier"])
    cand.to_csv(d / "similarity_screen_near_duplicate_candidates.csv", index=False)

    review = cand[cand.tier == "REVIEW"].drop_duplicates(["source", "image_id", "ham_image_id"]).copy()
    review["decision"] = ""
    rdir = Path(paths["work_dir"]) / "review"
    rdir.mkdir(parents=True, exist_ok=True)
    review.to_csv(rdir / "near_duplicates_TO_ADJUDICATE.csv", index=False)
    _contact_sheet(review.head(max_review_rows), srcs, rdir / "near_duplicates_review.html")
    summary["review_rows"] = int(len(review))
    summary["review_rows_truncated_in_html"] = bool(len(review) > max_review_rows)
    write_json(d / "similarity_screen_summary.json", summary)
    return summary


def _thumb(path: str, size: int = 160) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _contact_sheet(review: pd.DataFrame, srcs: dict, out: Path) -> None:
    path_of = {(s, r.image_id): r.path for s, df in srcs.items() for r in df.itertuples()}
    rows = []
    for r in review.itertuples():
        a, b = path_of.get((r.source, r.image_id)), path_of.get(("HAM", r.ham_image_id))
        if not a or not b:
            continue
        rows.append(f"<tr><td>{r.image_id}<br><img src='data:image/jpeg;base64,{_thumb(a)}'></td>"
                    f"<td>{r.ham_image_id}<br><img src='data:image/jpeg;base64,{_thumb(b)}'></td>"
                    f"<td>{r.metric} = {r.value:.4g}</td></tr>")
    out.write_text("<html><meta charset='utf-8'><body><h3>Near-duplicate review (decide in "
                   "near_duplicates_TO_ADJUDICATE.csv)</h3><table border=1>" + "".join(rows) + "</table></body></html>")


def ham_internal_links(paths: dict) -> pd.DataFrame:

    d = audit_dir(paths)
    srcs = load_all_sources(paths)
    h = pd.read_csv(d / "exact_overlap_hashes.csv", dtype={"phash": str, "dhash": str})
    h = h[(h.source == "HAM") & h.ok.astype(bool)].merge(srcs["HAM"][["image_id", "lesion_id"]], on="image_id").reset_index(drop=True)
    links = []
    for _, g in h.groupby("md5_pixels"):
        if g["lesion_id"].nunique() > 1:
            ids = g["image_id"].tolist()
            links += [(ids[0], x, "md5") for x in ids[1:]]
    cal = None
    similarity_screen = d / "similarity_screen_summary.json"
    if similarity_screen.exists():
        import json

        s = json.loads(similarity_screen.read_text())
        cal = s.get("ISIC2019_train", {}).get("calibration", {}).get("phash", {}).get("strict")
    if cal is not None:
        ph = hex_to_u64(h["phash"])
        les = h["lesion_id"].to_numpy()
        for s0 in range(0, len(h), 2048):
            D = hamming_matrix(ph[s0:s0 + 2048], ph)
            for i, j in zip(*np.nonzero(D <= cal)):
                ii = s0 + i
                if ii < j and les[ii] != les[j]:
                    links.append((h["image_id"].iloc[ii], h["image_id"].iloc[j], "phash_strict"))
    out = pd.DataFrame(links, columns=["image_a", "image_b", "method"])
    out.to_csv(d / "within_dataset_duplicates_HAM_internal_links.csv", index=False)
    return out


def label_free_screen_label_free_null(paths: dict, fpr_strict: float = 5e-4, fpr_review: float = 5e-3, chunk: int = 1024,
                        use_embeddings: bool = True) -> dict:


    d = audit_dir(paths)
    srcs = load_all_sources(paths)
    hashes = pd.read_csv(d / "exact_overlap_hashes.csv", dtype={"phash": str, "dhash": str})
    hashes = hashes[hashes["ok"].astype(bool)].copy()
    for c in ("phash", "dhash"):
        hashes[c] = hex_to_u64(hashes[c])
    ov = pd.read_csv(d / "provenance_id_lesion_overlap.csv")
    ham = hashes[hashes.source == "HAM"].merge(srcs["HAM"][["image_id", "lesion_id"]], on="image_id").reset_index(drop=True)
    ham_md5 = set(ham["md5_pixels"])
    ham_les = ham["lesion_id"].to_numpy()
    emb = None
    if use_embeddings and (d / "image_embeddings_embeddings.npy").exists():
        emb = np.load(d / "image_embeddings_embeddings.npy").astype(np.float32)
        idx = pd.read_csv(d / "image_embeddings_embedding_index.csv")
        idx["row"] = np.arange(len(idx))

    def rows(frame, source):
        m = frame[["image_id"]].merge(idx[idx.source == source][["image_id", "row"]], on="image_id", how="left")
        return emb[m["row"].astype(int).to_numpy()]

    metrics = [("phash", False), ("dhash", False)] + ([("cosine", True)] if emb is not None else [])
    E_ham = rows(ham, "HAM") if emb is not None else None

    def sim(A_frame, A_emb, r, m):
        if m == "cosine":
            return A_emb[r] @ E_ham.T
        return hamming_matrix(A_frame[m].to_numpy()[r], ham[m].to_numpy()).astype(float)


    null = {m: np.empty(len(ham)) for m, _ in metrics}
    for s in range(0, len(ham), chunk):
        r = np.arange(s, min(s + chunk, len(ham)))
        same = ham_les[r][:, None] == ham_les[None, :]
        for m, hi in metrics:
            M = sim(ham, E_ham, r, m)
            M = np.where(same, -np.inf if hi else np.inf, M)
            null[m][r] = M.max(1) if hi else M.min(1)
    res = {"method": "label-free null: HAM nearest neighbour among different-lesion HAM images",
           "fpr_strict_per_image": fpr_strict, "fpr_review_per_image": fpr_review}
    cand = pd.read_csv(d / "similarity_screen_near_duplicate_candidates.csv")
    lf_rows = []
    for name in [k for k in srcs if k != "HAM"]:
        t = hashes[hashes.source == name].merge(ov[ov.source == name][["image_id", "id_match", "lesion_match", "label_raw"]],
                                                on="image_id").reset_index(drop=True)
        elig = np.flatnonzero(~t["id_match"].to_numpy() & ~t["lesion_match"].to_numpy() & ~t["md5_pixels"].isin(ham_md5).to_numpy())
        E_t = rows(t, name) if emb is not None else None
        strict = np.zeros(len(elig), bool)
        review = np.zeros(len(elig), bool)
        thr = {}
        for m, hi in metrics:
            nn = np.empty(len(elig))
            for s in range(0, len(elig), chunk):
                r = elig[s:s + chunk]
                M = sim(t, E_t, r, m)
                nn[s:s + len(r)] = M.max(1) if hi else M.min(1)
            ts, tr = _threshold(null[m], fpr_strict, hi), _threshold(null[m], fpr_review, hi)
            thr[m] = {"strict": ts, "review": tr}
            strict |= (nn >= ts) if hi else (nn <= ts)
            review |= (nn >= tr) if hi else (nn <= tr)
        lf = set(t["image_id"].iloc[elig[review]])
        precautionary = set(cand.loc[cand.source == name, "image_id"])
        only_lf = sorted(lf - precautionary)
        lab = t.set_index("image_id")["label_raw"]
        res[name] = {"thresholds_label_free": thr,
                     "thresholds_label_informed_similarity_screen": read_json(d / "similarity_screen_summary.json").get(name, {}).get("calibration", {}),
                     "n_eligible": int(len(elig)), "n_flagged_label_free_review_or_strict": len(lf),
                     "n_flagged_label_free_strict": int(strict.sum()), "n_flagged_PX1": len(precautionary),
                     "n_both": len(lf & precautionary), "n_PX1_only": len(precautionary - lf), "n_label_free_only": len(only_lf),
                     "label_free_only_by_raw_label": lab.loc[only_lf].value_counts().to_dict() if only_lf else {}}
        lf_rows += [{"source": name, "image_id": i, "label_raw": lab.loc[i]} for i in only_lf]
    res["summary"] = {k: {kk: v[kk] for kk in ("n_flagged_PX1", "n_flagged_label_free_review_or_strict", "n_both",
                                                "n_PX1_only", "n_label_free_only")} for k, v in res.items() if isinstance(v, dict) and "n_both" in v}
    pd.DataFrame(lf_rows, columns=["source", "image_id", "label_raw"]).to_csv(d / "label_free_screen_label_free_only.csv", index=False)
    write_json(d / "label_free_screen_label_free_null.json", res)
    return res
