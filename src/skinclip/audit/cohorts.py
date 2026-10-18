from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .. import CLASSES
from ..common import (IntegrityError, PKG_ROOT, integrity_dir, load_yaml, now_iso, read_json, sha256_file, write_json)
from ..data.metadata import harmonise_sites
from ..data.splits import class_presence_probability, split_groups, stratified_group_split
from .steps import audit_dir, lesion_prefix, load_all_sources

CONFIRMED = ["unreadable", "id_match", "lesion_match", "exact_pixel_match", "adjudicated_duplicate",
             "confirmed_lesion_expansion"]
PRECAUTIONARY = ["precautionary_similarity_strict", "precautionary_similarity_review", "precautionary_lesion_expansion"]
DRAFT_ARCHIVE = "cohorts_initial_draft"


def cfg_paths(config_dir: Path | None = None) -> dict:
    c = Path(config_dir or PKG_ROOT / "config")
    return {"experiment": c / "experiment.yaml", "taxonomy": c / "taxonomy.yaml", "sites": c / "site_harmonisation.yaml",
            "attribution": c / "source_attribution.yaml", "prompts": c / "prompts.yaml",
            "policy": c / "near_duplicate_policy.yaml", "approval": c / "integrity" / "cohort_approval.yaml"}


def cohort_dir(paths: dict) -> Path:
    d = Path(paths["work_dir"]) / "cohorts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _adjudication(paths: dict) -> pd.DataFrame:
    f = Path(paths["work_dir"]) / "review" / "near_duplicates_TO_ADJUDICATE.csv"
    if not f.exists():
        return pd.DataFrame(columns=["source", "image_id", "decision"])
    a = pd.read_csv(f, dtype={"decision": str}).fillna({"decision": ""})
    bad = set(a["decision"].str.strip().str.lower()) - {"", "duplicate", "not_duplicate"}
    if bad:
        raise ValueError(f"invalid adjudication decisions {bad}; use 'duplicate' or 'not_duplicate'")
    return a


def near_dup_mode(cp: dict, final: bool) -> str:
    if not final:
        return "draft"
    dec = load_yaml(cp["approval"]).get("decisions", {}) if cp["approval"].exists() else {}
    if dec.get("near_duplicate_policy") == "precautionary":
        pol = load_yaml(cp["policy"]) if cp["policy"].exists() else {}
        if pol.get("policy_id") != "precautionary" or pol.get("status") != "APPROVED" or pol.get("confirmed_duplicate") is not False:
            raise IntegrityError("near_duplicate_policy.yaml must be policy_id precautionary, status APPROVED, confirmed_duplicate false")
        return "precautionary"
    if dec.get("near_duplicate_adjudication_complete") is True:
        return "adjudication"
    raise IntegrityError("cohort approval must set decisions.near_duplicate_policy: precautionary (or legacy adjudication_complete: true)")


def _has_les(s: pd.Series) -> pd.Series:
    return s.notna() & ~s.astype(str).isin(["", "nan", "<NA>"])


def build_cohorts(paths: dict, config_dir: Path | None = None, final: bool = False) -> dict:
    cp = cfg_paths(config_dir)
    exp, tax = load_yaml(cp["experiment"]), load_yaml(cp["taxonomy"])
    sites, attr = load_yaml(cp["sites"]), load_yaml(cp["attribution"])
    if final:
        for name, c in (("taxonomy", tax), ("site table", sites), ("source attribution", attr)):
            if c.get("status") != "APPROVED":
                raise IntegrityError(f"{name} status is {c.get('status')!r}; must be APPROVED at cohort")
    mode = near_dup_mode(cp, final)
    ad = audit_dir(paths)
    srcs = load_all_sources(paths)
    inv = pd.read_csv(ad / "inventory_inventory.csv")
    unreadable = set(zip(inv.loc[~inv["readable"].astype(bool), "source"], inv.loc[~inv["readable"].astype(bool), "image_id"]))
    ov = pd.read_csv(ad / "provenance_id_lesion_overlap.csv")
    hashes = pd.read_csv(ad / "exact_overlap_hashes.csv", dtype={"phash": str, "dhash": str})
    ham_md5 = set(hashes.loc[hashes.source == "HAM", "md5_pixels"])
    cand = pd.read_csv(ad / "similarity_screen_near_duplicate_candidates.csv")
    adj = _adjudication(paths)
    dec = adj["decision"].str.strip().str.lower()
    n_blank = int((dec == "").sum())
    if mode == "precautionary" and (dec != "").any():
        raise IntegrityError(f"precautionary requires every candidate-pair decision to stay blank; {int((dec != '').sum())} are filled")
    if mode == "adjudication" and n_blank:
        raise IntegrityError(f"{n_blank} near-duplicate pairs still un-adjudicated")
    out, flows, excl_tables, claims = {}, {}, {}, {}


    ham = srcs["HAM"].copy()
    ham = ham[~ham["image_id"].map(lambda i: ("HAM", i) in unreadable)]
    bad_lab = sorted(set(ham["label_raw"]) - set(tax["ham_map"]))
    if bad_lab:
        raise ValueError(f"HAM labels not in taxonomy: {bad_lab}")
    ham["label"] = ham["label_raw"].map(tax["ham_map"])
    links_f = ad / "within_dataset_duplicates_HAM_internal_links.csv"
    links = pd.read_csv(links_f) if links_f.exists() else None
    ham["split_group"] = split_groups(ham, links)
    ham["split"], split_rep = stratified_group_split(ham, "label", "split_group", tuple(exp["split_ratios"]), exp["split_seed"])
    ham["cluster_id"] = ham["lesion_id"]
    ham["site_group"] = harmonise_sites(ham, sites)
    ham["label_idx"] = ham["label"].map({c: i for i, c in enumerate(CLASSES)})
    ham["institution"] = "HAM10000"
    out["ham"] = ham
    gsize = ham.groupby("split_group").size()
    split_rep["largest_split_group_images"] = int(gsize.max())
    split_rep["n_split_groups_merging_lesions"] = int((ham.groupby("split_group")["lesion_id"].nunique() > 1).sum())
    split_rep["FLAG_large_groups"] = bool(gsize.max() > exp["audit"].get("max_split_group_images", 50))
    claims["HAM_splits_lesion_disjoint"] = "VERIFIED_BY_CONSTRUCTION (lesion_id groups + HAM-internal duplicate links)"
    pres = class_presence_probability(ham.loc[ham.split == "train", "label"], exp["train"]["batch_size"])


    externals = [("ISIC2019_train", "e1")] + ([("ISIC2019_test", "e2")] if exp.get("include_e2") and "ISIC2019_test" in srcs else [])
    for src, tag in externals:
        df = srcs[src].copy().merge(ov[ov.source == src][["image_id", "id_match", "lesion_match"]], on="image_id", how="left")
        df = df.merge(hashes[hashes.source == src][["image_id", "md5_pixels"]], on="image_id", how="left")
        c_src = cand[cand.source == src]
        strict = set(c_src.loc[c_src.tier == "AUTO_EXCLUDE", "image_id"])
        review = set(c_src.loc[c_src.tier == "REVIEW", "image_id"])
        a_src = adj[adj.source == src]
        adjudicated_dup = set(a_src.loc[dec[a_src.index] == "duplicate", "image_id"]) if mode == "adjudication" else set()
        cleared = set(a_src.loc[dec[a_src.index] == "not_duplicate", "image_id"]) if mode == "adjudication" else set()
        reason = pd.Series(pd.NA, index=df.index, dtype=object)

        def mark(mask, name):
            reason[(reason.isna()) & mask] = name

        mark(df["image_id"].map(lambda i: (src, i) in unreadable), "unreadable")
        mark(df["id_match"].fillna(False).astype(bool), "id_match")
        mark(df["lesion_match"].fillna(False).astype(bool), "lesion_match")
        mark(df["md5_pixels"].isin(ham_md5), "exact_pixel_match")
        mark(df["image_id"].isin(adjudicated_dup), "adjudicated_duplicate")
        mark(df["image_id"].isin(strict), "precautionary_similarity_strict")
        mark(df["image_id"].isin(review - cleared), "precautionary_similarity_review")
        has_les = _has_les(df["lesion_id"])
        conf_les = set(df.loc[reason.isin([r for r in CONFIRMED if r != "unreadable"]) & has_les, "lesion_id"])
        prec_les = set(df.loc[reason.isin(PRECAUTIONARY) & has_les, "lesion_id"])
        mark(has_les & df["lesion_id"].isin(conf_les), "confirmed_lesion_expansion")
        mark(has_les & df["lesion_id"].isin(prec_les), "precautionary_lesion_expansion")
        df["exclusion_reason"] = reason
        excl_tables[tag] = _exclusion_table(df, c_src)
        kept = df[reason.isna()].copy()
        kept["institution"] = kept["lesion_id"].map(lesion_prefix).map(attr["prefix_to_institution"]).fillna("unattributed")
        kept["cluster_id"] = np.where(_has_les(kept["lesion_id"]), kept["lesion_id"].astype(str), "IMG_" + kept["image_id"])
        kept["site_group"] = harmonise_sites(kept, sites)
        known = set(tax["isic2019_map"]) | set(tax["out_of_taxonomy"])
        if set(kept["label_raw"]) - known:
            raise ValueError(f"{src}: labels not covered by taxonomy {sorted(set(kept['label_raw']) - known)}")
        closed = kept[kept["label_raw"].isin(tax["isic2019_map"])].copy()
        closed["label"] = closed["label_raw"].map(tax["isic2019_map"])
        closed["label_idx"] = closed["label"].map({c: i for i, c in enumerate(CLASSES)})
        oot = kept[kept["label_raw"].isin(tax["out_of_taxonomy"])].copy()
        oot["label"], oot["label_idx"] = oot["label_raw"], -1
        out[tag], out[f"{tag}_oot"] = closed, oot
        flows[tag] = df.assign(stage=reason.fillna("kept")).groupby(["stage", "label_raw"]).size().unstack(fill_value=0)
        claims.update(_external_claims(tag, src, df, kept, closed, exp, attr, ad, mode))
    claims["patient_disjoint"] = "NOT CLAIMABLE: neither dataset provides patient identifiers"

    cd = cohort_dir(paths)
    cols = ["image_id", "source", "institution", "lesion_id", "cluster_id", "label_raw", "label", "label_idx", "age", "sex_raw",
            "site_raw", "site_group", "path"]
    for name, d in out.items():
        d[cols + (["split", "split_group"] if name == "ham" else [])].sort_values("image_id").to_csv(cd / f"{name}.csv", index=False)
    for tag, f in flows.items():
        f.to_csv(cd / f"{tag}_flow.csv")
    for tag, t in excl_tables.items():
        t.to_csv(cd / f"{tag}_exclusions.csv", index=False)
    summary = {"built": now_iso(), "final": final, "near_duplicate_mode": mode, "claims": claims, "split_report": split_rep,
               "train_class_presence_probability_B": pres,
               "n": {k: int(len(v)) for k, v in out.items()},
               "per_class": {k: v["label"].value_counts().to_dict() for k, v in out.items()},
               "candidate_pairs_blank_decisions": n_blank,
               "config_status": {"taxonomy": tax.get("status"), "sites": sites.get("status"), "attribution": attr.get("status")}}
    write_json(cd / "claims_and_summary.json", summary)
    return summary


def _exclusion_table(df: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:

    ex = df[df["exclusion_reason"].notna()].copy()
    ev = (cand.assign(e=cand["metric"] + "=" + cand["value"].round(4).astype(str) + "->" + cand["ham_image_id"])
          .groupby("image_id")["e"].agg(";".join))
    trig = df[df["exclusion_reason"].isin(PRECAUTIONARY[:2] + [r for r in CONFIRMED if r != "unreadable"]) & _has_les(df["lesion_id"])]
    trig = trig.groupby("lesion_id")["image_id"].agg(";".join)
    ex["category"] = np.where(ex["exclusion_reason"].isin(PRECAUTIONARY), "precautionary", "confirmed")
    ex["confirmed_duplicate"] = np.where(ex["category"] == "precautionary", False,
                                         np.where(ex["exclusion_reason"] == "unreadable", pd.NA, True))
    ex["evidence"] = np.select(
        [ex["exclusion_reason"].str.startswith("precautionary_similarity"),
         ex["exclusion_reason"].str.endswith("lesion_expansion")],
        [ex["image_id"].map(ev).fillna(""), "same lesion_id as excluded image(s): " + ex["lesion_id"].map(trig).fillna("")],
        default=ex["exclusion_reason"])
    ex["lesion_prefix"] = ex["lesion_id"].map(lesion_prefix)
    return ex[["image_id", "source", "lesion_id", "lesion_prefix", "label_raw", "exclusion_reason", "category",
               "confirmed_duplicate", "evidence"]].sort_values(["category", "exclusion_reason", "image_id"])


def _external_claims(tag, src, df, kept, closed, exp, attr, ad, mode) -> dict:
    r = df["exclusion_reason"]
    eligible = ~r.isin(["unreadable", "id_match", "lesion_match", "exact_pixel_match"])
    prec = r.isin(PRECAUTIONARY)
    pre = df["lesion_id"].map(lesion_prefix)
    rate = lambda key: {k: {"eligible": int(g.sum()), "precautionary_excluded": int((prec & eligible)[g.index].sum()),
                            "rate": round(float((prec & eligible)[g.index].sum()) / max(int(g.sum()), 1), 4)}
                        for k, g in eligible.groupby(key)}
    has_les = _has_les(kept["lesion_id"])
    n_no_les = int((~has_les).sum())
    les = closed.assign(h=_has_les(closed["lesion_id"])).groupby("label")
    thr = exp["min_external_lesions_per_class"]
    les_counts = {k: int(g.loc[g["h"], "lesion_id"].nunique()) for k, g in les}
    clu_counts = {k: int(g["cluster_id"].nunique()) for k, g in les}
    label_free_screen = ad / "label_free_screen_label_free_null.json"
    calib = read_json(ad / "similarity_screen_summary.json").get(src, {}).get("calibration", {})
    counts = r.value_counts().to_dict()
    return {
        f"{tag}_confirmed_overlap_removed": {k: int(counts.get(k, 0)) for k in CONFIRMED},
        f"{tag}_precautionary_exclusions": {
            "policy": "precautionary" if mode in ("precautionary", "draft") else mode,
            "counts": {k: int(counts.get(k, 0)) for k in PRECAUTIONARY},
            "confirmed_duplicate": False,
            "statement": "Images flagged by calibrated similarity screening, and further images sharing their lesion_id, "
                         "were excluded as a precaution. They were NOT confirmed as duplicates; candidate-pair decisions "
                         "were left blank."},
        f"{tag}_image_disjoint_from_HAM": {
            "status": ("VERIFIED for confirmed-overlap criteria (image ID, lesion ID, pixel-identical content); "
                       "similarity-flagged images excluded as a precaution (not confirmed)") if mode != "draft"
            else "PROVISIONAL (cohort not frozen)",
            "near_duplicate_calibration": calib},
        f"{tag}_selection_effect": {
            "statement": "Precautionary exclusion is non-random with respect to class and source (rates below). Its effect "
                         "on external performance estimates is unknown; it is not assumed to be conservative.",
            "removal_rate_by_raw_label": rate(df["label_raw"]),
            "removal_rate_by_lesion_prefix": rate(pre)},
        f"{tag}_threshold_calibration_disclosure": {
            "statement": "Similarity thresholds were calibrated on a pooled null formed by each eligible external image's "
                         "nearest HAM neighbour of a DIFFERENT diagnosis, so external ground-truth labels informed the global "
                         "threshold values (one per metric and tier). No per-image flag decision uses that image's own label "
                         "beyond this, and no model outputs or performance results existed or were used.",
            "label_free_sensitivity": read_json(label_free_screen)["summary"] if label_free_screen.exists() else "label_free_screen not run"},
        f"{tag}_lesion_disjoint_from_HAM": {
            "status": f"VERIFIED for {int(len(kept) - n_no_les)} images with lesion_id; NOT VERIFIABLE at lesion level "
                      f"for {n_no_les} images without lesion_id (image-level checks apply)"},
        f"{tag}_min_lesion_rule": {"threshold": thr, "lesions_with_id_per_class": les_counts,
                                   "clusters_incl_singletons_per_class": clu_counts,
                                   "classes_below_threshold": sorted(k for k, v in les_counts.items() if v < thr)},
        f"{tag}_institution_disjoint_from_HAM": {
            "status": ("CONDITIONAL on approved lesion-id-prefix attribution; claimable only for attributed images"
                       if attr.get("status") == "APPROVED" else "NOT YET CLAIMABLE: source attribution not approved"),
            "institution_counts": {k: int(v) for k, v in kept["institution"].value_counts().items()},
            "contains_HAM_attributed_images": bool("HAM10000" in set(kept["institution"]))},
    }


def cohort_report(paths: dict, config_dir: Path | None = None) -> Path:

    ad = audit_dir(paths)
    build_cohorts(paths, config_dir, final=False)
    parts = ["# Check cohort data-audit report (DRAFT — for researcher review)\n", f"Generated {now_iso()}\n"]
    for f in ["inventory_summary", "provenance_summary", "exact_overlap_summary", "similarity_screen_summary", "class_counts_summary", "metadata_audit_summary", "site_mapping_summary"]:
        p = ad / f"{f}.json"
        parts.append(f"\n## {f}\n\n```json\n{p.read_text() if p.exists() else 'NOT RUN'}\n```\n")
    parts.append("\n## Draft cohorts, flow and computed claims\n\n```json\n" +
                 (cohort_dir(paths) / "claims_and_summary.json").read_text() + "\n```\n")
    parts.append("\n## Decisions required from the researcher: see docs/COHORT_FREEZE.md\n")
    out = Path(paths["work_dir"]) / "cohort_REPORT.md"
    out.write_text("".join(parts))
    return out


def _archive_draft(paths: dict) -> Path:

    work = Path(paths["work_dir"])
    arch = work / DRAFT_ARCHIVE
    hf = arch / "ARCHIVE_SHA256.json"
    if arch.exists():
        rec = read_json(hf)
        bad = [f for f, h in rec["files"].items() if not (arch / f).exists() or sha256_file(arch / f) != h]
        if bad:
            raise IntegrityError(f"initial audit draft archive modified: {bad}")
        return arch
    src = work / "cohorts"
    if not (src / "e1.csv").exists():
        raise IntegrityError("no initial audit draft cohorts to archive (run initial audit first)")
    claims = read_json(src / "claims_and_summary.json")
    if claims.get("final"):
        raise IntegrityError("cohorts/ already holds a FINAL build; refusing to archive it as the initial audit draft")
    shutil.copytree(src, arch)
    write_json(hf, {"archived": now_iso(), "from": str(src),
                    "files": {p.name: sha256_file(p) for p in sorted(arch.iterdir()) if p.name != hf.name}})
    return arch


def _compare_with_draft(paths: dict, arch: Path) -> dict:
    cd, res = cohort_dir(paths), {}
    for name in ("e1", "e1_oot", "ham"):
        a = pd.read_csv(arch / f"{name}.csv", dtype=str)
        b = pd.read_csv(cd / f"{name}.csv", dtype=str)
        ia, ib = set(a["image_id"]), set(b["image_id"])
        r = {"draft": len(ia), "final": len(ib), "only_in_draft": sorted(ia - ib)[:50], "only_in_final": sorted(ib - ia)[:50],
             "identical_image_ids": ia == ib}
        if name == "ham":
            r["identical_split_assignment"] = a.set_index("image_id")["split"].sort_index().equals(
                b.set_index("image_id")["split"].sort_index())
        res[name] = r
    return res


def freeze_cohort(paths: dict, config_dir: Path | None = None) -> dict:
    cp = cfg_paths(config_dir)
    if not cp["approval"].exists():
        raise IntegrityError(f"{cp['approval']} missing (researcher confirmation required)")
    appr = load_yaml(cp["approval"])
    if appr.get("approved") is not True or not appr.get("approver") or not appr.get("date"):
        raise IntegrityError("cohort_approval.yaml must have approved: true, approver and date")
    dec = appr.get("decisions", {})
    empty = [k for k in ("taxonomy", "site_table", "source_attribution", "prompts", "small_class_rule",
                         "label_informed_calibration") if not dec.get(k)]
    if empty:
        raise IntegrityError(f"cohort decisions not recorded: {empty}")
    if load_yaml(cp["prompts"]).get("status") != "APPROVED":
        raise IntegrityError("prompts.yaml status must be APPROVED")
    mode = near_dup_mode(cp, final=True)
    work = Path(paths["work_dir"])
    adj_f = work / "review" / "near_duplicates_TO_ADJUDICATE.csv"
    adj_hash_before = sha256_file(adj_f) if adj_f.exists() else None
    arch = _archive_draft(paths)
    summ = build_cohorts(paths, config_dir, final=True)
    cmp_ = _compare_with_draft(paths, arch)
    checks = {"identical_to_initial_draft": all(v["identical_image_ids"] for v in cmp_.values())
              and cmp_["ham"]["identical_split_assignment"],
              "candidate_decisions_unchanged": (sha256_file(adj_f) if adj_f.exists() else None) == adj_hash_before,
              "min_lesion_rule_pass": not summ["claims"]["e1_min_lesion_rule"]["classes_below_threshold"]}
    if mode == "precautionary":
        exp_counts = load_yaml(cp["policy"]).get("expected", {})
        pc = summ["claims"]["e1_precautionary_exclusions"]["counts"]
        actual = {"precautionary_similarity_strict": pc["precautionary_similarity_strict"],
                  "precautionary_similarity_review": pc["precautionary_similarity_review"],
                  "precautionary_lesion_expansion": pc["precautionary_lesion_expansion"],
                  "e1": summ["n"]["e1"], "e1_oot": summ["n"]["e1_oot"]}
        checks["expected_counts_match_policy"] = all(int(exp_counts.get(k, -1)) == v for k, v in actual.items())
        checks["expected_vs_actual"] = {k: {"expected": exp_counts.get(k), "actual": v} for k, v in actual.items()}
    write_json(work / "integrity" / "cohort_freeze_checks.json", {"time": now_iso(), "mode": mode, "checks": checks, "comparison": cmp_})
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise IntegrityError(f"cohort freeze verification failed: {failed} (see integrity/cohort_freeze_checks.json); manifest NOT written")
    cd = cohort_dir(paths)
    arts = [*sorted(cd.glob("*.csv")), cd / "claims_and_summary.json", *[p for p in cp.values() if p.exists()],
            *sorted(p for p in (work / "audit").glob("*") if p.is_file() and p.suffix in (".json", ".csv", ".npy")
                    and not p.name.startswith("log_")),
            *sorted(p for p in (work / "review").glob("*") if p.is_file()),
            *sorted(p for p in arch.glob("*") if p.is_file())]
    if (work / "cohort_REPORT.md").exists():
        arts.append(work / "cohort_REPORT.md")
    man = {"gate": "cohort", "frozen": now_iso(), "approver": appr["approver"], "approval_date": str(appr["date"]),
           "near_duplicate_mode": mode, "verification": checks,
           "artifacts": {str(Path(a).resolve()): sha256_file(a) for a in arts}, "summary": summ}
    write_json(integrity_dir(paths) / "cohort_manifest.json", man)
    _freeze_report(work, man, cmp_)
    return man


def _freeze_report(work: Path, man: dict, cmp_: dict) -> None:
    c = man["summary"]["claims"]
    lines = ["# cohort freeze verification report (for researcher confirmation before preparation)", "",
             f"Frozen {man['frozen']} · mode **{man['near_duplicate_mode']}** · approver {man['approver']} ({man['approval_date']})", "",
             "## Verification checks", *[f"- {k}: **{v}**" for k, v in man["verification"].items() if not isinstance(v, dict)],
             "", "## Image-ID identity with the initial audit draft",
             *[f"- {k}: draft {v['draft']} / final {v['final']} — identical: **{v['identical_image_ids']}**" for k, v in cmp_.items()],
             "", "## Exclusions", f"- Confirmed: {c['e1_confirmed_overlap_removed']}",
             f"- Precautionary (NOT confirmed duplicates): {c['e1_precautionary_exclusions']['counts']}",
             "", "## Minimum-lesion rule", f"```\n{c['e1_min_lesion_rule']}\n```",
             "", "## Limitations carried into the manuscript",
             f"- {c['e1_lesion_disjoint_from_HAM']['status']}", f"- Institution: {c['e1_institution_disjoint_from_HAM']['status']}",
             f"- Patient: {c['patient_disjoint']}", f"- Selection: {c['e1_selection_effect']['statement']}",
             f"- Calibration: {c['e1_threshold_calibration_disclosure']['statement']}",
             "", f"Artifacts hash-locked in cohort_manifest.json: {len(man['artifacts'])}"]
    (work / "cohort_FREEZE_REPORT.md").write_text("\n".join(lines) + "\n")
