import json
import os
from pathlib import Path

import pandas as pd
import yaml

paths = yaml.safe_load(open(os.environ["SKINCLIP_PATHS"]))
W = Path(paths["work_dir"])
cd, ad, rd = W / "cohorts", W / "audit", W / "review"

e1 = pd.read_csv(cd / "e1.csv", dtype={"lesion_id": str, "cluster_id": str})
oot = pd.read_csv(cd / "e1_oot.csv", dtype={"lesion_id": str, "cluster_id": str})
flow = pd.read_csv(cd / "e1_flow.csv", index_col=0)
cand = pd.read_csv(ad / "similarity_screen_near_duplicate_candidates.csv")
cand = cand[cand["source"] == "ISIC2019_train"]

has_les = lambda d: d["lesion_id"].notna() & ~d["lesion_id"].astype(str).isin(["", "nan", "<NA>"])

def per_class(d, label_col="label"):
    g = d.assign(has=has_les(d)).groupby(label_col)
    return pd.DataFrame({
        "images": g.size(),
        "images_with_lesion_id": g["has"].sum().astype(int),
        "lesions_with_id": g.apply(lambda x: x.loc[has_les(x), "lesion_id"].nunique(), include_groups=False),
        "clusters_incl_singletons": g["cluster_id"].nunique(),
    })

flagged_strict = set(cand.loc[cand.tier == "AUTO_EXCLUDE", "image_id"])
flagged_review = set(cand.loc[cand.tier == "REVIEW", "image_id"])
out = {
    "flagged_images": {"strict_tier": len(flagged_strict), "review_tier": len(flagged_review),
                       "overlap_between_tiers": len(flagged_strict & flagged_review),
                       "union": len(flagged_strict | flagged_review)},
    "flow_stage_by_raw_label": flow.astype(int).to_dict(orient="index"),
    "flow_stage_totals": flow.sum(axis=1).astype(int).to_dict(),
    "e1_closed_set": per_class(e1).astype(int).to_dict(orient="index"),
    "e1_totals": {"images": int(len(e1)), "lesions_with_id": int(e1.loc[has_les(e1), "lesion_id"].nunique()),
                  "clusters_incl_singletons": int(e1["cluster_id"].nunique()),
                  "images_without_lesion_id": int((~has_les(e1)).sum())},
    "e1_oot": per_class(oot, "label").astype(int).to_dict(orient="index"),
    "e1_institution_counts": e1["institution"].value_counts().to_dict(),
}

provenance = pd.read_csv(ad / "provenance_id_lesion_overlap.csv")
provenance = provenance[provenance["source"] == "ISIC2019_train"].set_index("image_id")
for tier, ids in (("strict", flagged_strict), ("review", flagged_review)):
    out[f"flagged_{tier}_by_lesion_prefix"] = provenance.loc[sorted(ids & set(provenance.index)), "lesion_prefix"].value_counts().to_dict()
    out[f"flagged_{tier}_by_raw_label"] = provenance.loc[sorted(ids & set(provenance.index)), "label_raw"].value_counts().to_dict()
min_les = yaml.safe_load(open(Path(os.environ["SKINCLIP_ROOT"]) / "config" / "experiment.yaml"))["min_external_lesions_per_class"]
pc = per_class(e1)
out["min_lesion_rule"] = {"threshold": min_les,
                          "lesions_with_id_per_class": pc["lesions_with_id"].astype(int).to_dict(),
                          "clusters_incl_singletons_per_class": pc["clusters_incl_singletons"].astype(int).to_dict(),
                          "classes_below_threshold_strict_count": sorted(pc.index[pc["lesions_with_id"] < min_les]),
                          "classes_below_threshold_cluster_count": sorted(pc.index[pc["clusters_incl_singletons"] < min_les])}
rd.mkdir(parents=True, exist_ok=True)
(rd / "cohort_precautionary_counts.json").write_text(json.dumps(out, indent=2, default=int))
md = ["# cohort precautionary-exclusion counts (read-only query)", "", "## E1 closed set per class", "```\n" + pc.to_string() + "\n```", "",
      "## Exclusion flow (stage x raw ISIC label)", "```\n" + flow.to_string() + "\n```", "", "## Minimum-lesion rule",
      json.dumps(out["min_lesion_rule"], indent=1)]
(rd / "cohort_precautionary_counts.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
