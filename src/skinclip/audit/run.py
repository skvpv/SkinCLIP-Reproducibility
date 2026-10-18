from __future__ import annotations

import argparse
import json
import os
import sys

from ..common import PKG_ROOT, load_paths, load_yaml, slurm_context, software_versions, write_json
from . import cohorts, neardup, steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    paths = load_paths(a.paths)
    exp = load_yaml(f"{a.config_dir}/experiment.yaml")
    au = exp["audit"]
    site_table = f"{a.config_dir}/site_harmonisation.yaml"
    run = {
        "inventory": lambda: steps.inventory_inventory(paths, a.workers),
        "provenance": lambda: steps.provenance_id_lesion_overlap(paths),
        "exact_overlap": lambda: steps.exact_overlap_hashes(paths, a.workers),
        "image_embeddings": lambda: neardup.compute_embeddings(paths, au["embedding_model"], a.device,
                                                  pretrained=au.get("embedding_pretrained", True),
                                                  img_size=au.get("embedding_img_size", 224)),
        "similarity_screen": lambda: neardup.similarity_screen_near_duplicates(paths, au["fpr_strict"], au["fpr_review"], au["n_negative_pairs"],
                                                   au["seed"], use_embeddings=au.get("use_embeddings", True)),
        "within_dataset_duplicates": lambda: {"n_links": int(len(neardup.ham_internal_links(paths)))},
        "label_free_screen": lambda: neardup.label_free_screen_label_free_null(paths, au["fpr_strict"], au["fpr_review"],
                                                   use_embeddings=au.get("use_embeddings", True))["summary"],
        "class_counts": lambda: steps.class_counts_label_crosstab(paths),
        "metadata_audit": lambda: steps.metadata_audit_lesion_structure(paths),
        "site_mapping": lambda: steps.site_mapping_metadata(paths, site_table),
        "report": lambda: {"report": str(cohorts.cohort_report(paths, a.config_dir))},
        "freeze_cohort": lambda: {"manifest_artifacts": len(cohorts.freeze_cohort(paths, a.config_dir)["artifacts"])},
    }
    order = ["inventory", "provenance", "exact_overlap", "class_counts", "metadata_audit", "site_mapping"] if a.step == "all_cpu" else [a.step]
    results = {}
    for s in order:
        if s not in run:
            print(f"unknown step {s}", file=sys.stderr)
            return 2
        results[s] = run[s]()
        print(json.dumps({s: results[s]}, default=str)[:4000])
    write_json(steps.audit_dir(paths) / f"log_{a.step}.json",
               {"step": a.step, "slurm": slurm_context(), "versions": software_versions(), "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
