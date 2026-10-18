#!/usr/bin/env bash
source "$(dirname "$0")/lib/submit_common.sh"
[[ -f "$WORK/cohorts/e1.csv" ]] || { echo "ERROR: initial audit draft cohorts not found in $WORK/cohorts" >&2; exit 2; }
[[ ! -f "$WORK/integrity/cohort_manifest.json" ]] || { echo "ERROR: cohort already frozen; nothing to prepare" >&2; exit 2; }
V=$(sub slurm/00_validate_package.sh)
D=$(sub --dependency=afterok:"$V" slurm/13_audit_labelfree_cpu.sh)
echo "Submitted: validate=$V labelfree_diagnostic+counts=$D"
echo "Review: $WORK/audit/label_free_screen_label_free_null.json, $WORK/review/cohort_precautionary_counts.md, VALIDATION_REPORT.md"
