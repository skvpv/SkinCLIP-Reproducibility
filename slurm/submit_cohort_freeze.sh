#!/usr/bin/env bash
source "$(dirname "$0")/lib/submit_common.sh"
grep -qE '^approved: *true' config/integrity/cohort_approval.yaml 2>/dev/null || { echo "ERROR: config/integrity/cohort_approval.yaml not signed" >&2; exit 2; }
grep -qE '^status: *APPROVED' config/near_duplicate_policy.yaml || { echo "ERROR: near_duplicate_policy.yaml not APPROVED" >&2; exit 2; }
[[ ! -f "$WORK/integrity/cohort_manifest.json" ]] || { echo "ERROR: cohort manifest already exists" >&2; exit 2; }
F=$(sub slurm/20_freeze_cohort_cpu.sh)
echo "Submitted: freeze_cohort=$F"
echo "Then review $WORK/cohort_FREEZE_REPORT.md and $WORK/integrity/cohort_freeze_checks.json, sign config/integrity/cohort_freeze_verified.yaml,"
echo "and only then run: bash slurm/submit_prepare.sh"
