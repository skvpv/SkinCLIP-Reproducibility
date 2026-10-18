#!/usr/bin/env bash
source "$(dirname "$0")/lib/submit_common.sh"
[[ -f "$WORK/integrity/cohort_manifest.json" ]] || { echo "ERROR: cohort not frozen (run slurm/submit_cohort_freeze.sh)" >&2; exit 2; }
grep -qE '^verified: *true' config/integrity/cohort_freeze_verified.yaml 2>/dev/null || { echo "ERROR: cohort freeze not verified by the researcher (config/integrity/cohort_freeze_verified.yaml)" >&2; exit 2; }
WANT=$(awk -F': *' '/^cohort_manifest_sha256:/{gsub(/"/,"",$2);print $2}' config/integrity/cohort_freeze_verified.yaml)
HAVE=$(sha256sum "$WORK/integrity/cohort_manifest.json" | cut -d' ' -f1)
[[ "$WANT" == "$HAVE" ]] || { echo "ERROR: cohort_manifest.json hash differs from the verified hash" >&2; exit 2; }
R=$(sub slurm/21_prep_cpu.sh)
B=$(sub --dependency=afterok:"$R" slurm/22_bc_features_gpu.sh)
S=$(sub --dependency=afterok:"$B" slurm/30_smoke_gpu.sh)
P=$(sub --dependency=afterok:"$S" slurm/31_pilot_gpu.sh)
cat <<MSG
Submitted: prep=$R bc_features=$B smoke=$S pilot=$P
STOP at Check resources when $P completes:
  1. Read $WORK/smoke/smoke_report.json (PASS required) and $WORK/pilot/pilot_report.json
  2. Review $WORK/pilot/resources.SUGGESTED.env, copy to config/integrity/resources.env, set resources_STATUS=FROZEN, approver, date
  3. bash slurm/submit_training.sh
MSG
