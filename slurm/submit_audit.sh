#!/usr/bin/env bash
source "$(dirname "$0")/lib/submit_common.sh"
V=$(sub slurm/00_validate_package.sh)
P=$(sub --dependency=afterok:"$V" slurm/01_preflight_gpu.sh)
A=$(sub --dependency=afterok:"$P" slurm/10_audit_cpu.sh)
E=$(sub --dependency=afterok:"$A" slurm/11_audit_embed_gpu.sh)
N=$(sub --dependency=afterok:"$E" slurm/12_audit_neardup_cpu.sh)
cat <<MSG
Submitted: validate=$V preflight=$P audit=$A embed=$E neardup+report=$N
After these jobs succeed, follow docs/COHORT_FREEZE.md.
The published precautionary policy leaves candidate-pair decisions blank.
MSG
