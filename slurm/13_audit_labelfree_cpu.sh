#!/usr/bin/env bash
#SBATCH --job-name=skc_13_audit_labelfree_cpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.audit.run --step label_free_screen
python -I "$SKINCLIP_ROOT/scripts/cohort_counts.py"
echo "=== end $(date -Is)"
