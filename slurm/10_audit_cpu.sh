#!/usr/bin/env bash
#SBATCH --job-name=skc_10_audit_cpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.audit.run --step all_cpu --workers "${SLURM_CPUS_PER_TASK}"
echo "=== end $(date -Is)"
