#!/usr/bin/env bash
#SBATCH --job-name=skc_21_prep_cpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.engine.prep --task text_bank
python -m skinclip.engine.prep --task cache_ham --workers "${SLURM_CPUS_PER_TASK}"
echo "=== end $(date -Is)"
