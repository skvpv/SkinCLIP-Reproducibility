#!/usr/bin/env bash
#SBATCH --job-name=skc_51_stats_cpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.analysis.stats
echo "=== end $(date -Is)"
