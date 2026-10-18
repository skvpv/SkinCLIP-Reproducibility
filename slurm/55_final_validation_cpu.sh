#!/usr/bin/env bash
#SBATCH --job-name=skc_55_final_validation_cpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.validate --final --out "$ROOT/VALIDATION_REPORT_FINAL"
echo "=== end $(date -Is)"
