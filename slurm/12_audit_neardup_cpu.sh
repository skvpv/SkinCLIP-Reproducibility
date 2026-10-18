#!/usr/bin/env bash
#SBATCH --job-name=skc_12_audit_neardup_cpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.audit.run --step similarity_screen
python -m skinclip.audit.run --step within_dataset_duplicates
python -m skinclip.audit.run --step report
echo "=== end $(date -Is)"
