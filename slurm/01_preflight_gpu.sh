#!/usr/bin/env bash
#SBATCH --job-name=skc_01_preflight_gpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --gres=gpu:SET_GPU_TYPE:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.engine.preflight --expect-gpu --check-weights --min-free-gib 8
echo "=== end $(date -Is)"
