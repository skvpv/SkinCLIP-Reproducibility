#!/usr/bin/env bash
#SBATCH --job-name=skc_53_interpret_gpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --gres=gpu:SET_GPU_TYPE:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.engine.preflight --expect-gpu --min-free-gib 8
python -m skinclip.analysis.interpret
echo "=== end $(date -Is)"
