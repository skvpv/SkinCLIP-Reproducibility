#!/usr/bin/env bash
#SBATCH --job-name=skc_11_audit_embed_gpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --gres=gpu:SET_GPU_TYPE:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
python -m skinclip.engine.preflight --expect-gpu --min-free-gib 4
python -m skinclip.audit.run --step image_embeddings --device cuda
echo "=== end $(date -Is)"
