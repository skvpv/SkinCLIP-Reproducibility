#!/usr/bin/env bash
#SBATCH --job-name=skc_41_campaign_cpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
case "${1:?step}" in
  matrix1) python -m skinclip.engine.campaign matrix --phase 1 ;;
  matrix2) python -m skinclip.engine.campaign matrix --phase 2 ;;
  select_lambda) python -m skinclip.engine.campaign select_lambda ;;
  audit) python -m skinclip.engine.campaign audit ;;
  freeze_checkpoint) python -m skinclip.engine.campaign freeze_checkpoint ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
echo "=== end $(date -Is)"
