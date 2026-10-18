#!/usr/bin/env bash
#SBATCH --job-name=skc_train_cpu
#SBATCH --account=SET_ACCOUNT
#SBATCH --partition=SET_PARTITION
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
source "$SKINCLIP_ROOT/slurm/lib/job_env.sh"
TSV="${1:?campaign TSV}"; EXPECTED="${2:?expected number of runs}"
N=$(( $(wc -l < "$TSV") - 1 ))
if [[ "$N" -ne "$EXPECTED" ]]; then echo "ERROR: $TSV has $N runs, expected $EXPECTED" >&2; exit 3; fi
LINE=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 2 ))p" "$TSV")
ARM=$(cut -f1 <<<"$LINE"); SEED=$(cut -f2 <<<"$LINE"); LAM=$(cut -f3 <<<"$LINE")
echo "=== run arm=$ARM seed=$SEED lam=$LAM"
if [[ "$ARM" != "M0" ]]; then python -m skinclip.engine.preflight --expect-gpu --min-free-gib 8; fi
python -m skinclip.engine.train --arm "$ARM" --seed "$SEED" --lam "$LAM"
echo "=== end $(date -Is)"
