#!/usr/bin/env bash
source "$(dirname "$0")/lib/submit_common.sh"
[[ -f "$WORK/integrity/checkpoint_manifest.json" ]] || { echo "ERROR: checkpoint manifest missing; evaluation sets are locked" >&2; exit 2; }
source config/integrity/resources.env
GPU=(--partition="$GPU_PARTITION" --gres="$GPU_GRES" --cpus-per-task="$GPU_CPUS" --mem="$GPU_MEM")
EV=$(sub slurm/50_evaluate_gpu.sh)
ST=$(sub --dependency=afterok:"$EV" slurm/51_stats_cpu.sh)
CO=$(sub --dependency=afterok:"$EV" "${GPU[@]}" slurm/52_cost_gpu.sh)
IN=$(sub --dependency=afterok:"$EV" slurm/53_interpret_gpu.sh)
FG=$(sub --dependency=afterok:"$ST":"$CO":"$IN" slurm/54_figures_cpu.sh)
FV=$(sub --dependency=afterok:"$FG" slurm/55_final_validation_cpu.sh)
echo "Submitted: evaluate=$EV stats=$ST cost=$CO interpret=$IN figures=$FG final_validation=$FV"
