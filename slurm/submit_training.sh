#!/usr/bin/env bash
source "$(dirname "$0")/lib/submit_common.sh"
[[ -f config/integrity/resources.env ]] || { echo "ERROR: config/integrity/resources.env missing" >&2; exit 2; }
source config/integrity/resources.env
[[ "${resources_STATUS}" == "FROZEN" ]] || { echo "ERROR: resources_STATUS must be FROZEN (is ${resources_STATUS})" >&2; exit 2; }
[[ -f "$WORK/integrity/cohort_manifest.json" ]] || { echo "ERROR: cohort manifest missing" >&2; exit 2; }
GPU=(--partition="$GPU_PARTITION" --gres="$GPU_GRES" --cpus-per-task="$GPU_CPUS" --mem="$GPU_MEM" --time="$GPU_TIME")
LIGHT=(--partition="$LIGHT_PARTITION" --gres="$LIGHT_GRES" --cpus-per-task="$LIGHT_CPUS" --mem="$LIGHT_MEM" --time="$LIGHT_TIME")
CPUR=(--partition="$CPU_PARTITION" --cpus-per-task="$CPU_CPUS" --mem="$CPU_MEM" --time="$CPU_TIME")
M1=$(sub slurm/41_campaign_cpu.sh matrix1)
T1=$(sub --dependency=afterok:"$M1" --array=0-8%1 "${GPU[@]}" slurm/40_train_array.sh "$WORK/campaign/phase1_gpu.tsv" 9)
L1=$(sub --dependency=afterok:"$M1" --array=0-2%1 "${LIGHT[@]}" slurm/40_train_array.sh "$WORK/campaign/phase1_gpu_light.tsv" 3)
C1=$(sub --dependency=afterok:"$M1" --array=0-2%3 "${CPUR[@]}" slurm/42_train_cpu_array.sh "$WORK/campaign/phase1_cpu.tsv" 3)
SL=$(sub --dependency=afterok:"$T1" slurm/41_campaign_cpu.sh select_lambda)
M2=$(sub --dependency=afterok:"$SL" slurm/41_campaign_cpu.sh matrix2)
T2=$(sub --dependency=afterok:"$M2" --array=0-9%1 "${GPU[@]}" slurm/40_train_array.sh "$WORK/campaign/phase2_gpu.tsv" 10)
AU=$(sub --dependency=afterok:"$T2":"$L1":"$C1" slurm/41_campaign_cpu.sh audit)
FZ=$(sub --dependency=afterok:"$AU" slurm/41_campaign_cpu.sh freeze_checkpoint)
cat <<MSG
Submitted: matrix1=$M1 phase1_gpu=$T1 probes=$L1 meta_only=$C1 select_lambda=$SL matrix2=$M2 phase2_gpu=$T2 audit=$AU freeze_checkpoint=$FZ
Monitor: squeue -u \$USER ; tail logs/skc_train_*.out ; $WORK/campaign/run_audit.json
After freeze_checkpoint succeeds, evaluation is a separate researcher decision: bash slurm/submit_evaluation.sh
MSG
