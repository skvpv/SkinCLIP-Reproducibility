set -euo pipefail
: "${SKINCLIP_ROOT:?SKINCLIP_ROOT must be exported by the submit wrapper (absolute project root)}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this script must be submitted with sbatch, not executed on the login shell" >&2
  exit 2
fi
ROOT="$SKINCLIP_ROOT"
cd "$ROOT"
export NVCC_PREPEND_FLAGS="" NVCC_APPEND_FLAGS=""
set +u
source "$ROOT/env/activate_env.sh"
set -u
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SKINCLIP_PATHS="$ROOT/config/paths.yaml"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0
HF_HOME="$(python -c 'import yaml,os;print(yaml.safe_load(open(os.environ["SKINCLIP_PATHS"]))["weights"]["hf_home"])')"
TORCH_HOME="$(python -c 'import yaml,os;print(yaml.safe_load(open(os.environ["SKINCLIP_PATHS"]))["weights"]["torch_home"])')"
export HF_HOME TORCH_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== job ${SLURM_JOB_ID} (${SLURM_JOB_NAME:-}) array=${SLURM_ARRAY_TASK_ID:-none} partition=${SLURM_JOB_PARTITION:-} host=$(hostname) start=$(date -Is)"
echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (set by Slurm; never overridden)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
python - <<'PY'
import torch, sys
print("=== python", sys.version.split()[0], "torch", torch.__version__, "cuda build", torch.version.cuda,
      "cuda available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
PY
