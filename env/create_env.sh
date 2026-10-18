#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${SKINCLIP_ENV_PREFIX:-$ROOT/env/conda_env}"
conda create -y -p "$ENV_PREFIX" python=3.13
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
set -u
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/env/requirements.txt"
python -m pip freeze > "$ROOT/env/requirements.lock.$(hostname -s).txt"
sed "s#__ENV_PREFIX__#$ENV_PREFIX#" "$ROOT/env/activate_env.sh.template" > "$ROOT/env/activate_env.sh"
echo "environment created at $ENV_PREFIX; lock file written; activate via env/activate_env.sh"
