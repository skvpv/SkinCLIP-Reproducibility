#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SKINCLIP_PYTHON:?Set SKINCLIP_PYTHON to the environment Python executable}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$SKINCLIP_PYTHON" scripts/check_release.py
