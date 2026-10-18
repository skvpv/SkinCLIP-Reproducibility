# Installation and execution

## Environment

The recorded experiment used Python 3.13.16, PyTorch 2.14.1+cu130 and the dependency
versions in `env/requirements.txt`. A compatible NVIDIA driver is required for the
recorded CUDA build. These versions describe the recorded environment; this release
has not independently tested installation from a fresh package index.

Within an allocated job, run `bash env/create_env.sh` with conda available, or use
an existing environment matching the requirements. The script creates
`env/activate_env.sh`. Save the installed-package list and preflight output. All
scientific commands below run through Slurm; submission wrappers only submit jobs.

## CPU acceptance

Set `SKINCLIP_PYTHON` to the absolute Python executable of the chosen environment.
Submit the acceptance script with site-specific resources, for example:

```bash
export SKINCLIP_PYTHON=/absolute/path/to/environment/bin/python
sbatch --account=ACCOUNT --partition=CPU_PARTITION --cpus-per-task=4 \
  --mem=16G --time=00:30:00 --output=acceptance_%j.log \
  slurm/check_release.sh
```

The suite uses small synthetic fixtures and does not download images or weights.
Its result is written to `RELEASE_CHECK.json`. It does not reproduce scientific
training results. Run on CPU without a GPU allocation.

## Configure a new experiment

Copy `config/paths.yaml.template` to `config/paths.yaml`. Replace each `<SET_ME>`
with the intended absolute path. Set `work_dir` to a NEW empty directory with
sufficient storage for caches, checkpoints and predictions. Never reuse a work
directory produced by a differently packaged implementation.

In an allocated setup job, configure scheduler names:

```bash
python scripts/configure_cluster.py --account ACCOUNT \
  --cpu-partition CPU_PARTITION --gpu-partition GPU_PARTITION --gpu-type GPU_TYPE
```

Review the generated scheduler scripts and memory/time limits. GPU_TYPE must be a
verified typed resource on the local cluster. The setup does not infer GPU access.
Copy/create `env/activate_env.sh` to activate the intended environment. Staging
weights and datasets is described in DATA_AND_WEIGHTS.md.

## Full reproduction order

1. Submit `bash slurm/submit_audit.sh` and inspect successful completion.
2. Follow COHORT_FREEZE.md, including `submit_cohort_review.sh` and `submit_cohort_freeze.sh`.
3. Submit `bash slurm/submit_prepare.sh`; inspect smoke and resource-pilot results.
4. Confirm `config/integrity/resources.env` as described in COHORT_FREEZE.md.
5. Submit `bash slurm/submit_training.sh`. Selection, remaining runs, audit and checkpoint freeze are dependency-chained.
6. After a successful training audit and checkpoint freeze, submit `bash slurm/submit_evaluation.sh`.
7. Inspect final validation and compare result tables with `results/` in this repository.

Frozen evaluation results are not overwritten without an explicit reason. Do not
choose models or alter the protocol based on the reproduced test results.
GPU architecture and numerical libraries can affect deterministic repeatability;
record the environment and investigate discrepancies rather than assuming bitwise
agreement across platforms.

## Independent numerical verification

After evaluation, run the following inside a CPU allocation:

```bash
python scripts/independent_bootstrap.py \
  --eval-root /absolute/path/to/new_work/eval \
  --output-dir /absolute/path/to/new_work/independent_check
```

This implementation does not import the model analysis module. It reproduces the
bootstrap calculation from the original per-run predictions. It is a numerical
cross-check, not an independent reimplementation of model training. Preserve
input hashes and compare all three contrasts with `results/independent_bootstrap.json`.

## Publication acceptance

The source, synthetic tests and static checks are separate from the historical
experiment. Run this release's acceptance suite, retain its log, and freeze the
Git commit only after it passes. Record that commit with the archived version.
