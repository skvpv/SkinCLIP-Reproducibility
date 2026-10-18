set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SKINCLIP_ROOT="$ROOT"
cd "$ROOT"
mkdir -p logs
source config/cluster.env
if grep -qs REPLACE_CPU_PARTITION config/cluster.env slurm/*.sh; then
  echo "ERROR: CPU partition not configured. Run: python scripts/configure_cluster.py --help" >&2; exit 2
fi
[[ -f config/paths.yaml ]] || { echo "ERROR: config/paths.yaml missing (copy config/paths.yaml.template)" >&2; exit 2; }
[[ -f env/activate_env.sh ]] || { echo "ERROR: env/activate_env.sh missing (run env/create_env.sh)" >&2; exit 2; }
WORK="$(awk -F': *' '/^work_dir:/{print $2}' config/paths.yaml)"
[[ "$WORK" == /* ]] || { echo "ERROR: work_dir in config/paths.yaml must be absolute" >&2; exit 2; }
sub() {  # sbatch --parsable with a submission log
  local id; id=$(sbatch --parsable "$@"); echo "$(date -Is) $id sbatch $*" >> logs/submissions.log; echo "$id"
}
