"""Configure scheduler resources for a local installation."""
import argparse
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument("--account", required=True)
p.add_argument("--cpu-partition", required=True)
p.add_argument("--gpu-partition", required=True)
p.add_argument("--gpu-type", required=True)
a = p.parse_args()
for value in vars(a).values():
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        p.error("scheduler names may contain letters, digits, underscore, period and hyphen")
root = Path(__file__).resolve().parents[1]
for f in (root / "slurm").glob("[0-9]*.sh"):
    s = f.read_text()
    gpu = "#SBATCH --gres=" in s
    s = re.sub(r"(?m)^#SBATCH --account=.*", "#SBATCH --account=" + a.account, s)
    s = re.sub(r"(?m)^#SBATCH --partition=.*", "#SBATCH --partition=" + (a.gpu_partition if gpu else a.cpu_partition), s)
    s = re.sub(r"(?m)^#SBATCH --gres=.*", "#SBATCH --gres=gpu:" + a.gpu_type + ":1", s)
    f.write_text(s)
s = "ACCOUNT=" + a.account + "\nCPU_PARTITION=" + a.cpu_partition + "\n"
for prefix in ("DEV", "VAL", "PROD"):
    s += prefix + "_PARTITION=" + a.gpu_partition + "\n" + prefix + "_GRES=gpu:" + a.gpu_type + ":1\n"
(root / "config/cluster.env").write_text(s)
f = root / "config/integrity/resources.env.template"
s = f.read_text().replace("SET_PARTITION", a.gpu_partition).replace("SET_GPU_TYPE", a.gpu_type)
s += "\nCPU_PARTITION=" + a.cpu_partition + "\n"
f.write_text(s)
print("Configured scheduler scripts; review resource limits before submission.")
