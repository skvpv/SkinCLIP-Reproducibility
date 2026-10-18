"""Run syntax, configuration, integrity and CPU acceptance checks."""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--static-only", action="store_true")
a = p.parse_args()
checks = []
for f in sorted(root.rglob("*.py")):
    if "conda_env" in f.parts:
        continue
    ast.parse(f.read_text(), filename=str(f))
    checks.append(str(f.relative_to(root)))
for f in sorted((root / "slurm").rglob("*.sh")) + sorted((root / "env").glob("*.sh")):
    subprocess.run(["bash", "-n", str(f)], check=True)
import yaml
for f in sorted((root / "config").rglob("*.yaml")):
    yaml.safe_load(f.read_text())
from skinclip.validate import static_checks
static = static_checks()
if not static["pass"]:
    raise RuntimeError(static)
code = None
if not a.static_only:
    code = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=root).returncode
report = {"static_pass": True, "python_files": len(checks), "design_checks": static,
          "tests_executed": not a.static_only, "test_returncode": code,
          "acceptance_pass": code == 0 if code is not None else False}
(root / "RELEASE_CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(code or 0)
