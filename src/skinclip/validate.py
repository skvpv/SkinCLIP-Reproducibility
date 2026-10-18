from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from . import CLASSES
from .common import PKG_ROOT, load_yaml, now_iso, read_json, software_versions, write_json


def static_checks() -> dict:
    c = PKG_ROOT / "config"
    res = {}
    exp = load_yaml(c / "experiment.yaml")
    prompts = load_yaml(c / "prompts.yaml")
    tax = load_yaml(c / "taxonomy.yaml")
    res["experiment_not_synthetic"] = exp.get("synthetic") is False
    res["prompt_order_matches_classes"] = prompts["order"] == CLASSES and sorted(prompts["prompts"]) == sorted(CLASSES)
    res["taxonomy_classes_match"] = tax["classes"] == CLASSES and sorted(set(tax["isic2019_map"].values())) == sorted(CLASSES)
    res["scc_not_auto_mapped"] = "SCC" not in tax["isic2019_map"] and "SCC" in tax["out_of_taxonomy"] and not tax["sensitivity_maps"]
    from .engine.campaign import count_summary

    cnt = count_summary(exp, {"F11": 0.5, "F01": 0.5})
    res["run_counts"] = cnt
    res["run_counts_match_approval"] = (cnt["full_finetune_runs"] == 19 and cnt["light_probe_runs"] == 3
                                        and cnt["cpu_meta_only_runs"] == 3)
    res["lambda_selection_seed0_validation_only"] = exp["lambda"]["select_seed"] == 0
    pol = load_yaml(c / "near_duplicate_policy.yaml")
    res["precautionary_policy_not_confirmed_duplicates"] = (pol.get("policy_id") == "precautionary" and pol.get("confirmed_duplicate") is False
                                                  and set(pol.get("expected", {})) == {
                                                      "precautionary_similarity_strict", "precautionary_similarity_review",
                                                      "precautionary_lesion_expansion", "e1", "e1_oot"})
    res["precautionary_policy_status"] = pol.get("status")
    res["approval_status"] = {f: load_yaml(c / f).get("status") for f in
                              ("taxonomy.yaml", "site_harmonisation.yaml", "source_attribution.yaml", "prompts.yaml")}
    res["pass"] = all(v for k, v in res.items() if isinstance(v, bool))
    return res


REQUIRED = ["--account", "--partition", "--cpus-per-task", "--mem", "--time", "--output", "--error"]


def lint_slurm() -> dict:
    out, ok = {}, True
    for f in sorted((PKG_ROOT / "slurm").glob("[0-9]*.sh")):
        txt = f.read_text()
        sb = re.findall(r"^#SBATCH\s+(\S+)", txt, re.M)
        keys = {s.split("=")[0] for s in sb}
        issues = [f"missing {k}" for k in REQUIRED if k not in keys]
        if re.search(r"--gres=gpu:\d", txt):
            issues.append("untyped GRES")
        if re.search(r"^\s*(export\s+)?CUDA_VISIBLE_DEVICES=", txt, re.M):
            issues.append("sets CUDA_VISIBLE_DEVICES")
        if "job_env.sh" not in txt and "create_env.sh" not in txt:
            issues.append("does not source slurm/lib/job_env.sh")
        syn = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        if syn.returncode:
            issues.append("bash syntax: " + syn.stderr.strip())
        if "REPLACE_CPU_PARTITION" in txt:
            issues.append("NOTE: CPU partition placeholder (run scripts/configure_cluster.py)")
        out[f.name] = issues or ["ok"]
        ok &= not [i for i in issues if not i.startswith("NOTE")]
    for f in sorted((PKG_ROOT / "slurm").glob("submit_*.sh")) + sorted((PKG_ROOT / "slurm" / "lib").glob("*.sh")):
        syn = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        out[f.name] = ["ok"] if syn.returncode == 0 else ["bash syntax: " + syn.stderr.strip()]
        ok &= syn.returncode == 0
    out["pass"] = bool(ok)
    return out


def run_tests() -> dict:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(PKG_ROOT / "tests")],
                       capture_output=True, text=True, cwd=PKG_ROOT)
    tail = p.stdout.strip().splitlines()[-15:]
    return {"returncode": p.returncode, "summary": tail, "pass": p.returncode == 0}


CHECKS = {
    "discovery Discovery": "preflight/*.json",
    "allocation Allocation": "preflight/*.json",
    "environment Environment": "preflight/*.json",
    "cohort Smoke (+ data gate)": "smoke/smoke_report.json",
    "resources Pilot": "pilot/pilot_report.json",
    "checkpoint Freeze": "integrity/checkpoint_manifest.json",
    "run_audit Audit": "campaign/run_audit.json",
    "result_evidence Evidence": "results/statistics.json",
}


def evidence(final: bool) -> dict:
    try:
        from .common import load_paths

        work = Path(load_paths()["work_dir"])
    except Exception:
        return {g: "NOT RUN (no config/paths.yaml in this environment)" for g in CHECKS}
    out = {}
    for g, pat in CHECKS.items():
        hits = sorted(work.glob(pat))
        if not hits:
            out[g] = "NOT RUN"
            continue
        if g.startswith(("discovery", "allocation", "environment")):
            recs = [read_json(h) for h in hits]
            out[g] = "PASS (evidence on Slurm)" if any(r.get("PASS") and r["slurm"]["inside_slurm"] for r in recs) else "FAIL/INCOMPLETE"
        elif g.startswith("cohort"):
            out[g] = "PASS" if read_json(hits[0]).get("PASS") else "FAIL"
        else:
            out[g] = f"EVIDENCE PRESENT: {hits[0].name}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PKG_ROOT / "VALIDATION_REPORT"))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    rep = {"time": now_iso(), "versions": software_versions(), "static": static_checks(), "slurm_lint": lint_slurm(),
           "tests": {"pass": True, "summary": ["skipped"]} if a.skip_tests else run_tests(), "cluster_integrity": evidence(a.final)}
    rep["PASS"] = rep["static"]["pass"] and rep["slurm_lint"]["pass"] and rep["tests"]["pass"]
    write_json(Path(a.out + ".json"), rep)
    md = [f"# Validation report ({rep['time']})", "", f"**Overall (package checks): {'PASS' if rep['PASS'] else 'FAIL'}**", "",
          "## Static design checks", *[f"- {k}: {v}" for k, v in rep["static"].items()], "",
          "## Slurm lint", *[f"- {k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in rep["slurm_lint"].items()], "",
          "## Test suite", "```", *rep["tests"]["summary"], "```", "",
          "## Slurm integrity (evidence-based; never inferred)", *[f"- {k}: **{v}**" for k, v in rep["cluster_integrity"].items()]]
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
