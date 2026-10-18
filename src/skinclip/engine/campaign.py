from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..common import (PKG_ROOT, RunRecord, integrity_dir, load_paths, load_yaml, now_iso, read_json, sha256_file, write_json)


def _cdir(paths) -> Path:
    d = Path(paths["work_dir"]) / "campaign"
    d.mkdir(parents=True, exist_ok=True)
    return d


def phase1(exp: dict) -> list[tuple[str, int, float]]:
    s0 = exp["lambda"]["select_seed"]
    rows = [(arm, s0, float(l)) for arm in ("F11", "F01") for l in exp["lambda"]["grid"]]
    rows += [(arm, s, 0.0) for arm in ("MB", "BCLP", "M0") for s in exp["arms"][arm]["seeds"]]
    return rows


def phase2(exp: dict, sel: dict) -> list[tuple[str, int, float]]:
    s0 = exp["lambda"]["select_seed"]
    rows = [(arm, s, float(sel[arm])) for arm in ("F11", "F01") for s in exp["arms"][arm]["seeds"] if s != s0]
    rows += [("F10", s, float(sel["F11"])) for s in exp["arms"]["F10"]["seeds"]]
    rows += [("F00", s, float(sel["F01"])) for s in exp["arms"]["F00"]["seeds"]]
    return rows


def reported_runs(exp: dict, sel: dict) -> list[tuple[str, int, float]]:

    rows = [(arm, s, float(sel[arm])) for arm in ("F11", "F01") for s in exp["arms"][arm]["seeds"]]
    rows += [("F10", s, float(sel["F11"])) for s in exp["arms"]["F10"]["seeds"]]
    rows += [("F00", s, float(sel["F01"])) for s in exp["arms"]["F00"]["seeds"]]
    rows += [(arm, s, 0.0) for arm in ("MB", "BCLP", "M0") for s in exp["arms"][arm]["seeds"]]
    return rows


def expected_runs(exp: dict, sel: dict | None) -> list[tuple[str, int, float]]:
    return phase1(exp) + (phase2(exp, sel) if sel else [])


def count_summary(exp: dict, sel: dict) -> dict:
    allr = expected_runs(exp, sel)
    full = [r for r in allr if exp["arms"][r[0]]["kind"] in ("prototype", "metablock")]
    return {"full_finetune_runs": len(full), "light_probe_runs": sum(r[0] == "BCLP" for r in allr),
            "cpu_meta_only_runs": sum(r[0] == "M0" for r in allr), "total_listed": len(allr)}


def write_matrix(paths, exp, phase: int) -> Path:
    if phase == 1:
        rows = phase1(exp)
    else:
        rows = phase2(exp, read_json(_cdir(paths) / "lambda_selection.json")["selected"])
    df = pd.DataFrame(rows, columns=["arm", "seed", "lam"])
    df["lane"] = df["arm"].map(lambda a: exp["arms"][a]["lane"])
    out = _cdir(paths) / f"phase{phase}.tsv"
    df.to_csv(out, sep="\t", index=False)
    for lane, g in df.groupby("lane"):
        g.to_csv(_cdir(paths) / f"phase{phase}_{lane}.tsv", sep="\t", index=False)
    return out


def _rec(paths, arm, seed, lam) -> dict | None:
    p = Path(paths["work_dir"]) / "runs" / arm / f"lam{lam:g}" / f"seed{seed}" / "run_record.json"
    return read_json(p) if p.exists() else None


def select_lambda(paths, exp) -> dict:

    s0, out = exp["lambda"]["select_seed"], {"selected": {}, "table": {}, "rule": "max val BMA on seed 0; tie -> smaller",
                                             "limitation": "seed 0 is used both for selection and in the reported mean",
                                             "time": now_iso()}
    for arm in ("F11", "F01"):
        vals = {}
        for l in exp["lambda"]["grid"]:
            r = _rec(paths, arm, s0, float(l))
            if not r or r.get("state") not in ("TRAINING_COMPLETED", "COMPLETED"):
                raise RuntimeError(f"{arm} lam={l} seed={s0} not completed; cannot select")
            vals[float(l)] = next(h["best_val_bma"] for h in reversed(r["history"]) if "best_val_bma" in h)
        best = max(vals.values())
        out["selected"][arm] = min(l for l, v in vals.items() if v >= best - 1e-12)
        out["table"][arm] = vals
    write_json(_cdir(paths) / "lambda_selection.json", out)
    return out


def audit_runs(paths, exp) -> tuple[bool, dict]:

    sel_f = _cdir(paths) / "lambda_selection.json"
    sel = read_json(sel_f)["selected"] if sel_f.exists() else None
    problems, ok_runs = [], []
    if sel is None:
        problems.append("lambda_selection.json missing")
    for arm, seed, lam in expected_runs(exp, sel):
        rdir = Path(paths["work_dir"]) / "runs" / arm / f"lam{lam:g}" / f"seed{seed}"
        r = _rec(paths, arm, seed, lam)
        if r is None:
            problems.append(f"MISSING {arm} s{seed} lam{lam:g}")
            continue
        if r.get("state") not in ("TRAINING_COMPLETED", "COMPLETED"):
            problems.append(f"STATE {r.get('state')} {arm} s{seed} lam{lam:g}")
            continue
        cks = next((h["checksums"] for h in reversed(r["history"]) if "checksums" in h), {})
        bad = [f for f, d in cks.items() if not (rdir / f).exists() or sha256_file(rdir / f) != d]
        if bad or not cks:
            problems.append(f"CHECKSUM {arm} s{seed} lam{lam:g}: {bad or 'no checksums'}")
            continue
        ok_runs.append((arm, seed, lam))
    report = {"time": now_iso(), "n_expected": len(expected_runs(exp, sel)), "n_ok": len(ok_runs), "problems": problems,
              "counts": count_summary(exp, sel) if sel else None}
    if not problems:
        for arm, seed, lam in ok_runs:
            rdir = Path(paths["work_dir"]) / "runs" / arm / f"lam{lam:g}" / f"seed{seed}"
            rec = RunRecord(rdir, read_json(rdir / "run_record.json")["run_id"], {})
            if rec.data.get("state") != "COMPLETED":
                rec.event("COMPLETED", note="validation-level artifacts verified by run audit")
    write_json(_cdir(paths) / "run_audit.json", report)
    return not problems, report


def freeze_checkpoint(paths, exp, config_dir: Path) -> dict:
    ok, rep = audit_runs(paths, exp)
    if not ok:
        raise RuntimeError(f"run audit failed; cannot freeze: {rep['problems'][:10]}")
    sel = read_json(_cdir(paths) / "lambda_selection.json")
    runs, cks = [], {}
    for arm, seed, lam in reported_runs(exp, sel["selected"]):
        p = Path(paths["work_dir"]) / "runs" / arm / f"lam{lam:g}" / f"seed{seed}" / "best.pt"
        cks[str(p.resolve())] = sha256_file(p)
        runs.append({"arm": arm, "seed": seed, "lam": lam, "checkpoint": str(p.resolve())})
    plan = [PKG_ROOT / "docs" / f for f in ("PROTOCOL.md", "ANALYSIS_PLAN.md",
                                             "OBJECTIVE.md")]
    man = {"gate": "checkpoint", "status": "FROZEN", "frozen": now_iso(), "lambda_selection": sel, "runs": runs, "checkpoints": cks,
           "analysis_plan_sha256": {p.name: sha256_file(p) for p in plan if p.exists()},
           "experiment_sha256": sha256_file(config_dir / "experiment.yaml"),
           "cohort_manifest_sha256": sha256_file(integrity_dir(paths) / "cohort_manifest.json")}
    write_json(integrity_dir(paths) / "checkpoint_manifest.json", man)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["matrix", "select_lambda", "audit", "freeze_checkpoint", "counts"])
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--config-dir", default=str(PKG_ROOT / "config"))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args(argv)
    paths, cd = load_paths(a.paths), Path(a.config_dir)
    exp = load_yaml(cd / "experiment.yaml")
    if a.cmd == "matrix":
        print(write_matrix(paths, exp, a.phase))
    elif a.cmd == "select_lambda":
        print(select_lambda(paths, exp))
    elif a.cmd == "audit":
        ok, rep = audit_runs(paths, exp)
        print(rep)
        return 0 if ok else 1
    elif a.cmd == "freeze_checkpoint":
        print({k: v for k, v in freeze_checkpoint(paths, exp, cd).items() if k != "checkpoints"})
    elif a.cmd == "counts":
        print(count_summary(exp, {"F11": 0.0, "F01": 0.0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
