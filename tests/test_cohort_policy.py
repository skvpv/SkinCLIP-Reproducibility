import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

import synthetic
from skinclip.common import IntegrityError, load_paths, read_json


def _build_draft(root: Path):
    synthetic.make(root)
    pf = synthetic.make_paths(root)
    cd = synthetic.make_config(root, approve=False)
    from skinclip.audit import cohorts, neardup, steps

    p = load_paths(pf)
    steps.inventory_inventory(p, 1)
    steps.provenance_id_lesion_overlap(p)
    steps.exact_overlap_hashes(p, 1)
    neardup.compute_embeddings(p, "vit_tiny_patch16_224", "cpu", pretrained=False, img_size=32)
    neardup.similarity_screen_near_duplicates(p, 1e-2, 5e-2, 0, 1)
    neardup.ham_internal_links(p)
    cohorts.cohort_report(p, cd)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohortbase")
    _build_draft(root)
    return root


@pytest.fixture
def env(base, tmp_path):
    root = tmp_path / "w"
    shutil.copytree(base, root)
    pf = synthetic.make_paths(root)
    cd = root / "config"
    p = load_paths(pf)
    draft = read_json(Path(p["work_dir"]) / "cohorts" / "claims_and_summary.json")
    synthetic.approve_configs(cd)
    synthetic.approve_precautionary(cd, draft)
    synthetic.sign_cohort(cd, "precautionary")
    return p, cd, Path(p["work_dir"])


def test_precautionary_freeze_records_precautionary_not_confirmed(env):
    from skinclip.audit import cohorts

    p, cd, work = env
    man = cohorts.freeze_cohort(p, cd)
    checks = read_json(work / "integrity" / "cohort_freeze_checks.json")["checks"]
    assert checks["identical_to_initial_draft"] and checks["candidate_decisions_unchanged"] and checks["expected_counts_match_policy"]
    ex = pd.read_csv(work / "cohorts" / "e1_exclusions.csv")
    prec = ex[ex.category == "precautionary"]
    assert len(prec) == sum(man["summary"]["claims"]["e1_precautionary_exclusions"]["counts"].values())
    assert set(prec["exclusion_reason"]) <= {"precautionary_similarity_strict", "precautionary_similarity_review",
                                             "precautionary_lesion_expansion"}
    sim = prec[prec.exclusion_reason.str.startswith("precautionary_similarity")]
    assert sim["evidence"].str.contains("->").all()

    arts = " ".join(man["artifacts"])
    for name in ("near_duplicate_policy.yaml", "similarity_screen_near_duplicate_candidates.csv", "near_duplicates_TO_ADJUDICATE.csv",
                 "cohorts_initial_draft", "e1_exclusions.csv"):
        assert name in arts


def test_precautionary_refuses_filled_pair_decisions(env):
    from skinclip.audit import cohorts

    p, cd, work = env
    adj = work / "review" / "near_duplicates_TO_ADJUDICATE.csv"
    a = pd.read_csv(adj, dtype=str).fillna("")
    if len(a) == 0:
        pytest.skip("no review pairs in this synthetic draw")
    a.loc[a.index[0], "decision"] = "duplicate"
    a.to_csv(adj, index=False)
    with pytest.raises(IntegrityError, match="blank"):
        cohorts.freeze_cohort(p, cd)
    assert not (work / "integrity" / "cohort_manifest.json").exists()


def test_precautionary_refuses_count_mismatch(env):
    from skinclip.audit import cohorts

    p, cd, work = env
    pol = yaml.safe_load((cd / "near_duplicate_policy.yaml").read_text())
    pol["expected"]["e1"] += 1
    (cd / "near_duplicate_policy.yaml").write_text(yaml.safe_dump(pol))
    with pytest.raises(IntegrityError, match="verification failed"):
        cohorts.freeze_cohort(p, cd)
    assert not (work / "integrity" / "cohort_manifest.json").exists()


def test_refuses_when_draft_differs_or_archive_tampered(env):
    from skinclip.audit import cohorts

    p, cd, work = env
    e1 = pd.read_csv(work / "cohorts" / "e1.csv")
    e1.iloc[1:].to_csv(work / "cohorts" / "e1.csv", index=False)
    with pytest.raises(IntegrityError, match="verification failed"):
        cohorts.freeze_cohort(p, cd)
    arch = work / cohorts.DRAFT_ARCHIVE / "e1.csv"
    arch.write_text(arch.read_text() + "\n")
    with pytest.raises(IntegrityError, match="archive modified"):
        cohorts.freeze_cohort(p, cd)


def test_unapproved_policy_is_refused(env):
    from skinclip.audit import cohorts

    p, cd, work = env
    pol = yaml.safe_load((cd / "near_duplicate_policy.yaml").read_text())
    pol["status"] = "PROPOSED"
    (cd / "near_duplicate_policy.yaml").write_text(yaml.safe_dump(pol))
    with pytest.raises(IntegrityError, match="precautionary"):
        cohorts.freeze_cohort(p, cd)
