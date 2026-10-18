import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from skinclip.common import FirewallError, IntegrityError, load_paths, read_json


class FakeBC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.img = torch.nn.Linear(3 * 224 * 224, 16)
        self.txt = torch.nn.Embedding(1000, 16)
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.0))
        self.visual = torch.nn.Sequential(torch.nn.Flatten(), self.img)

    def encode_image(self, x):
        return self.img(x.flatten(1))

    def encode_text(self, t):
        return self.txt(t).mean(1)


def fake_load(model_id, device):
    return FakeBC().eval(), (lambda prompts, context_length=256: torch.randint(0, 1000, (len(prompts), 8)))


@pytest.fixture(scope="module")
def env(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("e2e")
    traps = synthetic.make(root)
    pf = synthetic.make_paths(root)
    cd = synthetic.make_config(root, approve=False)
    monkeypatch_module.setenv("SKINCLIP_PATHS", str(pf))
    import skinclip.models.biomedclip as bcm

    monkeypatch_module.setattr(bcm, "load_biomedclip", fake_load)


    monkeypatch_module.setattr(torch.cuda, "is_available", lambda: False)
    return {"root": root, "traps": traps, "paths": load_paths(pf), "cd": cd}


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_full_pipeline(env):
    from skinclip.audit import cohorts, neardup, steps
    from skinclip.engine import campaign, prep
    from skinclip.engine.evaluate import evaluate_all
    from skinclip.engine.train import Preempted, train_run

    paths, cd, traps = env["paths"], env["cd"], env["traps"]
    work = Path(paths["work_dir"])
    exp = __import__("yaml").safe_load((cd / "experiment.yaml").read_text())


    steps.inventory_inventory(paths, 2)
    s2 = steps.provenance_id_lesion_overlap(paths)["ISIC2019_train"]
    assert s2["n_id_match"] == len(traps["exact_id"]) and s2["n_lesion_match_without_id_match"] == 1
    steps.exact_overlap_hashes(paths, 2)
    neardup.compute_embeddings(paths, "vit_tiny_patch16_224", "cpu", pretrained=False, img_size=32)
    s3 = neardup.similarity_screen_near_duplicates(paths, 1e-2, 5e-2, 0, 1)
    cal = s3["ISIC2019_train"]["calibration"]
    assert max(c["sensitivity_review_on_ID_pairs"] for c in cal.values()) == 1.0
    assert s3["ISIC2019_train"]["null_union_flag_rate_review"] <= 0.15
    cand = pd.read_csv(work / "audit" / "similarity_screen_near_duplicate_candidates.csv")
    found = set(cand["image_id"])
    assert set(traps["near_dup"]) <= found, f"missed near-duplicates: {set(traps['near_dup']) - found}"
    neardup.ham_internal_links(paths)
    class_counts = steps.class_counts_label_crosstab(paths)
    assert class_counts["ISIC2019_train"]["class_counts"]["SCC"] == 14
    steps.metadata_audit_lesion_structure(paths)
    site_mapping = steps.site_mapping_metadata(paths, cd / "site_harmonisation.yaml")
    assert site_mapping["HAM"]["unmapped_site_values"] == [] and site_mapping["ISIC2019_train"]["unmapped_site_values"] == []


    label_free_screen = neardup.label_free_screen_label_free_null(paths, 1e-2, 5e-2)
    assert "n_both" in label_free_screen["ISIC2019_train"]
    cohorts.cohort_report(paths, cd)
    draft = json.loads((work / "cohorts" / "claims_and_summary.json").read_text())
    adj = work / "review" / "near_duplicates_TO_ADJUDICATE.csv"
    adj_hash = __import__("hashlib").sha256(adj.read_bytes()).hexdigest()
    with pytest.raises(IntegrityError):
        cohorts.freeze_cohort(paths, cd)
    synthetic.approve_configs(cd)
    synthetic.approve_precautionary(cd, draft)
    synthetic.sign_cohort(cd, "precautionary")
    man = cohorts.freeze_cohort(paths, cd)
    assert man["near_duplicate_mode"] == "precautionary" and man["verification"]["identical_to_initial_draft"]
    assert man["verification"]["expected_counts_match_policy"] and man["verification"]["candidate_decisions_unchanged"]
    assert __import__("hashlib").sha256(adj.read_bytes()).hexdigest() == adj_hash
    assert (pd.read_csv(adj, dtype=str).fillna("")["decision"] == "").all()
    claims = man["summary"]["claims"]
    assert claims["patient_disjoint"].startswith("NOT CLAIMABLE")
    assert claims["e1_precautionary_exclusions"]["confirmed_duplicate"] is False
    assert "pessimistic" not in json.dumps(claims).lower()
    assert "not assumed to be conservative" in claims["e1_selection_effect"]["statement"]
    assert "label" in claims["e1_threshold_calibration_disclosure"]["statement"]
    assert isinstance(claims["e1_threshold_calibration_disclosure"]["label_free_sensitivity"], dict)
    ex = pd.read_csv(work / "cohorts" / "e1_exclusions.csv")
    assert set(ex["category"]) <= {"confirmed", "precautionary"}
    assert not ex.loc[ex.category == "precautionary", "confirmed_duplicate"].astype(str).str.lower().eq("true").any()
    assert set(ex["exclusion_reason"]) <= set(cohorts.CONFIRMED) | set(cohorts.PRECAUTIONARY)
    assert not any("duplicate" == r for r in ex["exclusion_reason"])
    assert (work / "cohort_FREEZE_REPORT.md").exists() and (work / cohorts.DRAFT_ARCHIVE / "ARCHIVE_SHA256.json").exists()
    e1 = pd.read_csv(work / "cohorts" / "e1.csv")
    ham = pd.read_csv(work / "cohorts" / "ham.csv")
    assert set(e1["image_id"]) == set(pd.read_csv(work / cohorts.DRAFT_ARCHIVE / "e1.csv")["image_id"])
    assert not set(e1["image_id"]) & set(ham["image_id"])
    assert not set(traps["near_dup"]) & set(e1["image_id"]) and not set(traps["lesion_only"]) & set(e1["image_id"])
    assert "SCC" not in set(e1["label_raw"]) and set(pd.read_csv(work / "cohorts" / "e1_oot.csv")["label_raw"]) == {"SCC"}
    assert ham.groupby("lesion_id")["split"].nunique().max() == 1


    prep.text_bank(paths, cd)
    prep.cache_cohort(paths, cd, "ham", 2)
    prep.bc_features_trainval(paths, cd, "cpu")


    smoke = {"epochs": 2, "max_batches": 4, "num_workers": 0}
    r_a = work / "t6a"
    train_run("F11", 0, 0.5, paths, cd, {**smoke}, out_root=r_a)
    r_b = work / "t6b"
    with pytest.raises(Preempted):
        train_run("F11", 0, 0.5, paths, cd, {**smoke, "preempt_after_epoch": 0}, out_root=r_b)
    rec = train_run("F11", 0, 0.5, paths, cd, {**smoke, "preempt_after_epoch": 0}, out_root=r_b)
    assert rec["state"] == "TRAINING_COMPLETED"
    wa = torch.load(r_a / "F11/lam0.5/seed0/last.pt", weights_only=False)["model"]
    wb = torch.load(r_b / "F11/lam0.5/seed0/last.pt", weights_only=False)["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa), "resume is not deterministic"


    orig = (cd / "prompts.yaml").read_text()
    (cd / "prompts.yaml").write_text(orig.replace("melanoma,", "melanoma ,"))
    with pytest.raises(Exception):
        train_run("F11", 1, 0.0, paths, cd)
    (cd / "prompts.yaml").write_text(orig)


    campaign.write_matrix(paths, exp, 1)
    inventory = pd.read_csv(work / "campaign" / "phase1.tsv", sep="\t")
    assert len(pd.read_csv(work / "campaign" / "phase1_gpu.tsv", sep="\t")) == 9
    for r in inventory.itertuples():
        train_run(r.arm, int(r.seed), float(r.lam), paths, cd)
    sel = campaign.select_lambda(paths, exp)
    assert set(sel["selected"]) == {"F11", "F01"}
    campaign.write_matrix(paths, exp, 2)
    provenance = pd.read_csv(work / "campaign" / "phase2_gpu.tsv", sep="\t")
    assert len(provenance) == 10
    with pytest.raises(FirewallError):
        evaluate_all(paths, cd)
    for r in provenance.itertuples():
        train_run(r.arm, int(r.seed), float(r.lam), paths, cd)
    cnt = campaign.count_summary(exp, sel["selected"])
    assert (cnt["full_finetune_runs"], cnt["light_probe_runs"], cnt["cpu_meta_only_runs"]) == (19, 3, 3)
    ok, rep = campaign.audit_runs(paths, exp)
    assert ok, rep["problems"]
    checkpoint = campaign.freeze_checkpoint(paths, exp, cd)
    assert len(checkpoint["runs"]) == 21


    done = evaluate_all(paths, cd)
    assert all(v == "evaluated" for v in done.values())
    assert evaluate_all(paths, cd)["F11_s0"].startswith("exists")
    from skinclip.analysis import cost, figures, interpret, stats

    st = stats.run_stats(paths, cd)
    assert set(st["splits"]["e1"]["confirmatory"]) == {"H1", "H2", "H3"}
    assert all("p_holm" in v for v in st["splits"]["e1"]["confirmatory"].values())
    c = cost.run_cost(paths, cd)
    assert {"F11", "MB", "BiomedCLIP_visual"} <= set(c["arm"])
    s = interpret.run_interpret(paths, cd)
    assert s["splits"]["e1"]["items"]
    figures.main(["--paths", str(Path(env["root"]) / "paths.yaml")])
    assert (work / "results" / "figures" / "fig_confusion_F11_e1.png").exists()
