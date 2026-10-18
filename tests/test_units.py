import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from skinclip import NUM_CLASSES
from skinclip.analysis.stats import cluster_weights, holm, mcnemar_one_per_cluster, paired
from skinclip.audit.neardup import _threshold, hamming_matrix
from skinclip.common import FirewallError, assert_split_access
from skinclip.data.metadata import META_DIM, encode_metadata
from skinclip.data.splits import split_groups, stratified_group_split
from skinclip.models.core import LinearProbe, MetaBlock, MetaBlockNet, MetaOnlyNet, PrototypeNet


class TinyEnc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(3 * 8 * 8, 32)

    def forward(self, x):
        return self.l(x.flatten(1))


@pytest.mark.parametrize("use_meta,protos", [(True, "text"), (True, "learned"), (False, "text"), (False, "learned")])
def test_T1_label_free_inference(use_meta, protos):
    torch.manual_seed(0)
    m = PrototypeNet(TinyEnc(), 32, use_meta, protos, NUM_CLASSES, 16, 8, 16, 0.1,
                     text_H=torch.randn(NUM_CLASSES, 24) if protos == "text" else None).eval()

    for fwd in (m.forward, MetaOnlyNet(NUM_CLASSES).forward, LinearProbe(8, NUM_CLASSES).forward, MetaBlockNet.forward):
        assert not {"label", "labels", "y", "target"} & set(inspect.signature(fwd).parameters)

    from skinclip.engine.train import model_inputs

    x, meta = torch.randn(10, 3, 8, 8), torch.randn(10, META_DIM)
    b1 = {"image": x, "meta": meta, "label": torch.arange(10) % 7, "index": torch.arange(10)}
    b2 = {**b1, "label": torch.randint(0, 7, (10,))}
    with torch.no_grad():
        o1 = m(**model_inputs(b1, torch.device("cpu")))["logits"]
        o2 = m(**model_inputs(b2, torch.device("cpu")))["logits"]
    assert torch.equal(o1, o2)
    assert set(model_inputs(b1, torch.device("cpu"))) == {"image", "meta"}


def test_text_prototypes_use_frozen_H_and_scale_clamped():
    H = torch.randn(NUM_CLASSES, 24)
    m = PrototypeNet(TinyEnc(), 32, True, "text", NUM_CLASSES, 16, 8, 16, text_H=H)
    assert "text_H" in dict(m.named_buffers()) and "text_H" not in dict(m.named_parameters())
    with torch.no_grad():
        m.theta.fill_(10.0)
    assert m.scale().item() == 100.0
    p = m.prototypes()
    assert torch.allclose(p.norm(dim=-1), torch.ones(NUM_CLASSES), atol=1e-5)


def test_metablock_matches_official_formula():
    torch.manual_seed(0)
    mb = MetaBlock(4, 3).eval()
    V, U = torch.randn(2, 4, 5), torch.randn(2, 3)
    ref = torch.sigmoid(torch.tanh(V * mb.fb(U).unsqueeze(-1)) + mb.gb(U).unsqueeze(-1))
    assert torch.allclose(mb(V, U), ref)


def test_T3_firewall_blocks_eval_splits():
    for s in ("internal_test", "e1", "e1_oot"):
        with pytest.raises(FirewallError):
            assert_split_access(s, "training", {"work_dir": "/nonexistent"})
        with pytest.raises(FirewallError):
            assert_split_access(s, "evaluation", {"work_dir": "/tmp/__no_checkpoint__"})
    assert_split_access("train", "training") is None


def test_metadata_encoding_layout():
    df = pd.DataFrame({"age": [50, np.nan], "sex_raw": ["male", "unknown"], "site_group": ["torso", "unknown"]})
    x = encode_metadata(df)
    assert x.shape == (2, META_DIM) == (2, 12)
    assert x[0, 0] == 0.5 and x[0, 1] == 0 and x[0, 2] == 1 and x[0, 6] == 1
    assert x[1, 0] == 0 and x[1, 1] == 1 and x[1, 4] == 1 and x[1, 11] == 1


def test_T2_grouped_split_is_lesion_disjoint_and_links_merge_groups():
    rng = np.random.default_rng(0)
    rows = [{"image_id": f"i{i}", "lesion_id": f"L{i // 2}", "label": ["a", "b", "c"][(i // 2) % 3]} for i in range(300)]
    df = pd.DataFrame(rows)
    links = pd.DataFrame({"image_a": ["i0"], "image_b": ["i10"], "method": ["md5"]})
    df["split_group"] = split_groups(df, links)
    assert df.loc[df.image_id == "i0", "split_group"].item() == df.loc[df.image_id == "i10", "split_group"].item()
    df["split"], rep = stratified_group_split(df, "label", "split_group", (0.7, 0.15, 0.15), 1)
    assert df.groupby("split_group")["split"].nunique().max() == 1
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.05 and abs(frac["val"] - 0.15) < 0.05
    assert rng is not None


def test_hamming_and_thresholds():
    a = np.array([0, 0xFF, 2**64 - 1], dtype=np.uint64)
    D = hamming_matrix(a, a)
    assert D[0, 1] == 8 and D[0, 2] == 64 and D[1, 1] == 0
    neg = np.arange(0, 64).repeat(100)
    t = _threshold(neg, 0.01, False)
    assert (neg <= t).mean() <= 0.01
    c = _threshold(np.linspace(0, 1, 10001), 1e-3, True)
    assert (np.linspace(0, 1, 10001) >= c).mean() <= 1e-3


def test_stats_cluster_bootstrap_paired_holm_mcnemar():
    n = 120
    df = pd.DataFrame({"image_id": [f"img_{i}" for i in range(n)], "cluster_id": [f"c{i // 3}" for i in range(n)], "label_idx": [(i // 3) % 7 for i in range(n)]})
    W = cluster_weights(df, 50, 0)
    assert W.shape == (50, n) and np.allclose(W.sum(1), n)
    rng = np.random.default_rng(0)
    probs = rng.random((n, 7))
    pred = pd.concat([df, pd.DataFrame(probs, columns=[f"p_{c}" for c in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]])], axis=1)
    r = paired([pred], [pred], W)
    assert r["delta"] == 0 and r["ci95"] == [0.0, 0.0]
    h = holm({"H1": 0.01, "H2": 0.04, "H3": 0.03})
    assert h["H1"] == 0.03 and h["H3"] == 0.06 and h["H2"] == 0.06
    m = mcnemar_one_per_cluster(pred, pred, 0)
    assert m["n_clusters"] == 40 and m["p_exact"] == 1.0
