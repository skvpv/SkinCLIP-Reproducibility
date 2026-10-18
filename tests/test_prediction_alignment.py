import pandas as pd
import pytest
from skinclip.analysis.stats import assert_prediction_alignment

def test_alignment_rejects_same_class_swap():
    a = pd.DataFrame({"image_id": ["a", "b"], "cluster_id": ["x", "y"], "label_idx": [0, 0]})
    assert_prediction_alignment(a, a.copy())
    with pytest.raises(ValueError):
        assert_prediction_alignment(a, a.iloc[::-1].reset_index(drop=True))

def test_alignment_rejects_duplicate_ids():
    a = pd.DataFrame({"image_id": ["a", "a"], "cluster_id": ["x", "y"], "label_idx": [0, 0]})
    with pytest.raises(ValueError):
        assert_prediction_alignment(a, a)
