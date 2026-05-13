import pandas as pd
from evaluation.slm_shap_faithfulness.shap_truth import build_truth_rows


def test_truth_rows_rank_sign_band():
    row = pd.Series({"A": 0.8, "B": -0.6, "C": 0.2})
    out = build_truth_rows(row, top_k=3)
    assert [x["feature"] for x in out] == ["A", "B", "C"]
    assert [x["sign"] for x in out] == ["increase", "decrease", "increase"]
    assert [x["rank"] for x in out] == [1, 2, 3]
