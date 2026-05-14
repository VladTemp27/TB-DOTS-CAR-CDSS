from evaluation.slm_shap_faithfulness.regression import compare_runs


def test_compare_runs_deltas():
    out = compare_runs({"pass_rate": 0.8}, {"pass_rate": 0.7})
    assert abs(out["pass_rate_delta"] - (-0.1)) < 1e-6


def test_compare_runs_improvement():
    out = compare_runs({"pass_rate": 0.6}, {"pass_rate": 0.9})
    assert out["pass_rate_delta"] > 0
