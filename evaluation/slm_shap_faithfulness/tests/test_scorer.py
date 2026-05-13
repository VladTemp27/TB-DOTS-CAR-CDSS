from evaluation.slm_shap_faithfulness.scorer import score_case


def test_score_case_applies_conservative_gate():
    truth = [
        {"feature": "Age", "sign": "increase", "magnitude_band": "strong"},
        {"feature": "Days_To_Treatment", "sign": "decrease", "magnitude_band": "moderate"},
    ]
    claims = [
        {"feature": "Age", "direction": "increase", "magnitude": "strong"},
        {"feature": "Days_To_Treatment", "direction": "decrease", "magnitude": "moderate"},
    ]
    out = score_case(truth, claims)
    assert out["passed"] is True
    assert out["composite"] >= 0.82
