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
    assert out["composite"] >= 0.60


def test_score_case_empty_claims_fails():
    truth = [{"feature": "Age", "sign": "increase", "magnitude_band": "strong"}]
    out = score_case(truth, [])
    assert out["passed"] is False
    assert out["feature_f1"] == 0.0


def test_score_case_wrong_sign_lowers_score():
    truth = [{"feature": "Age", "sign": "increase", "magnitude_band": "strong"}]
    claims = [{"feature": "Age", "direction": "decrease", "magnitude": "strong"}]
    out = score_case(truth, claims)
    assert out["sign_accuracy"] == 0.0
    assert out["passed"] is False


def test_score_case_missing_required_keys_raises():
    import pytest
    truth = [{"feature": "Age"}]  # missing "sign" and "magnitude_band"
    claims = [{"feature": "Age", "direction": "increase", "magnitude": "strong"}]
    with pytest.raises(ValueError):
        score_case(truth, claims)
