import pytest
from slm_shap_pipeline.feature_aggregator import aggregate_shap, build_groups, validate_coverage, top_k

MOCK_GROUPS = {
    "Treatment_Adherence": ["M0_pct_adherence", "M1_pct_adherence", "mean_pct_adherence"],
    "Age": ["age"],
    "Sex": ["sex_Male", "sex_Female"],
}

MOCK_RAW_SHAP = {
    "M0_pct_adherence": 0.1,
    "M1_pct_adherence": -0.3,
    "mean_pct_adherence": 0.2,
    "age": 0.5,
    "sex_Male": 0.3,
    "sex_Female": -0.1,
}


def test_aggregate_signed_sum():
    result = aggregate_shap(MOCK_RAW_SHAP, MOCK_GROUPS, epsilon=1e-6)
    # 0.1 + (-0.3) + 0.2 = 0.0 → mixed
    assert result["Treatment_Adherence"]["sign"] == "mixed"
    assert abs(result["Treatment_Adherence"]["signed_sum"]) < 1e-5


def test_aggregate_abs_sum():
    result = aggregate_shap(MOCK_RAW_SHAP, MOCK_GROUPS, epsilon=1e-6)
    # |0.1| + |-0.3| + |0.2| = 0.6
    assert abs(result["Treatment_Adherence"]["abs_sum"] - 0.6) < 1e-6


def test_one_hot_dummies_collapse_to_single_group():
    result = aggregate_shap(MOCK_RAW_SHAP, MOCK_GROUPS, epsilon=1e-6)
    assert "Sex" in result
    assert abs(result["Sex"]["signed_sum"] - 0.2) < 1e-6
    assert abs(result["Sex"]["abs_sum"] - 0.4) < 1e-6
    assert result["Sex"]["sign"] == "+"


def test_aggregate_positive_sign():
    result = aggregate_shap(MOCK_RAW_SHAP, MOCK_GROUPS, epsilon=1e-6)
    assert result["Age"]["sign"] == "+"
    assert abs(result["Age"]["signed_sum"] - 0.5) < 1e-6


def test_validate_coverage_raises_on_missing():
    with pytest.raises(ValueError, match="ungrouped"):
        validate_coverage({"feat_x": 0.1}, MOCK_GROUPS)


def test_validate_coverage_passes_when_complete():
    validate_coverage(MOCK_RAW_SHAP, MOCK_GROUPS)  # no exception


def test_constituents_stored():
    result = aggregate_shap(MOCK_RAW_SHAP, MOCK_GROUPS, epsilon=1e-6)
    assert "M0_pct_adherence" in result["Treatment_Adherence"]["constituents"]
    assert "sex_Male" in result["Sex"]["constituents"]
    assert "sex_Female" in result["Sex"]["constituents"]


def test_top_k_returns_sorted_by_abs_sum():
    result = aggregate_shap(MOCK_RAW_SHAP, MOCK_GROUPS, epsilon=1e-6)
    ranked = top_k(result, k=2)
    assert len(ranked) == 2
    assert ranked[0][1]["abs_sum"] >= ranked[1][1]["abs_sum"]
