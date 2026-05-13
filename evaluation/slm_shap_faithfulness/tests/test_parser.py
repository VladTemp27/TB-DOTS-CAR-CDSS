from evaluation.slm_shap_faithfulness.parser import parse_explanation


def test_parse_extracts_feature_direction_magnitude():
    parsed = parse_explanation(
        "Age strongly increases risk while Days To Treatment moderately reduces risk."
    )
    assert parsed["claims"][0]["feature"] == "Age"
    assert parsed["claims"][0]["direction"] == "increase"
    assert parsed["claims"][0]["magnitude"] == "strong"
