from evaluation.slm_shap_faithfulness.parser import parse_explanation


def test_parse_extracts_feature_direction_magnitude():
    parsed = parse_explanation(
        "Age strongly increases risk while Days To Treatment moderately reduces risk."
    )
    assert parsed["claims"][0]["feature"] == "Age"
    assert parsed["claims"][0]["direction"] == "increase"
    assert parsed["claims"][0]["magnitude"] == "strong"


def test_parse_no_recognized_feature_returns_parse_failed():
    parsed = parse_explanation("The patient has a normal profile.")
    assert parsed["status"] == "parse_failed"
    assert parsed["claims"] == []


def test_parse_feature_without_direction_excluded():
    parsed = parse_explanation("Age is mentioned but nothing else.")
    # "Age" present but no direction word → should not appear in claims
    assert all(c["feature"] != "Age" for c in parsed["claims"])


def test_parse_magnitude_unspecified_when_absent():
    parsed = parse_explanation("Age increases risk.")
    assert parsed["claims"][0]["direction"] == "increase"
    assert parsed["claims"][0]["magnitude"] == "unspecified"
