from evaluation.slm_shap_faithfulness.feature_map import canonicalize_feature


def test_canonicalize_temporal_aliases():
    assert canonicalize_feature("Days To Treatment") == "Days_To_Treatment"
    assert canonicalize_feature("Registration Group") == "Registration_Group"
    assert canonicalize_feature("Unknown Field") is None
