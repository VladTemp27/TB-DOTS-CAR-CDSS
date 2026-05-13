from evaluation.slm_shap_faithfulness.config import BenchmarkConfig


def test_conservative_defaults_and_mode_choices():
    cfg = BenchmarkConfig()
    assert cfg.top_k == 5
    assert cfg.threshold_feature_f1 == 0.80
    assert cfg.threshold_sign_accuracy == 0.90
    assert cfg.threshold_magnitude_accuracy == 0.75
    assert cfg.threshold_composite == 0.82
    assert cfg.parser_coverage_min == 0.95
    assert cfg.allowed_modes == ("artifact", "runtime")
    assert cfg.weight_feature_f1 == 0.45
    assert cfg.weight_sign_accuracy == 0.35
    assert cfg.weight_magnitude_accuracy == 0.20
