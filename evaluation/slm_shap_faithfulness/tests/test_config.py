from dataclasses import FrozenInstanceError

import pytest

from evaluation.slm_shap_faithfulness.config import BenchmarkConfig


def test_conservative_defaults_and_mode_choices():
    cfg = BenchmarkConfig()
    assert cfg.top_k == 5
    assert cfg.threshold_feature_f1 == 0.60
    assert cfg.threshold_sign_accuracy == 0.60
    assert cfg.threshold_magnitude_accuracy == 0.50
    assert cfg.threshold_composite == 0.60
    assert cfg.parser_coverage_min == 0.95
    assert cfg.allowed_modes == ("artifact", "runtime")
    assert cfg.weight_feature_f1 == 0.45
    assert cfg.weight_sign_accuracy == 0.35
    assert cfg.weight_magnitude_accuracy == 0.20


def test_config_is_immutable():
    cfg = BenchmarkConfig()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        cfg.top_k = 99
