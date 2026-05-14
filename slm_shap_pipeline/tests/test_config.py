from slm_shap_pipeline.config import PipelineConfig
import dataclasses
import pytest


def test_default_provider_is_cli():
    cfg = PipelineConfig()
    assert cfg.provider == "cli"
    assert cfg.gemini_model.startswith("gemini-")


def test_default_month_is_12():
    cfg = PipelineConfig()
    assert cfg.month_of_prediction == 12


def test_config_is_frozen():
    cfg = PipelineConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.provider = "api"  # type: ignore[misc]


def test_provider_override():
    cfg = PipelineConfig(provider="api")
    assert cfg.provider == "api"
