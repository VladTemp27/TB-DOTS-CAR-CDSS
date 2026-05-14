import pytest
from pathlib import Path
from slm_shap_pipeline.config import PipelineConfig
from slm_shap_pipeline.model_loader import load_model_bundle, ModelBundle

MODEL_PATH = Path("models/Temporal/v2/output/lightgbm/lgb_smoteenn_model.txt")
SKIP_IF_MISSING = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="V2 model not present"
)


@SKIP_IF_MISSING
def test_load_model_bundle_returns_bundle():
    cfg = PipelineConfig()
    bundle = load_model_bundle(cfg)
    assert isinstance(bundle, ModelBundle)
    assert hasattr(bundle.booster, "predict")


@SKIP_IF_MISSING
def test_bundle_exposes_model_hash():
    cfg = PipelineConfig()
    bundle = load_model_bundle(cfg)
    assert len(bundle.model_hash) == 64        # sha256 hex


@SKIP_IF_MISSING
def test_booster_has_399_features():
    cfg = PipelineConfig()
    bundle = load_model_bundle(cfg)
    names = bundle.booster.feature_name()
    assert len(names) == 399, f"Expected 399 features, got {len(names)}"
    assert names[0] == "Column_0"             # V2 uses anonymous Column_N names
