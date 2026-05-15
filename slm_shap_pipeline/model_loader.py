from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb

from slm_shap_pipeline.config import PipelineConfig


@dataclass
class ModelBundle:
    booster: lgb.Booster
    model_hash: str    # sha256 of lgb_smoteenn_model.txt


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model_bundle(config: PipelineConfig) -> ModelBundle:
    if not config.model_path.exists():
        raise FileNotFoundError(f"V2 booster not found: {config.model_path}")
    booster = lgb.Booster(model_file=str(config.model_path))
    return ModelBundle(
        booster=booster,
        model_hash=_sha256(config.model_path),
    )
