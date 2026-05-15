from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

_MODEL_DIR = Path("models/Temporal/v2/output/lightgbm")


@dataclass(frozen=True)
class PipelineConfig:
    csv_path: Path = Path("dataset/temporal/output/cleaned_human_readable.csv")
    model_path: Path = _MODEL_DIR / "lgb_smoteenn_model.txt"
    feature_groups_path: Path = Path("slm_shap_pipeline/feature_groups.json")
    output_base: Path = Path("outputs/slm_shap_pipeline")
    cache_dir: Path = Path("slm_shap_pipeline/cache")
    top_k: int = 5
    sign_epsilon: float = 1e-6
    month_of_prediction: int = 12
    random_seed: int = 42
    train_frac: float = 0.7
    val_frac: float = 0.2
    test_frac: float = 0.1
    provider: str = "cli"
    gemini_model: str = "gemini-2.5-pro"
    gemini_timeout_seconds: int = 120
    google_api_key_env: str = "GOOGLE_API_KEY"
    medgemma_model_path: Path = Path("models/medgemma-1.5-4b-it-IQ4_XS.gguf")
    medgemma_n_ctx: int = 2048
    medgemma_max_tokens: int = 1536
