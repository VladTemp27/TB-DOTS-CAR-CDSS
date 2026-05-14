from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from slm_shap_pipeline.config import PipelineConfig


@runtime_checkable
class SLMProvider(Protocol):
    name: str

    def generate(self, prompt: str) -> str:
        """Return the SLM's text response. Raise RuntimeError on failure."""
        ...


def make_provider(config: PipelineConfig) -> SLMProvider:
    if config.provider == "cli":
        from slm_shap_pipeline.providers.cli import GeminiCLIProvider
        return GeminiCLIProvider(model=config.gemini_model, timeout=config.gemini_timeout_seconds)
    if config.provider == "api":
        from slm_shap_pipeline.providers.api import GoogleAPIProvider
        api_key = os.environ.get(config.google_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{config.google_api_key_env} not set; cannot construct GoogleAPIProvider. "
                "Set the env var or use --provider cli."
            )
        return GoogleAPIProvider(api_key=api_key, model=config.gemini_model)
    if config.provider == "medgemma":
        from slm_shap_pipeline.providers.medgemma import MedGemmaProvider
        return MedGemmaProvider(
            model_path=config.medgemma_model_path,
            n_ctx=config.medgemma_n_ctx,
            max_tokens=config.medgemma_max_tokens,
        )
    raise ValueError(f"Unknown provider: {config.provider!r}")
