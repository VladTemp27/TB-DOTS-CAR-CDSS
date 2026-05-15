from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from slm_shap_pipeline.providers.base import SLMProvider


def make_cache_key(
    patient_id: str,
    condition: str,
    month_of_prediction: int,
    prompt: str,
    model_hash: str,
    scaler_static_hash: str,
    scaler_temporal_hash: str,
    feature_groups_hash: str,
) -> str:
    """SHA-256 hash of the 8 inputs that determine the SLM response.

    Intentionally provider-agnostic — a cached CLI response is reusable
    when the run flips to the API provider. Provider provenance is
    recorded inside the cache JSON body, not in the key.
    """
    payload = "|".join([
        patient_id, condition, str(month_of_prediction), prompt,
        model_hash, scaler_static_hash, scaler_temporal_hash, feature_groups_hash,
    ])
    return hashlib.sha256(payload.encode()).hexdigest()


def call_slm(
    prompt: str,
    cache_key: str,
    provider: SLMProvider,
    *,
    cache_dir: Path = Path("slm_shap_pipeline/cache"),
    dry_run: bool = False,
) -> str:
    """Return cached response, dry-run stub, or fresh provider output.

    On cache miss, delegates to provider.generate() and persists the
    response (+ provider.name) to the cache JSON. Failures propagate
    cleanly; the cache file is never written on error.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())["response"]

    if dry_run:
        response = f"[DRY RUN] Patient prompt received ({len(prompt)} chars). Stub explanation."
    else:
        response = provider.generate(prompt)  # raises on failure → no cache write

    cache_file.write_text(
        json.dumps(
            {
                "response": response,
                "provider": provider.name,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    return response
