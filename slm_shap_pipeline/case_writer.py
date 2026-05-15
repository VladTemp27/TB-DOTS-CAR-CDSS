from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_case(
    patient_id: str,
    condition: str,
    explanation: str,
    aggregated_shap: dict[str, dict],
    model_prediction: float,
    metadata: dict,
    out_dir: Path,
    month_of_prediction: int = 12,
) -> Path:
    """Write a case_<patient_id>.json file for one patient/condition pair.

    shap_values: signed_sum per clinical group (positive = increases failure risk).
    shap_metadata: full per-group aggregation details (abs_sum, sign, constituents).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    case_file = out_dir / f"case_{patient_id}.json"

    shap_values = {base: data["signed_sum"] for base, data in aggregated_shap.items()}
    shap_metadata = {base: data for base, data in aggregated_shap.items()}

    payload = {
        "patient_id": patient_id,
        "condition": condition,
        "month_of_prediction": month_of_prediction,
        "model_prediction": model_prediction,
        "explanation": explanation,
        "shap_values": dict(sorted(shap_values.items())),
        "shap_metadata": shap_metadata,
        "prediction_metadata": metadata,
    }

    case_file.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return case_file


def write_manifest(
    out_dir: Path,
    condition: str,
    config_snapshot: dict,
    stats: dict,
) -> Path:
    """Write a manifest.json summarising the full pipeline run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = out_dir / "manifest.json"
    manifest = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "config_snapshot": config_snapshot,
        "stats": stats,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))
    return manifest_file
