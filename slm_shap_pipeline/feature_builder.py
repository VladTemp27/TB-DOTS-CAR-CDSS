from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import models.Temporal.model_utils as mu
from slm_shap_pipeline.data_loader import PatientRow


def build_features_for_patient(
    patient: PatientRow,
    up_to_month: int = 12,
) -> tuple[np.ndarray, list[str]]:
    """Build the flat feature matrix for a single patient at up_to_month.

    Returns (X_flat shape (1, n_features), feature_names list).
    Feature names are semantic (e.g., "M0_pct_adherence"), not anonymous Column_N.
    """
    X_temporal = patient.X_temporal_scaled[np.newaxis, :, :]   # (1, n_months, n_temporal)
    X_static = patient.X_static_scaled[np.newaxis, :]          # (1, n_static)

    X_flat, feature_names = mu.build_features_at_month(
        X_temporal, X_static,
        up_to_month=up_to_month,
        temporal_names=patient.temporal_names,
        static_names=patient.static_names,
    )
    return X_flat, feature_names
