from __future__ import annotations

import hashlib
import io
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import models.Temporal.model_utils as mu
from slm_shap_pipeline.config import PipelineConfig


@dataclass
class PatientRow:
    patient_id: str
    age: float
    sex: str                    # "M" or "F" (normalized from Male/Female)
    days_to_treatment: float
    registration_group: str
    bacteriologic_status: str
    microscopy_result: str
    anatomical_site: None = None       # Not in V2 CSV; always None
    source_of_patient: None = None     # Not in V2 CSV; always None
    patient_type: None = None          # Not in V2 CSV; always None
    X_static_scaled: Optional[np.ndarray] = None
    X_temporal_scaled: Optional[np.ndarray] = None
    static_names: Optional[list] = None
    temporal_names: Optional[list] = None


@dataclass
class TestCohort:
    patients: list
    scaler_static: object
    scaler_temporal: object
    scaler_static_hash: str
    scaler_temporal_hash: str
    static_names: list
    temporal_names: list


def _scaler_hash(scaler) -> str:
    """Deterministic sha256 of a fitted sklearn scaler."""
    buf = io.BytesIO()
    pickle.dump(scaler, buf, protocol=4)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def load_test_patients(config: PipelineConfig) -> TestCohort:
    """Load the V2 test cohort (44 patients) via model_utils.

    Uses the same random_state=42 split as training so there is no
    leakage from train/val rows. Returns a TestCohort with patients
    sorted by patient_id plus fitted scalers and their hashes.
    """
    import pandas as pd

    # Single entry point: load, split, encode/impute (but NOT scale).
    # Must pass the same drop_feature_cols used during training so feature count
    # matches the saved model (399 features for temporal_v2_cleaned_output_facility_v1).
    data = mu.prepare_default_model_inputs(
        str(config.csv_path),
        val_frac=config.val_frac,
        test_frac=config.test_frac,
        random_state=config.random_seed,
        drop_feature_cols=mu.get_temporal_v2_drop_feature_cols(),
    )

    # Key names confirmed by reading _build_model_arrays in model_utils.py
    X_static = data["X_static"]                          # (n_patients, n_static_feats)
    X_temporal = data["X_temporal"]                      # (n_patients, n_months, n_temporal_feats)
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    df_static = data["df_static"]

    static_names = data["static_feature_names"]          # list[str]
    temporal_names = data["temporal_feature_names"]      # list[str]

    # Scale: fit scalers on train, transform all patients
    scaled = mu.scale_full_arrays(X_static, X_temporal, train_idx)
    X_static_scaled = scaled["X_static_scaled"]          # (n_patients, n_static)
    X_temporal_scaled = scaled["X_temporal_scaled"]      # (n_patients, n_months, n_temporal)
    scaler_static = scaled["scaler_static"]
    scaler_temporal = scaled["scaler_temporal"]

    rows: list[PatientRow] = []
    for i in test_idx:
        row = df_static.iloc[i]
        patient_id = str(row.get("patient_id", i))

        sex_raw = str(row.get("sex", "")).strip()
        sex = "M" if sex_raw.lower().startswith("m") else "F"

        try:
            dod = pd.to_datetime(row.get("date_of_diagnosis"))
            tsd = pd.to_datetime(row.get("treatment_start_date"))
            days = max(0.0, float((tsd - dod).days))
        except Exception:
            days = 0.0

        rows.append(PatientRow(
            patient_id=patient_id,
            age=float(row.get("age", 0) or 0),
            sex=sex,
            days_to_treatment=days,
            registration_group=str(row.get("case_registration_group", "") or ""),
            bacteriologic_status=str(row.get("bacteriologic_status", "") or ""),
            microscopy_result=str(row.get("smear_microscopy", "") or ""),
            X_static_scaled=X_static_scaled[i],
            X_temporal_scaled=X_temporal_scaled[i],
            static_names=list(static_names),
            temporal_names=list(temporal_names),
        ))

    rows.sort(key=lambda r: r.patient_id)

    return TestCohort(
        patients=rows,
        scaler_static=scaler_static,
        scaler_temporal=scaler_temporal,
        scaler_static_hash=_scaler_hash(scaler_static),
        scaler_temporal_hash=_scaler_hash(scaler_temporal),
        static_names=list(static_names),
        temporal_names=list(temporal_names),
    )
