import numpy as np
import pytest
from slm_shap_pipeline.feature_builder import build_features_for_patient
from slm_shap_pipeline.data_loader import PatientRow

N_STATIC = 5
N_TEMPORAL = 3
N_MONTHS = 13


def _mock_patient() -> PatientRow:
    return PatientRow(
        patient_id="P000",
        age=35.0,
        sex="M",
        days_to_treatment=3.0,
        registration_group="New",
        bacteriologic_status="BC",
        microscopy_result="1",
        X_static_scaled=np.zeros(N_STATIC),
        X_temporal_scaled=np.zeros((N_MONTHS, N_TEMPORAL)),
        static_names=[f"s{i}" for i in range(N_STATIC)],
        temporal_names=[f"t{i}" for i in range(N_TEMPORAL)],
    )


def test_build_features_returns_correct_shape():
    """X_flat shape must be (1, n_features); names list length must match."""
    patient = _mock_patient()
    X_flat, names = build_features_for_patient(patient, up_to_month=12)
    assert X_flat.shape[0] == 1
    assert X_flat.shape[1] == len(names)


def test_feature_names_are_consistent():
    """Calling twice with same inputs must return identical name list."""
    patient = _mock_patient()
    _, names_a = build_features_for_patient(patient, up_to_month=12)
    _, names_b = build_features_for_patient(patient, up_to_month=12)
    assert names_a == names_b


def test_feature_count_for_known_inputs():
    """With N_STATIC=5, N_TEMPORAL=3, up_to_month=12:
    static(5) + raw(13*3=39) + agg(4*3=12) + trend(3) + latest(3) + months_avail(1) = 63"""
    patient = _mock_patient()
    X_flat, names = build_features_for_patient(patient, up_to_month=12)
    # static: 5
    # raw temporal M0..M12: 13 months * 3 features = 39
    # aggregates mean/std/min/max: 4 * 3 = 12
    # trend: 3
    # latest: 3
    # months_available: 1
    expected = 5 + 39 + 12 + 3 + 3 + 1
    assert X_flat.shape[1] == expected, f"Expected {expected} features, got {X_flat.shape[1]}"
    assert len(names) == expected
