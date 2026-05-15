from slm_shap_pipeline.data_loader import PatientRow
from slm_shap_pipeline.prompt_builder import build_prompt_for_patient

import pytest
import numpy as np

MOCK_PATIENT = PatientRow(
    patient_id="P001",
    age=45.0,
    sex="M",
    days_to_treatment=5.0,
    registration_group="New",
    bacteriologic_status="BC",
    microscopy_result="1",
    # anatomical_site, source_of_patient, patient_type are None (V2 CSV missing)
    X_static_scaled=np.zeros(5),
    X_temporal_scaled=np.zeros((13, 8)),
    static_names=["age"],
    temporal_names=["pct_adherence"],
)

MOCK_AGGREGATED = {
    "Treatment_Adherence": {"signed_sum": 0.34, "abs_sum": 0.41, "sign": "+", "constituents": []},
    "Age": {"signed_sum": 0.18, "abs_sum": 0.18, "sign": "+", "constituents": []},
    "Sex": {"signed_sum": -0.05, "abs_sum": 0.05, "sign": "-", "constituents": []},
    "Bacteriologic_Status": {"signed_sum": 0.22, "abs_sum": 0.22, "sign": "+", "constituents": []},
    "Days_To_Treatment": {"signed_sum": 0.12, "abs_sum": 0.12, "sign": "+", "constituents": []},
}


def test_sighted_prompt_contains_contributions():
    prompt = build_prompt_for_patient(MOCK_PATIENT, 0.73, MOCK_AGGREGATED, condition="sighted")
    assert "Top contributing factors" in prompt
    assert "Treatment Adherence" in prompt or "Treatment_Adherence" in prompt


def test_blind_prompt_contains_inference_instruction():
    prompt = build_prompt_for_patient(MOCK_PATIENT, 0.73, MOCK_AGGREGATED, condition="blind")
    assert "not provided" in prompt.lower() or "list" in prompt.lower()


def test_blind_prompt_does_not_contain_shap_deltas():
    prompt = build_prompt_for_patient(MOCK_PATIENT, 0.73, MOCK_AGGREGATED, condition="blind")
    assert "+34" not in prompt and "0.34" not in prompt


def test_prompt_omits_missing_fields():
    """anatomical_site / source_of_patient / type are None — must not appear in prompt."""
    prompt = build_prompt_for_patient(MOCK_PATIENT, 0.73, MOCK_AGGREGATED, condition="sighted")
    assert "anatomical" not in prompt.lower()
    assert "source of patient" not in prompt.lower()


def test_prompt_includes_month_context():
    """month_of_prediction should appear in the prompt."""
    prompt = build_prompt_for_patient(
        MOCK_PATIENT, 0.73, MOCK_AGGREGATED, condition="sighted", month_of_prediction=12
    )
    assert "12" in prompt and "month" in prompt.lower()


def test_invalid_condition_raises():
    with pytest.raises(ValueError, match="condition"):
        build_prompt_for_patient(MOCK_PATIENT, 0.73, MOCK_AGGREGATED, condition="unknown")
