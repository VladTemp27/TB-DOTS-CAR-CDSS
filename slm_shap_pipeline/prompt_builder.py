from __future__ import annotations

from types import SimpleNamespace

from backend.prompt import build_prompt
from slm_shap_pipeline.data_loader import PatientRow
from slm_shap_pipeline.feature_aggregator import top_k


def build_prompt_for_patient(
    patient: PatientRow,
    failure_probability: float,
    aggregated_shap: dict[str, dict],
    condition: str,
    month_of_prediction: int | None = None,
) -> str:
    """Build the Gemma prompt for one patient under blind or sighted condition.

    condition "sighted": top-5 SHAP contributions included in the prompt.
    condition "blind":   contributions block omitted; SLM must infer factors.
    """
    if condition not in ("sighted", "blind"):
        raise ValueError(f"condition must be 'sighted' or 'blind', got {condition!r}")

    if condition == "sighted":
        contributions = [
            SimpleNamespace(feature=_display_name(base), delta=data["signed_sum"])
            for base, data in top_k(aggregated_shap, k=5)
        ]
    else:
        contributions = []

    req = SimpleNamespace(
        patient_name=patient.patient_id,
        age=int(patient.age),
        sex=patient.sex,
        anatomical_site=patient.anatomical_site,        # None in V2
        source_of_patient=patient.source_of_patient,    # None in V2
        type=patient.patient_type,                      # None in V2
        registration_group=patient.registration_group,
        bacteriologic_status=patient.bacteriologic_status,
        microscopy_result=patient.microscopy_result,
        days_to_treatment=int(patient.days_to_treatment),
        failure_probability=failure_probability,
        contributions=contributions,
        month_of_prediction=month_of_prediction,
    )

    return build_prompt(req)


def _display_name(canonical: str) -> str:
    """Convert canonical snake_case group name to display form."""
    return canonical.replace("_", " ")
