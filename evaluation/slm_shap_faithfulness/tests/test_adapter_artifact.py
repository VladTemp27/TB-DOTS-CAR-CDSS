from pathlib import Path
from evaluation.slm_shap_faithfulness.adapters.artifact_adapter import load_cases_from_artifacts


def test_artifact_adapter_outputs_normalized_cases(tmp_path):
    out = load_cases_from_artifacts(tmp_path)
    assert isinstance(out, list)


def test_artifact_adapter_loads_json_artifact(tmp_path):
    import json
    artifact = {
        "patient_id": "P001",
        "explanation": "Age increases risk.",
        "shap_values": {"Age": 0.8, "BMI": -0.3},
    }
    (tmp_path / "case_P001.json").write_text(json.dumps(artifact))
    out = load_cases_from_artifacts(tmp_path)
    assert len(out) == 1
    assert out[0]["patient_id"] == "P001"
    assert "explanation" in out[0]
    assert "shap_values" in out[0]
