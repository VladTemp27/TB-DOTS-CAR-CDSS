import json
import pytest
from pathlib import Path
from slm_shap_pipeline.case_writer import write_case, write_manifest


MOCK_AGGREGATED = {
    "Age": {"signed_sum": 0.18, "abs_sum": 0.18, "sign": "+", "constituents": ["age"]},
    "Treatment_Adherence": {"signed_sum": -0.34, "abs_sum": 0.41, "sign": "-",
                            "constituents": ["M0_pct_adherence"]},
}


def test_write_case_creates_valid_json(tmp_path):
    out_dir = tmp_path / "cases"
    write_case(
        patient_id="P001",
        condition="sighted",
        explanation="Age strongly increases risk.",
        aggregated_shap=MOCK_AGGREGATED,
        model_prediction=0.73,
        metadata={"model_hash": "abc123", "feature_groups_hash": "def456",
                  "prompt_git_sha": "abc", "gemini_call_timestamp": "2026-05-14T10:00:00Z"},
        out_dir=out_dir,
    )
    case_file = out_dir / "case_P001.json"
    assert case_file.exists()
    data = json.loads(case_file.read_text())
    assert data["patient_id"] == "P001"
    assert data["condition"] == "sighted"
    assert data["explanation"] == "Age strongly increases risk."
    assert "Age" in data["shap_values"]
    assert "Treatment_Adherence" in data["shap_values"]
    assert data["shap_values"]["Age"] == pytest.approx(0.18)


def test_write_case_shap_values_are_signed_sums(tmp_path):
    out_dir = tmp_path / "cases"
    write_case("P002", "blind", "explanation", MOCK_AGGREGATED, 0.5, {}, out_dir)
    data = json.loads((out_dir / "case_P002.json").read_text())
    assert data["shap_values"]["Treatment_Adherence"] == pytest.approx(-0.34)


def test_write_case_includes_metadata(tmp_path):
    out_dir = tmp_path / "cases"
    meta = {
        "model_hash": "abc",
        "scaler_static_hash": "s1" * 32,
        "scaler_temporal_hash": "s2" * 32,
        "feature_groups_hash": "def",
        "feature_policy_version": "temporal_v2_cleaned_output_facility_v1",
        "prompt_git_sha": "ghi",
        "gemini_call_timestamp": "2026-05-14T10:00:00Z",
    }
    write_case("P003", "sighted", "exp", MOCK_AGGREGATED, 0.6, meta, out_dir,
               month_of_prediction=12)
    data = json.loads((out_dir / "case_P003.json").read_text())
    assert data["prediction_metadata"]["model_hash"] == "abc"
    assert data["prediction_metadata"]["scaler_static_hash"] == "s1" * 32
    assert data["prediction_metadata"]["scaler_temporal_hash"] == "s2" * 32
    assert data["prediction_metadata"]["feature_policy_version"] == "temporal_v2_cleaned_output_facility_v1"
    assert data["month_of_prediction"] == 12


def test_write_manifest(tmp_path):
    write_manifest(
        out_dir=tmp_path,
        condition="sighted",
        config_snapshot={
            "model_path": "models/Temporal/v2/output/lightgbm/lgb_smoteenn_model.txt",
            "month_of_prediction": 12,
        },
        stats={"cache_hits": 5, "cache_misses": 39, "gemini_failures": 0, "duration_seconds": 120},
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["condition"] == "sighted"
    assert manifest["stats"]["cache_hits"] == 5
    assert manifest["config_snapshot"]["month_of_prediction"] == 12
