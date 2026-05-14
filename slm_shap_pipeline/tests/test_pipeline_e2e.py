"""Smoke test: 2 patients, dry-run Gemini, verify case files are written correctly."""
import json
import pytest
from pathlib import Path

from slm_shap_pipeline.config import PipelineConfig
from slm_shap_pipeline.pipeline import run


_REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = _REPO_ROOT / "dataset/temporal/output/cleaned_human_readable.csv"
MODEL_PATH = _REPO_ROOT / "models/Temporal/v2/output/lightgbm/lgb_smoteenn_model.txt"
GROUPS_PATH = _REPO_ROOT / "slm_shap_pipeline/feature_groups.json"

SKIP_IF_MISSING = pytest.mark.skipif(
    not (CSV_PATH.exists() and MODEL_PATH.exists() and GROUPS_PATH.exists()),
    reason="V2 dataset, model, or feature_groups.json not present",
)


@SKIP_IF_MISSING
def test_sighted_smoke(tmp_path):
    config = PipelineConfig(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        feature_groups_path=GROUPS_PATH,
        output_base=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )
    out_dir = run(
        condition="sighted",
        config=config,
        patient_limit=2,
        dry_run_gemini=True,
    )
    cases_dir = out_dir / "cases"
    case_files = list(cases_dir.glob("case_*.json"))
    assert len(case_files) == 2

    for cf in case_files:
        data = json.loads(cf.read_text())
        assert "patient_id" in data
        assert "explanation" in data
        assert "shap_values" in data
        assert isinstance(data["shap_values"], dict)
        assert len(data["shap_values"]) > 0
        assert data["condition"] == "sighted"
        assert data["month_of_prediction"] == 12

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["condition"] == "sighted"
    assert manifest["config_snapshot"]["patient_count"] == 2
    assert manifest["config_snapshot"]["month_of_prediction"] == 12


@SKIP_IF_MISSING
def test_blind_smoke(tmp_path):
    config = PipelineConfig(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        feature_groups_path=GROUPS_PATH,
        output_base=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )
    out_dir = run(
        condition="blind",
        config=config,
        patient_limit=2,
        dry_run_gemini=True,
    )
    case_files = list((out_dir / "cases").glob("case_*.json"))
    assert len(case_files) == 2
    for cf in case_files:
        data = json.loads(cf.read_text())
        assert "patient_id" in data
        assert "explanation" in data
        assert "shap_values" in data
        assert isinstance(data["shap_values"], dict)
        assert data["condition"] == "blind"
        assert data["month_of_prediction"] == 12


@SKIP_IF_MISSING
def test_case_files_match_eval_schema(tmp_path):
    """Eval adapter requires: patient_id, explanation, shap_values."""
    config = PipelineConfig(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        feature_groups_path=GROUPS_PATH,
        output_base=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )
    out_dir = run("sighted", config=config, patient_limit=1, dry_run_gemini=True)
    case_files = list((out_dir / "cases").glob("case_*.json"))
    assert len(case_files) == 1, f"Expected 1 case file, got {len(case_files)}"
    cf = case_files[0]
    data = json.loads(cf.read_text())

    assert "patient_id" in data
    assert isinstance(data["explanation"], str)
    assert isinstance(data["shap_values"], dict)
    assert all(isinstance(v, float) for v in data["shap_values"].values())
    assert "month_of_prediction" in data
    assert data["prediction_metadata"]["feature_policy_version"] == "temporal_v2_cleaned_output_facility_v1"


@SKIP_IF_MISSING
def test_manifest_records_provider_name(tmp_path):
    """The selected provider should be recorded in the manifest for traceability."""
    config = PipelineConfig(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        feature_groups_path=GROUPS_PATH,
        output_base=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
        provider="cli",
    )
    out_dir = run("sighted", config=config, patient_limit=1, dry_run_gemini=True)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config_snapshot"]["provider"] == "gemini-cli"


@SKIP_IF_MISSING
def test_provider_hotswap_to_api(tmp_path, monkeypatch):
    """API provider can be selected via config without changing pipeline code."""
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-for-construction")
    config = PipelineConfig(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        feature_groups_path=GROUPS_PATH,
        output_base=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
        provider="api",
    )
    out_dir = run("sighted", config=config, patient_limit=1, dry_run_gemini=True)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config_snapshot"]["provider"] == "google-api"
