from pathlib import Path
from evaluation.slm_shap_faithfulness.run_benchmark import run_benchmark


def test_runner_writes_mode_invariant_artifacts(tmp_path):
    out = run_benchmark(input_dir=tmp_path, output_dir=tmp_path, mode="artifact")
    assert "patients" in out and "summary" in out


def test_runner_returns_pass_rate_in_summary(tmp_path):
    import json
    artifact = {
        "patient_id": "P001",
        "explanation": "Age strongly increases risk.",
        "shap_values": {"Age": 0.8, "BMI": -0.3},
    }
    (tmp_path / "case_P001.json").write_text(json.dumps(artifact))
    out = run_benchmark(input_dir=tmp_path, output_dir=tmp_path, mode="artifact")
    assert "pass_rate" in out["summary"]
    assert isinstance(out["summary"]["pass_rate"], float)
