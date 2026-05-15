from pathlib import Path
from evaluation.slm_shap_faithfulness.adapters.runtime_adapter import load_cases_from_runtime


def test_runtime_adapter_empty_dir_returns_list(tmp_path):
    out = load_cases_from_runtime(tmp_path)
    assert isinstance(out, list)


def test_runtime_adapter_loads_runtime_json(tmp_path):
    import json
    case = {
        "patient_id": "P002",
        "explanation": "Province decreases risk.",
        "shap_values": {"Province": -0.5},
        "timestamp": "2026-05-13T00:00:00",
    }
    (tmp_path / "runtime_P002.json").write_text(json.dumps(case))
    out = load_cases_from_runtime(tmp_path)
    assert len(out) == 1
    assert out[0]["patient_id"] == "P002"
    assert "explanation" in out[0]


def test_runtime_adapter_raises_when_shap_missing_and_recompute_false(tmp_path):
    import json
    import pytest
    case = {
        "patient_id": "P003",
        "explanation": "Age increases risk.",
        # shap_values intentionally absent
    }
    (tmp_path / "runtime_P003.json").write_text(json.dumps(case))
    with pytest.raises(ValueError):
        load_cases_from_runtime(tmp_path, allow_recompute_shap=False)
