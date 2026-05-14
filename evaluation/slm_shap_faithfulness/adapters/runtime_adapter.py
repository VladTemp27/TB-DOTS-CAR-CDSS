"""Runtime adapter: loads SLM explanation cases generated at inference time.

Runtime files are JSON with keys: patient_id, explanation, shap_values, and
optionally a timestamp for temporal context validation. The adapter scans for
files matching `runtime_*.json`.
"""
import json
from pathlib import Path


def load_cases_from_runtime(
    input_dir: str | Path,
    allow_recompute_shap: bool = True,
) -> list[dict]:
    """Load and normalize cases from runtime-generated JSON files.

    Scans input_dir for files matching `runtime_*.json`. Each file must contain
    patient_id, explanation, shap_values, and optionally timestamp.

    Args:
        input_dir: Directory containing runtime JSON files.
        allow_recompute_shap: If True, missing shap_values are acceptable
            (a downstream recompute path may fill them). If False, a missing
            shap_values key raises ValueError.

    Returns:
        Normalized list of case dicts with keys: patient_id, explanation,
        shap_values, and timestamp (if present in source JSON).

    Note:
        The "timestamp" key is optional and may be absent from the returned
        case dicts. It is only included when present in the source JSON.
    """
    input_dir = Path(input_dir)
    cases = []
    for path in sorted(input_dir.glob("runtime_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Failed to load runtime case {path}: {exc}") from exc

        shap_values = data.get("shap_values")
        if shap_values is None and not allow_recompute_shap:
            raise ValueError(
                f"Runtime case {path} is missing shap_values and allow_recompute_shap=False."
            )

        case = {
            "patient_id": data.get("patient_id", path.stem),
            "explanation": data.get("explanation", ""),
            "shap_values": shap_values if shap_values is not None else {},
        }
        if "timestamp" in data:
            case["timestamp"] = data["timestamp"]
        cases.append(case)
    return cases
