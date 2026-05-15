"""Artifact adapter: loads pre-computed SLM explanation artifacts from a directory.

Artifact files are JSON with keys: patient_id, explanation, shap_values.
The adapter scans the input_dir for files matching `case_*.json` and normalizes them
into a shared case schema list[dict].
"""
import json
from pathlib import Path


def load_cases_from_artifacts(input_dir: str | Path) -> list[dict]:
    """Load and normalize cases from pre-computed JSON artifact files.

    Scans input_dir for files matching `case_*.json`. Each file must contain
    a JSON object with at minimum: patient_id, explanation, shap_values.

    Returns:
        Normalized list of case dicts with keys: patient_id, explanation, shap_values.
        Returns empty list if no matching files are found.

    Note:
        The "timestamp" key is never included in artifact cases.
        Use runtime_adapter if timestamp tracking is needed.
    """
    input_dir = Path(input_dir)
    cases = []
    for path in sorted(input_dir.glob("case_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Failed to load artifact {path}: {exc}") from exc
        cases.append({
            "patient_id": data.get("patient_id", path.stem),
            "explanation": data.get("explanation", ""),
            "shap_values": data.get("shap_values", {}),
        })
    return cases
