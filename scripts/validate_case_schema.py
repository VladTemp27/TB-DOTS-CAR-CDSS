#!/usr/bin/env python3
"""
Validate that case_*.json files match the evaluator's expected schema.
Checks: required fields, shap_values keys in known aliases, explanation parseable.

Usage:
    python scripts/validate_case_schema.py outputs/slm_shap_pipeline/latest/sighted/cases/
"""
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so evaluation.* imports work when script
# is invoked directly (./scripts/...) rather than as a module (python3 scripts/...)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.slm_shap_faithfulness.feature_map import canonicalize_feature
from evaluation.slm_shap_faithfulness.parser import parse_explanation


def validate_case(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]

    for field in ("patient_id", "explanation", "shap_values"):
        if field not in data:
            issues.append(f"missing required field: {field!r}")

    shap = data.get("shap_values", {})
    if not isinstance(shap, dict):
        issues.append("shap_values is not a dict")
    else:
        for key in shap:
            if canonicalize_feature(key) is None:
                issues.append(f"shap_values key not in alias map: {key!r}")

    explanation = data.get("explanation", "")
    if explanation and not explanation.startswith("[DRY RUN]"):
        parsed = parse_explanation(explanation)
        if not parsed.get("claims"):
            issues.append("parser found 0 feature claims in explanation")

    return issues


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_case_schema.py <cases_dir>")
        sys.exit(1)

    cases_dir = Path(sys.argv[1])
    case_files = sorted(cases_dir.glob("case_*.json"))
    if not case_files:
        print(f"No case files found in {cases_dir}")
        sys.exit(1)

    errors = 0
    for cf in case_files:
        issues = validate_case(cf)
        if issues:
            print(f"FAIL {cf.name}:")
            for issue in issues:
                print(f"  - {issue}")
            errors += 1

    print(f"\n{len(case_files)} cases checked. {errors} with issues.")
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
