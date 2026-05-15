from __future__ import annotations

import json
from pathlib import Path


def build_groups(feature_groups_path: Path) -> dict[str, list[str]]:
    """Load feature_groups.json → {clinical_name: [raw_feature_names]}."""
    with open(feature_groups_path) as f:
        return json.load(f)


def validate_coverage(
    raw_shap: dict[str, float],
    groups: dict[str, list[str]],
) -> None:
    """Raise ValueError if any raw SHAP feature is not covered by any group."""
    all_grouped = {feat for members in groups.values() for feat in members}
    ungrouped = set(raw_shap.keys()) - all_grouped
    if ungrouped:
        raise ValueError(
            f"SHAP features not in feature_groups.json — ungrouped: {sorted(ungrouped)}\n"
            "Run scripts/discover_feature_groups.py and update feature_groups.json."
        )


def aggregate_shap(
    raw_shap: dict[str, float],
    groups: dict[str, list[str]],
    epsilon: float = 1e-6,
) -> dict[str, dict]:
    """Aggregate raw per-feature SHAP values into clinical base groups.

    Returns:
        {"Treatment_Adherence": {"signed_sum": 0.34, "abs_sum": 0.41,
                                  "sign": "+", "constituents": [...]}, ...}
    sign: "+" if signed_sum > epsilon, "-" if < -epsilon, else "mixed"
    """
    result: dict[str, dict] = {}
    for base_name, members in sorted(groups.items()):
        signed_sum = sum(raw_shap.get(m, 0.0) for m in members)
        abs_sum = sum(abs(raw_shap.get(m, 0.0)) for m in members)
        if signed_sum > epsilon:
            sign = "+"
        elif signed_sum < -epsilon:
            sign = "-"
        else:
            sign = "mixed"
        result[base_name] = {
            "signed_sum": signed_sum,
            "abs_sum": abs_sum,
            "sign": sign,
            "constituents": [m for m in members if m in raw_shap],
        }
    return result


def top_k(aggregated: dict[str, dict], k: int = 5) -> list[tuple[str, dict]]:
    """Return top-k groups by abs_sum, descending."""
    return sorted(aggregated.items(), key=lambda x: x[1]["abs_sum"], reverse=True)[:k]
