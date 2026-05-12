"""
report.py
=========
Generates a structured JSON audit report of all missing-data decisions
and saves it alongside the other pipeline outputs.

The report is the paper trail that satisfies the research document's
recommendation for pre-registration and methodological transparency:
every column's missing rate, diagnosed mechanism, and assigned pathway
is recorded before downstream models see the data.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from .strategy import Pathway


def generate_report(
    diagnostics_result: dict,
    routing: dict[str, Pathway],
    pre_stats: dict,
    post_stats: dict,
    output_dir: str,
) -> tuple[dict, str]:
    """
    Build and persist the missing-data audit report.

    Parameters
    ----------
    diagnostics_result : dict
        Output of the diagnostics phase (missing_rates, mcar_test,
        temporal_patterns, mechanisms).
    routing : dict[str, Pathway]
        Per-column pathway assignments from strategy.route_columns().
    pre_stats : dict
        Dataset shape before applying strategies (n_rows, n_static_cols, etc.).
    post_stats : dict
        Dataset shape after applying strategies.
    output_dir : str
        Directory where missing_data_report.json will be written.

    Returns
    -------
    (report_dict, report_path)
    """
    missing_rates     = diagnostics_result.get("missing_rates", {})
    mechanisms        = diagnostics_result.get("mechanisms", {})
    temporal_patterns = diagnostics_result.get("temporal_patterns", {})
    mcar_result       = diagnostics_result.get("mcar_test", {})

    # ---- Per-column decision table ----------------------------------------
    column_decisions: dict[str, dict] = {}
    for col, pathway in routing.items():
        col_temporal_pattern = temporal_patterns.get(col, "N/A")
        column_decisions[col] = {
            "missing_rate_pct": round(missing_rates.get(col, 0.0) * 100, 2),
            "mechanism":        mechanisms.get(col, "unknown"),
            "temporal_pattern": col_temporal_pattern,
            "pathway":          pathway.value if isinstance(pathway, Pathway) else str(pathway),
            "pathway_rationale": _rationale(
                pathway,
                missing_rates.get(col, 0.0),
                mechanisms.get(col, "MNAR"),
                col_temporal_pattern,
            ),
        }

    # ---- Pathway summary counts -------------------------------------------
    pathway_counts: dict[str, int] = {p.value: 0 for p in Pathway}
    for p in routing.values():
        key = p.value if isinstance(p, Pathway) else str(p)
        pathway_counts[key] = pathway_counts.get(key, 0) + 1

    # ---- Full report -------------------------------------------------------
    report = {
        "generated_at":  datetime.now().isoformat(),
        "pipeline":      "TB-DOTS CAR CDSS — Missing Data Handling",
        "dataset_info":  pre_stats,
        "mcar_test":     mcar_result,
        "pathway_summary": pathway_counts,
        "column_decisions": column_decisions,
        "post_imputation":  post_stats,
    }

    # ---- Persist -----------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "missing_data_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ---- Console summary ---------------------------------------------------
    _print_summary(report, column_decisions)

    return report, report_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rationale(
    pathway: Pathway,
    missing_rate: float,
    mechanism: str,
    temporal_pattern: str,
) -> str:
    pct = f"{missing_rate * 100:.1f}%"
    if pathway == Pathway.ALPHA_DROP:
        return (
            f"Dropped: {pct} missing"
            + (" (exceeds hard threshold)." if missing_rate > 0.80 else " and low feature importance.")
        )
    if pathway == Pathway.BETA_LISTWISE:
        return (
            f"Listwise deletion: {pct} missing, mechanism={mechanism}, "
            "sample size sufficient after deletion."
        )
    if pathway == Pathway.GAMMA_INDICATOR:
        reason = (
            f"temporal dropout pattern detected (MNAR)"
            if temporal_pattern == "MNAR_dropout"
            else f"mechanism={mechanism}, moderate-to-high missingness ({pct})"
        )
        return f"Missing indicator + fill: {reason}."
    if pathway == Pathway.DELTA_MICE:
        return (
            f"MICE (missForest): {pct} missing, mechanism=MAR — "
            "missingness is explainable by observed variables; "
            "stochastic imputation preserves variance."
        )
    return "Unknown pathway."


def _print_summary(report: dict, column_decisions: dict) -> None:
    summary = report.get("pathway_summary", {})
    mcar    = report.get("mcar_test", {})
    post    = report.get("post_imputation", {})

    print("\n  ┌─ Missing Data Audit ────────────────────────────────────────")
    print(f"  │  MCAR test: {mcar.get('note', 'N/A')}")
    print(f"  │  Columns routed:")
    for pathway, count in summary.items():
        if count > 0:
            print(f"  │    {pathway:<20} {count:>3} columns")
    print(f"  │  Post-imputation nulls:")
    print(f"  │    Static:   {post.get('remaining_static_nulls', '?')}")
    print(f"  │    Temporal: {post.get('remaining_temporal_nulls', '?')}")
    print(f"  │  Rows retained: {post.get('n_rows', '?')}")
    print("  └─────────────────────────────────────────────────────────────")
