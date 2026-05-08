"""
missing_data
============
Diagnostic-first missing data handling pipeline for the TB-DOTS CAR CDSS.

Public API
----------
    handle_missing_data(df_static, df_temporal, config=None, output_dir=...)
        Drop-in replacement for the original impute_missing_mice().
        Returns (df_static, df_temporal) with all NaNs resolved.

Pathway Architecture (from research document)
----------------------------------------------
    Alpha  — ALPHA_DROP      : Drop column (near-null or unimportant)
    Beta   — BETA_LISTWISE   : Listwise deletion (MCAR + low miss + large N)
    Gamma  — GAMMA_INDICATOR : Missing indicator + forward/median fill (MNAR)
    Delta  — DELTA_MICE      : Stochastic MICE w/ ExtraTrees/missForest (MAR)

Decision flow
-------------
    1. Diagnostics   — missing rates, Little's MCAR test, temporal patterns,
                       missingness-to-observed correlations
    2. Importance    — RF feature importance on complete cases
    3. Routing       — per-column pathway assignment
    4. Application   — Alpha → Beta → Gamma → Delta (in that order)
    5. Safety net    — catch any residual NaNs after all pathways
    6. Report        — JSON audit trail to output/missing_data_report.json
"""

from __future__ import annotations

import os
from collections import Counter

import pandas as pd

from .apply import apply_alpha, apply_beta, apply_delta, apply_gamma
from .diagnostics import (
    classify_mechanism,
    compute_missing_rates,
    compute_missingness_correlations,
    detect_temporal_pattern,
    littles_mcar_test,
)
from .feature_importance import rank_feature_importance
from .report import generate_report
from .strategy import DEFAULT_CONFIG, Pathway, route_columns

__all__ = ["handle_missing_data", "Pathway", "DEFAULT_CONFIG"]

_DEFAULT_OUTPUT_DIR = os.path.join("dataset", "temporal", "output")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def handle_missing_data(
    df_static: pd.DataFrame,
    df_temporal: pd.DataFrame,
    config: dict | None = None,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diagnostic-first missing data handler — drop-in for impute_missing_mice().

    Parameters
    ----------
    df_static : pd.DataFrame
        One row per patient; output of structure_temporal_data().
    df_temporal : pd.DataFrame
        Long format with columns [patient_id, month, ...features...].
    config : dict, optional
        Override any key in strategy.DEFAULT_CONFIG (thresholds, n_total, etc.)
    output_dir : str
        Where to write missing_data_report.json.

    Returns
    -------
    (df_static, df_temporal) with all NaN values resolved.
    """
    print("=" * 70)
    print("STAGE 7: Missing Data Handling (Diagnostic-First Pipeline)")
    print("=" * 70)

    pre_stats = {
        "n_rows":           len(df_static),
        "n_static_cols":    len(df_static.columns),
        "n_temporal_cols":  len(df_temporal.columns),
        "n_static_nulls":   int(df_static.isnull().sum().sum()),
        "n_temporal_nulls": int(df_temporal.isnull().sum().sum()),
    }

    # ------------------------------------------------------------------
    # Phase 1: Diagnostics
    # ------------------------------------------------------------------
    print("\n  [1/5] Running missingness diagnostics...")

    skip_temporal_id = {"patient_id", "month"}
    temporal_feature_df = df_temporal.drop(columns=list(skip_temporal_id), errors="ignore")

    missing_rates_static   = compute_missing_rates(df_static)
    missing_rates_temporal = compute_missing_rates(temporal_feature_df)
    all_missing_rates = {**missing_rates_static, **missing_rates_temporal}

    if not all_missing_rates:
        print("  No missing values detected — skipping imputation pipeline.")
        _emit_empty_report(pre_stats, output_dir)
        return df_static, df_temporal

    print(f"  Columns with missing data: {len(all_missing_rates)}")

    mcar_result       = littles_mcar_test(df_static)
    temporal_patterns = detect_temporal_pattern(df_temporal)
    miss_corrs        = compute_missingness_correlations(df_static)

    is_mcar_global = mcar_result.get("is_mcar")
    print(f"  Little's MCAR: {mcar_result.get('note', 'N/A')}")

    mechanisms: dict[str, str] = {}
    for col, miss_rate in all_missing_rates.items():
        temporal_pattern = temporal_patterns.get(col)
        mechanisms[col] = classify_mechanism(
            col, miss_rate, is_mcar_global, temporal_pattern, miss_corrs
        )

    diagnostics_result = {
        "missing_rates":     all_missing_rates,
        "mcar_test":         mcar_result,
        "temporal_patterns": temporal_patterns,
        "mechanisms":        mechanisms,
    }

    # ------------------------------------------------------------------
    # Phase 2: Feature importance
    # ------------------------------------------------------------------
    print("  [2/5] Ranking feature importance on complete cases...")
    importance_scores = rank_feature_importance(df_static, df_temporal)
    if importance_scores:
        print(f"  Importance scores computed for {len(importance_scores)} features.")
    else:
        print("  Warning: Could not compute importance scores — routing will use mechanism only.")

    # ------------------------------------------------------------------
    # Phase 3: Route columns to pathways
    # ------------------------------------------------------------------
    print("  [3/5] Routing columns to pathways...")
    cfg = {**DEFAULT_CONFIG, "n_total": len(df_static), **(config or {})}
    # Identify temporal feature columns so the router can protect them from Alpha
    temporal_feature_cols = set(
        c for c in df_temporal.columns if c not in skip_temporal_id
    )
    routing = route_columns(
        diagnostics_result, importance_scores, cfg, temporal_cols=temporal_feature_cols
    )

    pathway_counts = Counter(p.value for p in routing.values())
    for pathway, count in sorted(pathway_counts.items()):
        print(f"    {pathway:<20} → {count} column(s)")

    # ------------------------------------------------------------------
    # Phase 4: Apply strategies  (Alpha → Beta → Gamma → Delta)
    # ------------------------------------------------------------------
    print("  [4/5] Applying strategies...")

    alpha_cols  = [c for c, p in routing.items() if p == Pathway.ALPHA_DROP]
    beta_cols   = [c for c, p in routing.items() if p == Pathway.BETA_LISTWISE]
    gamma_cols  = [c for c, p in routing.items() if p == Pathway.GAMMA_INDICATOR]
    delta_cols  = [c for c, p in routing.items() if p == Pathway.DELTA_MICE]

    if alpha_cols:
        print(f"    Alpha: dropping {len(alpha_cols)} column(s): {alpha_cols}")
        df_static, df_temporal = apply_alpha(df_static, df_temporal, alpha_cols)

    if beta_cols:
        print(f"    Beta: listwise deletion for {len(beta_cols)} column(s): {beta_cols}")
        df_static, df_temporal = apply_beta(df_static, df_temporal, beta_cols)

    if gamma_cols:
        print(f"    Gamma: missing indicators for {len(gamma_cols)} column(s)")
        df_static, df_temporal = apply_gamma(df_static, df_temporal, gamma_cols)

    if delta_cols:
        print(f"    Delta: MICE (missForest) for {len(delta_cols)} column(s)")
        df_static, df_temporal = apply_delta(
            df_static, df_temporal, delta_cols, all_missing_rates
        )

    # ------------------------------------------------------------------
    # Safety net — catch any residual NaNs after all pathways
    # (e.g. columns with missing data that weren't in routing because
    #  they had 0% missing at audit time but gained NaNs after Beta)
    # ------------------------------------------------------------------
    static_residual   = int(df_static.isnull().sum().sum())
    temporal_residual = int(df_temporal.isnull().sum().sum())

    if static_residual > 0:
        _safety_fill(df_static)
    if temporal_residual > 0:
        _safety_fill(df_temporal)

    final_static_nulls   = int(df_static.isnull().sum().sum())
    final_temporal_nulls = int(df_temporal.isnull().sum().sum())

    if static_residual > 0 or temporal_residual > 0:
        print(
            f"    Safety net: resolved {static_residual + temporal_residual} "
            f"residual NaN(s); {final_static_nulls + final_temporal_nulls} remaining."
        )

    # ------------------------------------------------------------------
    # Phase 5: Audit report
    # ------------------------------------------------------------------
    print("  [5/5] Generating audit report...")

    post_stats = {
        "n_rows":                   len(df_static),
        "n_static_cols":            len(df_static.columns),
        "n_temporal_cols":          len(df_temporal.columns),
        "remaining_static_nulls":   final_static_nulls,
        "remaining_temporal_nulls": final_temporal_nulls,
    }

    _, report_path = generate_report(
        diagnostics_result, routing, pre_stats, post_stats, output_dir
    )
    print(f"    Audit report → {report_path}")
    print()

    return df_static, df_temporal


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safety_fill(df: pd.DataFrame) -> None:
    """
    Last-resort fill for any NaNs that slipped through all pathways.

    Each column touched is logged as a WARNING — unexpected residuals after
    a complete pipeline pass usually indicate a routing or logic bug.
    """
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing == 0:
            continue
        miss_pct = n_missing / len(df) * 100
        print(f"    WARNING safety-net: filling {n_missing} NaNs ({miss_pct:.1f}%) "
              f"in '{col}' — check routing logic if unexpected.")
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) > 0 else "Unknown")


def _emit_empty_report(pre_stats: dict, output_dir: str) -> None:
    """Write a minimal report when no missing data is found."""
    generate_report(
        diagnostics_result={"missing_rates": {}, "mcar_test": {}, "temporal_patterns": {}, "mechanisms": {}},
        routing={},
        pre_stats=pre_stats,
        post_stats={**pre_stats, "remaining_static_nulls": 0, "remaining_temporal_nulls": 0},
        output_dir=output_dir,
    )
