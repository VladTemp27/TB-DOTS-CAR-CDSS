"""
strategy.py
===========
Routes each missing column to one of four literature-backed pathways
based on diagnostic results and feature importance scores.

Pathways (from the research document):
    ALPHA_DROP      — Drop the column entirely (near-null or unimportant)
    BETA_LISTWISE   — Listwise (complete-case) deletion  [low missingness + MCAR + large N]
    GAMMA_INDICATOR — Missing indicator + fill           [MNAR or moderate missingness]
    DELTA_MICE      — Stochastic MICE w/ missForest      [MAR + high-importance feature]
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Pathway(str, Enum):
    ALPHA_DROP      = "ALPHA_DROP"
    BETA_LISTWISE   = "BETA_LISTWISE"
    GAMMA_INDICATOR = "GAMMA_INDICATOR"
    DELTA_MICE      = "DELTA_MICE"


# Default thresholds — can be overridden via the config dict passed to route_columns()
DEFAULT_CONFIG: dict = {
    # Alpha: always drop above this rate regardless of importance
    "alpha_hard_threshold":        0.80,
    # Alpha: drop if above this rate AND feature is in bottom importance quartile
    "alpha_soft_threshold":        0.50,
    # Beta: listwise only viable when column missingness is below this fraction
    "beta_max_missing":            0.15,
    # Beta: minimum remaining rows after listwise deletion
    "beta_min_remaining_n":        500,
    # Delta: use MICE only if feature ranks above this importance percentile
    "delta_min_importance_pct":    50,
    # N of the dataset — must be supplied at runtime via config or __init__.py
    "n_total":                     None,
}


def route_columns(
    diagnostics_result: dict,
    importance_scores: dict[str, float],
    config: dict | None = None,
    temporal_cols: set[str] | None = None,
) -> dict[str, Pathway]:
    """
    Assign each column with missing data to a handling pathway.

    Parameters
    ----------
    diagnostics_result : dict
        Output of the diagnostics phase containing:
        - "missing_rates"  : {col: float}
        - "mechanisms"     : {col: "MCAR" | "MAR" | "MNAR"}
    importance_scores : dict[str, float]
        RF feature importances keyed by column name.
    config : dict, optional
        Override any DEFAULT_CONFIG key.

    Returns
    -------
    dict {col: Pathway}  — only columns with missing data are included.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    missing_rates  = diagnostics_result.get("missing_rates", {})
    mechanisms     = diagnostics_result.get("mechanisms", {})

    # Pre-compute importance quartile threshold.
    # If importance_scores is empty (RF couldn't fit on available complete cases),
    # disable importance-gated Alpha entirely — we must not drop clinical features
    # solely because we lacked sufficient complete-case data to rank them.
    has_importance = bool(importance_scores)
    if has_importance:
        imp_values = np.array(list(importance_scores.values()))
        q25 = float(np.percentile(imp_values, 25))
        q50 = float(np.percentile(imp_values, 50))
    else:
        q25 = q50 = 0.0

    _temporal_cols: set[str] = temporal_cols or set()
    routing: dict[str, Pathway] = {}

    for col, miss_rate in missing_rates.items():
        if miss_rate <= 0:
            continue  # nothing to do

        importance  = importance_scores.get(col, 0.0)
        mechanism   = mechanisms.get(col, "MNAR")   # conservative default
        is_low_imp  = importance <= q25
        is_mid_imp  = importance <= q50

        # ------------------------------------------------------------------
        # ALPHA: Drop column
        # ------------------------------------------------------------------
        # Temporal features are NEVER soft-Alpha-dropped: their missingness
        # is structural (late months have fewer readings) and the absence IS
        # the signal. The research doc specifies GAMMA as the default for all
        # temporal features. Hard threshold (>80%) still applies even to them.
        is_temporal = col in _temporal_cols

        if miss_rate > cfg["alpha_hard_threshold"] and not is_temporal:
            routing[col] = Pathway.ALPHA_DROP
            continue

        # Soft threshold: requires confirmed importance scores AND applies
        # only to static features — never to temporal columns.
        if has_importance and not is_temporal and miss_rate > cfg["alpha_soft_threshold"] and is_low_imp:
            routing[col] = Pathway.ALPHA_DROP
            continue

        # ------------------------------------------------------------------
        # BETA: Listwise deletion
        # Only safe when: MCAR proven + low missingness + large remaining N
        # With N=600, deleting even 15% leaves 510 rows — barely viable.
        # ------------------------------------------------------------------
        n_total = cfg.get("n_total")
        if n_total is None:
            raise ValueError(
                "strategy.route_columns: 'n_total' must be set in config. "
                "Pass config={'n_total': len(df_static)} from handle_missing_data()."
            )
        remaining_n = n_total * (1.0 - miss_rate)
        if (
            mechanism == "MCAR"
            and miss_rate < cfg["beta_max_missing"]
            and remaining_n > cfg["beta_min_remaining_n"]
        ):
            routing[col] = Pathway.BETA_LISTWISE
            continue

        # ------------------------------------------------------------------
        # DELTA: Stochastic MICE (missForest-style)
        # Use when: MAR mechanism + feature is informative (or importance unknown)
        # When importance scores are unavailable, MAR mechanism alone is sufficient
        # justification — don't gate on importance we don't have.
        # ------------------------------------------------------------------
        if mechanism == "MAR" and (not has_importance or not is_mid_imp):
            routing[col] = Pathway.DELTA_MICE
            continue

        # ------------------------------------------------------------------
        # GAMMA: Missing indicator + fill
        # Default for MNAR, temporal features, and moderate missingness
        # ------------------------------------------------------------------
        routing[col] = Pathway.GAMMA_INDICATOR

    return routing
