"""
apply.py
========
Execution layer — applies the four pathway strategies to df_static and
df_temporal.

Pathway summaries
-----------------
Alpha  (ALPHA_DROP)      — Drop the column. Already handled by ALWAYS_DROP_COLUMNS
                           for near-null columns; Alpha here catches stragglers
                           identified by the diagnostic + importance gate.

Beta   (BETA_LISTWISE)   — Listwise deletion on rows where this column is missing.
                           Only called for MCAR + low missingness + large remaining N.

Gamma  (GAMMA_INDICATOR) — Engineer a binary `is_missing_{col}` indicator, then
                           fill with forward-fill (temporal) or median/mode (static).
                           Preserves the predictive signal of *absence* — critical
                           for TB dropout patterns where missingness IS the signal.

Delta  (DELTA_MICE)      — Stochastic MICE using ExtraTreesRegressor (missForest-
                           style). Non-parametric, captures non-linear relationships,
                           max_iter scaled to FMI (Von Hippel's rule). Forward-fill
                           applied to temporal features first to respect time ordering
                           and prevent future data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_max_iter(missing_rate: float) -> int:
    """
    Scale max_iter to the Fraction of Missing Information (FMI).

    Von Hippel's rule: n_imputations ≈ FMI ≈ missing_rate (%).
    Floor at 20 (original value) and cap at 100 to limit compute time.
    """
    return max(20, min(100, int(missing_rate * 100)))


def _make_extratrees_imputer(missing_rate: float, random_state: int = 42) -> IterativeImputer:
    """
    Build an IterativeImputer backed by ExtraTreesRegressor (missForest-style).

    ExtraTrees is preferred over BayesianRidge because:
    - Captures non-linear relationships between TB vitals
    - Handles mixed-scale numeric data without standardisation
    - Stochastic by design (random feature subsets) — no sample_posterior needed
    """
    return IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=10,        # keep fast; 10 trees is sufficient per MICE chain
            random_state=random_state,
        ),
        max_iter=_estimate_max_iter(missing_rate),
        random_state=random_state,
        sample_posterior=False,     # incompatible with tree estimators; stochasticity
        initial_strategy="median",  # comes from ExtraTrees' random splits instead
        skip_complete=True,
    )


# ---------------------------------------------------------------------------
# Alpha — drop
# ---------------------------------------------------------------------------

def apply_alpha(
    df_static: pd.DataFrame,
    df_temporal: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns from whichever DataFrame they belong to."""
    static_drop   = [c for c in cols if c in df_static.columns]
    temporal_drop = [c for c in cols if c in df_temporal.columns]

    if static_drop:
        df_static = df_static.drop(columns=static_drop)
    if temporal_drop:
        df_temporal = df_temporal.drop(columns=temporal_drop)

    return df_static, df_temporal


# ---------------------------------------------------------------------------
# Beta — listwise deletion
# ---------------------------------------------------------------------------

def apply_beta(
    df_static: pd.DataFrame,
    df_temporal: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows where any of the given static columns are missing.
    Cascades to df_temporal so patient_id sets stay consistent.
    """
    static_beta = [c for c in cols if c in df_static.columns]

    if static_beta:
        if "patient_id" not in df_static.columns:
            raise ValueError(
                "apply_beta: df_static must have 'patient_id' as a column, not as an index. "
                "Call df_static.reset_index() before passing to apply_beta()."
            )
        before = len(df_static)
        df_static  = df_static.dropna(subset=static_beta)
        after  = len(df_static)
        print(f"    Beta listwise: removed {before - after} rows "
              f"({(before - after) / before:.1%} of dataset)")

        # Keep temporal rows only for retained patients
        retained_ids = set(df_static["patient_id"])
        df_temporal  = df_temporal[df_temporal["patient_id"].isin(retained_ids)]

    return df_static, df_temporal


# ---------------------------------------------------------------------------
# Gamma — missing indicator + fill
# ---------------------------------------------------------------------------

def apply_gamma(
    df_static: pd.DataFrame,
    df_temporal: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each column with missing values:
      1. Add binary `is_missing_{col}` indicator (1 = was missing)
      2. Fill remaining NaNs:
         - Temporal numeric: forward-fill within patient (no future leakage),
           then median fill for any remaining gaps at the start of a sequence
         - Static numeric: median fill
         - Categorical (both): mode fill

    The indicator captures the *signal of absence* — in TB treatment data,
    a missing M9 weight reading often means the patient stopped attending,
    which is itself a strong predictor of treatment failure.
    """
    # ---- Static -----------------------------------------------------------
    static_cols = [c for c in cols if c in df_static.columns]
    for col in static_cols:
        if not df_static[col].isnull().any():
            continue

        df_static[f"is_missing_{col}"] = df_static[col].isnull().astype(np.int8)

        if pd.api.types.is_numeric_dtype(df_static[col]):
            fill = df_static[col].median()
            df_static[col] = df_static[col].fillna(fill)
        else:
            mode = df_static[col].mode()
            fill = mode.iloc[0] if len(mode) > 0 else "Unknown"
            df_static[col] = df_static[col].fillna(fill)

    # ---- Temporal ---------------------------------------------------------
    temporal_cols = [c for c in cols if c in df_temporal.columns]
    if temporal_cols:
        df_temporal = df_temporal.sort_values(["patient_id", "month"])

    for col in temporal_cols:
        if not df_temporal[col].isnull().any():
            continue

        df_temporal[f"is_missing_{col}"] = df_temporal[col].isnull().astype(np.int8)

        if pd.api.types.is_numeric_dtype(df_temporal[col]):
            # Forward-fill within patient (past → future only — no leakage)
            df_temporal[col] = df_temporal.groupby("patient_id")[col].ffill()
            # For sequence-start gaps (M0 missing): use global median, NOT bfill.
            # bfill would fill M0 with M2 data — a future measurement the model
            # cannot have at inference time, causing direct temporal leakage.
            df_temporal[col] = df_temporal[col].fillna(df_temporal[col].median())
        else:
            df_temporal[col] = df_temporal.groupby("patient_id")[col].ffill()
            # Same logic: no bfill; fall back to global mode for start-of-sequence gaps
            mode = df_temporal[col].mode()
            fill = mode.iloc[0] if len(mode) > 0 else "Unknown"
            df_temporal[col] = df_temporal[col].fillna(fill)

    return df_static, df_temporal


# ---------------------------------------------------------------------------
# Delta — stochastic MICE (missForest-style)
# ---------------------------------------------------------------------------

def apply_delta(
    df_static: pd.DataFrame,
    df_temporal: pd.DataFrame,
    cols: list[str],
    missing_rates: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proper non-parametric MICE imputation for MAR columns.

    Static path
    -----------
    All static numeric Delta columns are imputed together in one
    IterativeImputer pass so the chained equations can leverage
    cross-feature relationships.

    Temporal path
    -------------
    1. Forward-fill within patient first (preserves time ordering, no leakage)
    2. MICE on remaining gaps using `month` as an extra temporal context feature
    3. Round `month` back to int after imputation (MICE may slightly perturb it)

    max_iter per pathway is driven by the worst-case (highest) missing rate
    among the columns being imputed — consistent with the FMI scaling rule.
    """
    # ---- Static numeric Delta columns ------------------------------------
    static_num_cols = [
        c for c in cols
        if c in df_static.columns
        and pd.api.types.is_numeric_dtype(df_static[c])
        and df_static[c].isnull().any()
    ]

    if static_num_cols:
        worst_rate = max(missing_rates.get(c, 0.0) for c in static_num_cols)
        imputer = _make_extratrees_imputer(worst_rate)
        df_static[static_num_cols] = imputer.fit_transform(df_static[static_num_cols])

    # ---- Temporal numeric Delta columns ----------------------------------
    temporal_num_cols = [
        c for c in cols
        if c in df_temporal.columns
        and pd.api.types.is_numeric_dtype(df_temporal[c])
        and df_temporal[c].isnull().any()
    ]

    if temporal_num_cols:
        df_temporal = df_temporal.sort_values(["patient_id", "month"])

        # Step 1: forward-fill within patient (respects time ordering)
        for col in temporal_num_cols:
            df_temporal[col] = df_temporal.groupby("patient_id")[col].ffill()

        remaining_missing = df_temporal[temporal_num_cols].isnull().sum().sum()

        # Step 2: MICE for anything forward-fill couldn't reach
        if remaining_missing > 0:
            worst_rate = max(missing_rates.get(c, 0.0) for c in temporal_num_cols)
            imputer_t  = _make_extratrees_imputer(worst_rate)

            impute_cols = ["month"] + temporal_num_cols
            df_temporal[impute_cols] = imputer_t.fit_transform(df_temporal[impute_cols])

            # Restore month to integer (imputer may introduce fractional values)
            df_temporal["month"] = df_temporal["month"].round().astype(int)

    return df_static, df_temporal
