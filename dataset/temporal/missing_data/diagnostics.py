"""
diagnostics.py
==============
Missingness audit, Little's MCAR test, temporal dropout detection,
and MAR/MNAR mechanism classification.

All functions operate on DataFrames already produced by
structure_temporal_data() — i.e. df_static and df_temporal in long format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Missing rate audit
# ---------------------------------------------------------------------------

def compute_missing_rates(df: pd.DataFrame) -> dict[str, float]:
    """Return {col: fraction_missing} for every column with at least one NaN."""
    rates = df.isnull().mean()
    return rates[rates > 0].to_dict()


# ---------------------------------------------------------------------------
# Little's MCAR test (simplified chi-square formulation)
# ---------------------------------------------------------------------------

def littles_mcar_test(df: pd.DataFrame) -> dict:
    """
    Simplified Little's MCAR test on numeric columns.

    Groups rows by missingness pattern, compares each group's observed-column
    means to the grand column means using a diagonal-covariance chi-square
    approximation.

    H0: data is Missing Completely at Random (MCAR)
    Reject H0 (p < 0.05) → data is at least MAR.

    Returns
    -------
    dict with keys:
        chi2      : float  — test statistic
        p_value   : float
        df        : int    — degrees of freedom
        is_mcar   : bool   — True if p > 0.05
        n_patterns: int    — number of distinct missingness patterns found
        note      : str    — human-readable summary
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        return {
            "chi2": None,
            "p_value": None,
            "df": None,
            "is_mcar": None,
            "n_patterns": None,
            "note": "Insufficient numeric columns (need ≥ 2).",
        }

    mu_hat = numeric_df.mean()          # grand column means (EM ≈ observed mean)
    sigma   = numeric_df.var()           # column variances
    sigma   = sigma.where(sigma > 1e-10, 1e-10)  # guard against zero-variance cols

    miss_indicator = numeric_df.isnull()
    pattern_series = miss_indicator.apply(lambda row: tuple(row), axis=1)

    d2_total  = 0.0
    df_total  = 0
    n_patterns = 0

    for pattern, group_idx in pattern_series.groupby(pattern_series).groups.items():
        observed_mask = [not m for m in pattern]
        observed_cols = list(numeric_df.columns[observed_mask])

        if not observed_cols:
            continue

        group = numeric_df.loc[group_idx, observed_cols]
        n_k   = len(group)

        if n_k < 2:
            continue

        n_patterns += 1
        group_mean = group.mean()
        diff       = group_mean - mu_hat[observed_cols]

        # Diagonal chi-square contribution (scalar form)
        d2 = float(n_k * (diff ** 2 / sigma[observed_cols]).sum())
        d2_total  += d2
        df_total  += len(observed_cols)

    # Adjust df: subtract only the number of columns that actually have missing values.
    # Using all numeric columns (p) over-deflates and can silently clamp df to 1.
    # Little (1988): df = sum_k(p_k) - q  where q = number of columns with missingness.
    q      = int(numeric_df.isnull().any().sum())   # columns with ≥1 missing value
    df_adj = df_total - q

    if df_adj <= 0:
        return {
            "chi2":       round(d2_total, 4),
            "p_value":    None,
            "df":         df_adj,
            "is_mcar":    None,
            "n_patterns": n_patterns,
            "note": (
                f"Test unreliable: adjusted df={df_adj} ≤ 0 "
                f"(df_total={df_total}, q={q}). "
                "Insufficient pattern diversity for this approximation. "
                "Treating data as non-MCAR (conservative)."
            ),
        }

    p_value = float(1 - stats.chi2.cdf(d2_total, df_adj))
    is_mcar = p_value > 0.05

    return {
        "chi2":       round(d2_total, 4),
        "p_value":    round(p_value, 4),
        "df":         df_adj,
        "is_mcar":    is_mcar,
        "n_patterns": n_patterns,
        "note": (
            f"p={p_value:.4f} — cannot reject MCAR (H0); listwise deletion is unbiased."
            if is_mcar else
            f"p={p_value:.4f} — reject MCAR; data is at least MAR. Avoid listwise deletion."
        ),
    }


# ---------------------------------------------------------------------------
# Temporal dropout detection
# ---------------------------------------------------------------------------

def detect_temporal_pattern(df_temporal: pd.DataFrame) -> dict[str, str]:
    """
    Check whether missingness increases monotonically with treatment month.

    A strong positive Spearman correlation between month and monthly missing
    rate is a clinical signal of MNAR dropout (patients stop attending because
    of adverse outcomes, not at random).

    Returns
    -------
    dict {col: "MNAR_dropout" | "random" | "N/A"}
    """
    skip_cols = {"patient_id", "month"}
    feature_cols = [c for c in df_temporal.columns if c not in skip_cols]
    results: dict[str, str] = {}

    for col in feature_cols:
        monthly_miss = (
            df_temporal.groupby("month")[col]
            .apply(lambda x: x.isnull().mean())
            .reset_index(name="miss_rate")
        )

        if len(monthly_miss) < 3:
            results[col] = "N/A"
            continue

        corr, p = stats.spearmanr(monthly_miss["month"], monthly_miss["miss_rate"])
        results[col] = "MNAR_dropout" if (corr > 0.6 and p < 0.05) else "random"

    return results


# ---------------------------------------------------------------------------
# Missingness-to-observed correlation (MAR proxy)
# ---------------------------------------------------------------------------

def compute_missingness_correlations(df: pd.DataFrame) -> dict[str, dict]:
    """
    For each column with missing values, compute the point-biserial correlation
    between its missingness indicator (0/1) and all other observed numeric columns.

    A high max correlation suggests MAR: the missingness is predictable from
    observed variables (the imputation model can account for it).

    Returns
    -------
    dict {col: {"max_abs_corr": float, "top_predictors": list[str]}}
    """
    numeric_df = df.select_dtypes(include=[np.number])
    results: dict[str, dict] = {}

    for col in numeric_df.columns:
        if not df[col].isnull().any():
            continue

        is_missing = df[col].isnull().astype(float)

        # Only use predictors that are mostly observed (>50% present)
        other_cols = [
            c for c in numeric_df.columns
            if c != col and df[c].notna().mean() > 0.5
        ]
        if not other_cols:
            results[col] = {"max_abs_corr": 0.0, "top_predictors": []}
            continue

        predictors = numeric_df[other_cols].fillna(numeric_df[other_cols].median())

        # Guard: if is_missing is constant (all 0 or all 1) correlation is undefined
        if is_missing.std() < 1e-10:
            results[col] = {"max_abs_corr": 0.0, "top_predictors": []}
            continue

        corrs = predictors.corrwith(is_missing).abs()
        corrs = corrs.dropna()

        results[col] = {
            "max_abs_corr": round(float(corrs.max()), 4) if len(corrs) > 0 else 0.0,
            "top_predictors": list(corrs.nlargest(3).index),
        }

    return results


# ---------------------------------------------------------------------------
# Per-column mechanism classification
# ---------------------------------------------------------------------------

def classify_mechanism(
    col: str,
    missing_rate: float,
    is_mcar_global: bool | None,
    temporal_pattern: str | None,
    missingness_correlations: dict[str, dict],
) -> str:
    """
    Classify each column's missing data mechanism.

    Priority order (most to least specific):
    1. Temporal MNAR dropout signal (strongest clinical evidence)
    2. High correlation with observed variables → MAR
    3. Global MCAR test passed → MCAR
    4. Default conservative assumption → MNAR

    Returns
    -------
    "MCAR" | "MAR" | "MNAR"
    """
    # 1. Temporal dropout is a strong MNAR signal in TB treatment data
    if temporal_pattern == "MNAR_dropout":
        return "MNAR"

    # 2. Missingness predictable from observed columns → MAR
    col_corr = missingness_correlations.get(col, {})
    max_corr  = col_corr.get("max_abs_corr", 0.0)
    if max_corr > 0.3:
        return "MAR"

    # 3. Global MCAR test passed
    if is_mcar_global:
        return "MCAR"

    # 4. Conservative fallback
    return "MNAR"
