"""
feature_importance.py
=====================
Random Forest feature importance ranking on complete cases.

Used by the strategy router to distinguish high-value sparse features
(worth imputing) from low-value ones (safe to drop via Pathway Alpha).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def rank_feature_importance(
    df_static: pd.DataFrame,
    df_temporal: pd.DataFrame,
    target_col: str = "outcome",
) -> dict[str, float]:
    """
    Fit a Random Forest on complete cases and return per-feature importance.

    Strategy
    --------
    1. Compute per-patient temporal means (collapses the long-format temporal
       DataFrame into one row per patient).
    2. Join with df_static to get a flat feature matrix.
    3. Use only rows with no missing values in numeric columns.
    4. Fit RandomForestClassifier on the encoded target.

    Returns
    -------
    dict {col_name: importance_score}  — empty dict if fitting is not possible.
    """
    # ---- Build flat feature matrix ----------------------------------------
    skip_temporal = {"patient_id", "month"}
    temporal_numeric = df_temporal.drop(columns=list(skip_temporal), errors="ignore")
    temporal_numeric = temporal_numeric.select_dtypes(include=[np.number])

    temporal_means = (
        df_temporal[["patient_id"]]
        .join(temporal_numeric)
        .groupby("patient_id")
        .mean()
    )

    # Prefix temporal means to avoid column name collisions with static features
    # (e.g. xpert_mtb_rif exists in both df_static and temporal monthly readings)
    temporal_means.columns = [f"{c}_temporal_mean" for c in temporal_means.columns]

    static_indexed = df_static.set_index("patient_id") if "patient_id" in df_static.columns else df_static
    combined = static_indexed.join(temporal_means, how="left")

    # ---- Resolve target column --------------------------------------------
    if target_col not in combined.columns:
        cat_cols = combined.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) == 0:
            return {}
        target_col = cat_cols[0]

    y_raw = combined[target_col]
    X_raw = combined.drop(columns=[target_col], errors="ignore")

    # ---- Keep only numeric features ---------------------------------------
    X_num = X_raw.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return {}

    # ---- Available-case analysis: require target present, median-fill features ----
    # Requiring zero missing values across ALL features is too strict for a clinical
    # dataset with high feature-level missingness — it produces zero usable rows.
    # Instead: keep rows where the target is observed, then median-fill feature NaNs.
    # This is a deliberately coarse imputation used only to rank features, not for
    # model training; the ranking signal is far more valuable than the purity loss.
    target_present_mask = y_raw.notna()
    X_avail = X_num[target_present_mask].copy()
    y_complete = y_raw[target_present_mask]

    # Median-fill remaining NaNs in features (available-case imputation)
    for col in X_avail.columns:
        if X_avail[col].isnull().any():
            X_avail[col] = X_avail[col].fillna(X_avail[col].median())

    X_complete = X_avail

    if len(X_complete) < 10:
        # Still not enough rows even with available-case analysis
        return {}

    # ---- Encode target (handles string labels) ----------------------------
    y_encoded, _ = pd.factorize(y_complete)

    # ---- Fit RF -----------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    try:
        rf.fit(X_complete, y_encoded)
    except Exception:
        return {}

    return dict(zip(X_complete.columns, rf.feature_importances_))
