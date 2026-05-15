from __future__ import annotations

import numpy as np
import shap
import lightgbm as lgb


def compute_shap_for_row(
    booster: lgb.Booster,
    X_flat: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute SHAP values for a single patient row.

    X_flat shape: (1, 399) — output of build_features_for_patient.
    feature_names: semantic names matching the 399 feature columns.
    Returns {semantic_feature_name: shap_value}.

    NOTE: booster.feature_name() returns Column_0..Column_398 (anonymous V2 names).
    We use the reconstructed feature_names instead.
    """
    explainer = shap.TreeExplainer(booster)
    raw = explainer.shap_values(X_flat)

    # TreeExplainer returns list[array] for binary classifiers; take positive-class (index 1)
    if isinstance(raw, list):
        values = raw[1][0]
    else:
        values = raw[0]

    return {name: float(val) for name, val in zip(feature_names, values)}
