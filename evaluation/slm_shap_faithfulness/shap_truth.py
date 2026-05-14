import pandas as pd


def build_truth_rows(shap_series: pd.Series, top_k: int = 5) -> list[dict]:
    """Return top-k SHAP features ranked by absolute value.

    Each entry contains: feature, sign, abs_shap, rank (1-based), magnitude_band.
    magnitude_band is threshold-based: features with abs_shap >= 30% of the top
    feature's abs_shap are "strong"; the rest are "moderate". This matches how
    SLMs naturally use magnitude language (large percentages → "strong").
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    ranked = sorted(shap_series.items(), key=lambda x: abs(float(x[1])), reverse=True)[:top_k]
    max_abs = abs(float(ranked[0][1])) if ranked else 1.0
    return [
        {
            "feature": f,
            "sign": "increase" if v >= 0 else "decrease",
            "abs_shap": abs(float(v)),
            "rank": i + 1,
            "magnitude_band": "strong" if abs(float(v)) >= 0.30 * max_abs else "moderate",
        }
        for i, (f, v) in enumerate(ranked)
    ]
