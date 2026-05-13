import pandas as pd


def build_truth_rows(shap_series: pd.Series, top_k: int = 5) -> list[dict]:
    """Return top-k SHAP features ranked by absolute value.

    Each entry contains: feature, sign, abs_shap, rank (1-based), magnitude_band.
    magnitude_band is a rank-based scaffold: rank-1 → "strong", all others → "moderate".
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    ranked = sorted(shap_series.items(), key=lambda x: abs(float(x[1])), reverse=True)[:top_k]
    return [
        {
            "feature": f,
            "sign": "increase" if v >= 0 else "decrease",
            "abs_shap": abs(float(v)),
            "rank": i + 1,
            # magnitude_band: rank-based scaffold; rank-1 is "strong", others "moderate"
            "magnitude_band": "strong" if i == 0 else "moderate",
        }
        for i, (f, v) in enumerate(ranked)
    ]
