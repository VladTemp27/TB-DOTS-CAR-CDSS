import pandas as pd


def build_truth_rows(shap_series: pd.Series, top_k: int = 5) -> list[dict]:
    ranked = sorted(shap_series.items(), key=lambda x: abs(float(x[1])), reverse=True)[:top_k]
    return [
        {
            "feature": f,
            "sign": "increase" if v >= 0 else "decrease",
            "abs_shap": abs(float(v)),
            "rank": i + 1,
            "magnitude_band": "strong" if i == 0 else "moderate",
        }
        for i, (f, v) in enumerate(ranked)
    ]
