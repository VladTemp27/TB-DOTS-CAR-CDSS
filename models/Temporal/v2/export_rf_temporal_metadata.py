"""Export deployable metadata for the temporal Random Forest model.

Writes:
- models/Temporal/v2/output/random_forest/rf_temporal_feature_metadata.json
- models/Temporal/v2/output/random_forest/rf_temporal_golden_samples.json

These artifacts are used to reproduce the exact 399-length input vector
expected by the ONNX model during web inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def build_features_at_month(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    up_to_month: int,
    temporal_names: list[str],
    static_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Mirror the notebook's build_features_at_month for feature ordering."""

    n_patients = X_temporal.shape[0]
    n_temporal_feats = X_temporal.shape[2]
    valid_months = up_to_month + 1

    feature_list: list[np.ndarray] = []
    feature_name_list: list[str] = []

    # 1) Static features
    feature_list.append(X_static)
    feature_name_list.extend(static_names)

    # 2) Flattened temporal features M0..Mt
    temporal_flat = X_temporal[:, :valid_months, :].reshape(n_patients, -1)
    for m in range(valid_months):
        for feat in temporal_names:
            feature_name_list.append(f"M{m}_{feat}")
    feature_list.append(temporal_flat)

    # 3) Aggregates
    temporal_slice = X_temporal[:, :valid_months, :]
    for agg_name, agg_fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        agg = agg_fn(temporal_slice, axis=1)
        feature_list.append(agg)
        for feat in temporal_names:
            feature_name_list.append(f"{agg_name}_{feat}")

    # 4) Trend slopes
    if valid_months >= 2:
        slopes = np.zeros((n_patients, n_temporal_feats), dtype=np.float32)
        x_time = np.arange(valid_months, dtype=np.float32)
        for i in range(n_patients):
            for j in range(n_temporal_feats):
                vals = temporal_slice[i, :, j]
                if np.std(vals) > 1e-8:
                    slope = np.polyfit(x_time, vals, 1)[0]
                else:
                    slope = 0.0
                slopes[i, j] = slope
        feature_list.append(slopes)
        for feat in temporal_names:
            feature_name_list.append(f"trend_{feat}")
    else:
        feature_list.append(np.zeros((n_patients, n_temporal_feats), dtype=np.float32))
        for feat in temporal_names:
            feature_name_list.append(f"trend_{feat}")

    # 5) Latest month
    latest = X_temporal[:, up_to_month, :]
    feature_list.append(latest)
    for feat in temporal_names:
        feature_name_list.append(f"latest_{feat}")

    # 6) Month indicator
    feature_list.append(np.full((n_patients, 1), up_to_month, dtype=np.float32))
    feature_name_list.append("months_available")

    X_flat = np.hstack(feature_list).astype(np.float32)
    X_flat = np.nan_to_num(X_flat, nan=0.0)
    return X_flat, feature_name_list


def main() -> int:
    # __file__ = <repo>/models/Temporal/v2/export_rf_temporal_metadata.py
    repo_root = Path(__file__).resolve().parents[3]
    model_dir = repo_root / "models" / "Temporal" / "v2" / "output" / "random_forest"
    cfg_path = model_dir / "rf_model_config.json"
    model_path = model_dir / "rf_smoteenn_model.pkl"

    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    threshold = float(cfg.get("threshold", 0.67))
    seed = int(cfg.get("seed", 42))

    import sys

    sys.path.append(str(repo_root / "models" / "Temporal"))
    import model_utils as mu  # type: ignore

    csv_path = repo_root / "dataset" / "temporal" / "output" / "cleaned_human_readable.csv"
    model_data = mu.prepare_default_model_inputs(
        str(csv_path),
        train_frac=0.70,
        val_frac=0.20,
        test_frac=0.10,
        random_state=seed,
        drop_feature_cols=mu.get_temporal_v2_drop_feature_cols(),
    )

    X_temporal = model_data["X_temporal"]
    X_static = model_data["X_static"]
    y = model_data["y"]
    static_names = list(model_data["static_feature_names"])
    temporal_names = list(model_data["temporal_feature_names"])
    train_idx = np.asarray(model_data["train_idx"], dtype=int)
    val_idx = np.asarray(model_data["val_idx"], dtype=int)
    test_idx = np.asarray(model_data["test_idx"], dtype=int)

    X_dummy, feature_names = build_features_at_month(
        X_temporal[:1], X_static[:1], up_to_month=12, temporal_names=temporal_names, static_names=static_names
    )
    n_features_full = int(X_dummy.shape[1])
    if n_features_full != 399:
        raise RuntimeError(f"Expected 399 features, got {n_features_full}")
    if len(feature_names) != 399:
        raise RuntimeError("Feature name list must be length 399")
    if len(set(feature_names)) != len(feature_names):
        raise RuntimeError("Feature names are not unique")

    rf_model = joblib.load(model_path)
    if int(getattr(rf_model, "n_features_in_", -1)) != 399:
        raise RuntimeError("rf_smoteenn_model.pkl n_features_in_ != 399")

    metadata = {
        "model_name": "random_forest_temporal",
        "model_type": cfg.get("model", "Random Forest"),
        "augmentation": cfg.get("augmentation", ""),
        "input_name": "float_input",
        "n_features": 399,
        "max_month": 12,
        "threshold": threshold,
        "positive_class": "Success",
        "failure_class_index": 0,
        "success_class_index": 1,
        "label_mapping": {"0": "Failure", "1": "Success"},
        "feature_policy_version": cfg.get("feature_policy_version"),
        "dropped_feature_cols": cfg.get("dropped_feature_cols", []),
        "static_feature_names": static_names,
        "temporal_feature_names": temporal_names,
        "feature_names": feature_names,
        "feature_builder": {
            "function": "build_features_at_month",
            "order": [
                "static",
                "temporal_flattened",
                "temporal_aggregates",
                "temporal_trends",
                "latest_temporal",
                "months_available",
                "right_padding",
            ],
            "aggregate_functions": ["mean", "std", "min", "max"],
            "trend_method": "linear_slope_polyfit",
            "single_month_trend_value": 0,
            "nan_replacement": 0,
        },
        "padding": {"strategy": "right_pad_zeros_to_m12_width", "padding_value": 0},
        "splits": {
            "n_patients": int(len(y)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }

    meta_path = model_dir / "rf_temporal_feature_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    golden = []
    if len(test_idx) > 0:
        pid = int(test_idx[0])
        for month in (0, 3, 6, 9, 12):
            X_m, _ = build_features_at_month(
                X_temporal[[pid]], X_static[[pid]], up_to_month=month, temporal_names=temporal_names, static_names=static_names
            )
            if X_m.shape[1] < 399:
                X_m = np.hstack([X_m, np.zeros((1, 399 - X_m.shape[1]), dtype=np.float32)])
            proba = rf_model.predict_proba(X_m)[0]
            pred = int(rf_model.predict(X_m)[0])
            golden.append(
                {
                    "sample_id": f"test_idx0_m{month}",
                    "patient_index": pid,
                    "month": month,
                    "input_vector": X_m[0].astype(float).tolist(),
                    "failure_probability": float(proba[0]),
                    "success_probability": float(proba[1]),
                    "predicted_label": pred,
                }
            )

    golden_path = model_dir / "rf_temporal_golden_samples.json"
    golden_path.write_text(json.dumps(golden, indent=2), encoding="utf-8")

    print(f"Wrote: {meta_path}")
    print(f"Wrote: {golden_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
