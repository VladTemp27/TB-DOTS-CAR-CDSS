"""Smoke test to verify patient-level splitting and train-only scaling.

Run this script from repository root or the dataset/temporal folder.
"""
from pathlib import Path
import numpy as np
from models.Temporal.model_utils import load_cleaned_csv, patient_level_split, scale_train_val_test, save_scalers


def main():
    data_csv = Path(__file__).resolve().parent / "output" / "cleaned_human_readable.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Expected cleaned CSV at {data_csv}")

    print(f"Loading cleaned CSV from: {data_csv}")
    model_data = load_cleaned_csv(str(data_csv))

    X_temporal = model_data["X_temporal"]
    X_static = model_data["X_static"]
    n_patients = model_data["n_patients"]

    print(f"Loaded data: n_patients={n_patients}, X_temporal={X_temporal.shape}, X_static={X_static.shape}")

    train_idx, val_idx, test_idx = patient_level_split(n_patients, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42)
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    scaled = scale_train_val_test(X_static, X_temporal, train_idx, val_idx, test_idx)

    # Basic checks
    for split in ("train", "val", "test"):
        xt = scaled["X_temporal_scaled"][split]
        xs = scaled["X_static_scaled"][split]
        print(f"{split}: X_temporal {xt.shape}, X_static {xs.shape}, nan_counts: temporal={np.isnan(xt).sum()}, static={np.isnan(xs).sum()}")

    s1, s2 = save_scalers(scaled["scaler_static"], scaled["scaler_temporal"], output_dir=str(Path(__file__).resolve().parent / "output"))
    print(f"Saved scalers: {s1}, {s2}")


if __name__ == "__main__":
    main()
