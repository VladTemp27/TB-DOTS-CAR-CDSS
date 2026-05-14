"""Generate a random temporal patient trajectory and run RF ONNX inference.

This is a smoke-test / demo script.

It builds a *coherent* 399-length feature vector in the same layout as the
training notebook for any chosen month t (M0..M12):

1) static baseline (taken from an existing golden sample vector)
2) flattened temporal features for months 0..t
3) aggregates over months 0..t (mean/std/min/max)
4) trend slopes over months 0..t
5) latest month values
6) months_available indicator
7) right-pad zeros to the full 399 width

Notes:
- This does NOT produce clinically realistic patients; it is for verifying
  inference wiring and demonstrating how predictions change across months.
- Use derived labels based on success probability + threshold; the ONNX
  output_label is not reliable for this export.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MonthRow:
    month: int
    cumulative_doses_taken: float
    height: float
    monthly_doses_taken: float
    monthly_missed_doses: float
    pct_adherence: float
    smear_tb_lamp: float
    weight: float
    xpert_mtb_rif: float


def _repo_root() -> Path:
    # <repo>/models/Temporal/v2/random_patient_temporal_inference.py
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build_vector(
    *,
    base_static: np.ndarray,
    temporal_names: list[str],
    temporal_rows: list[dict[str, float]],
    up_to_month: int,
) -> np.ndarray:
    temporal_slice = temporal_rows[: up_to_month + 1]
    X = np.asarray([[r[name] for name in temporal_names] for r in temporal_slice], dtype=np.float32)

    parts = [base_static]
    parts.append(X.reshape(-1))
    parts.extend([X.mean(axis=0), X.std(axis=0), X.min(axis=0), X.max(axis=0)])

    if up_to_month >= 1:
        x_time = np.arange(up_to_month + 1, dtype=np.float32)
        slopes = []
        for j in range(X.shape[1]):
            vals = X[:, j]
            slopes.append(float(np.polyfit(x_time, vals, 1)[0]) if float(np.std(vals)) > 1e-8 else 0.0)
        parts.append(np.asarray(slopes, dtype=np.float32))
    else:
        parts.append(np.zeros((len(temporal_names),), dtype=np.float32))

    parts.append(X[-1])
    parts.append(np.asarray([float(up_to_month)], dtype=np.float32))

    v = np.concatenate(parts).astype(np.float32)
    if v.shape[0] < 399:
        v = np.concatenate([v, np.zeros((399 - v.shape[0],), dtype=np.float32)])
    if v.shape != (399,):
        raise RuntimeError(f"Expected 399-length vector, got {v.shape}")
    return v


def main() -> int:
    root = _repo_root()
    model_dir = root / "models" / "Temporal" / "v2" / "output" / "random_forest"

    meta = _load_json(model_dir / "rf_temporal_feature_metadata.json")
    golden = _load_json(model_dir / "rf_temporal_golden_samples.json")
    if not golden:
        raise RuntimeError("Golden samples missing. Run export_rf_temporal_metadata.py first.")

    static_names = meta["static_feature_names"]
    temporal_names = meta["temporal_feature_names"]
    threshold = float(meta["threshold"])

    # Use a real static baseline slice to keep vector structure plausible.
    static_count = len(static_names)
    base_static = np.asarray(golden[-1]["input_vector"][:static_count], dtype=np.float32)

    # Seed can be overridden by setting RANDOM_SEED env var later if needed.
    # Here we default to a fresh random trajectory each run.
    rng = random.Random()

    height = rng.uniform(150.0, 175.0)
    start_weight = rng.uniform(42.0, 72.0)
    cumulative = 0

    temporal_rows: list[dict[str, float]] = []
    month_rows: list[MonthRow] = []

    for m in range(13):
        doses_taken = float(rng.randint(12, 30))
        missed = float(rng.randint(0, 18))
        cumulative += int(doses_taken)
        total = doses_taken + missed
        pct = (100.0 * doses_taken / total) if total > 0 else 0.0

        # Mild drift with noise. Clamp to plausible minimum.
        weight = max(30.0, start_weight + rng.uniform(-0.9, 0.5) * m + rng.uniform(-1.5, 1.5))

        smear = float(rng.choice([0.0, 0.0, 0.0, 1.0]))
        xpert = float(rng.choice([0.0, 0.0, 0.0, 1.0]))

        row = {
            "cumulative_doses_taken": float(cumulative),
            "height": float(height),
            "is_missing_cumulative_doses_taken": 0.0,
            "is_missing_height": 0.0,
            "is_missing_monthly_doses_taken": 0.0,
            "is_missing_monthly_missed_doses": 0.0,
            "is_missing_pct_adherence": 0.0,
            "is_missing_smear_tb_lamp": 0.0,
            "is_missing_weight": 0.0,
            "is_missing_xpert_mtb_rif": 0.0,
            "monthly_doses_taken": float(doses_taken),
            "monthly_missed_doses": float(missed),
            "pct_adherence": float(pct),
            "smear_tb_lamp": float(smear),
            "weight": float(weight),
            "xpert_mtb_rif": float(xpert),
        }

        temporal_rows.append(row)
        month_rows.append(
            MonthRow(
                month=m,
                cumulative_doses_taken=row["cumulative_doses_taken"],
                height=row["height"],
                monthly_doses_taken=row["monthly_doses_taken"],
                monthly_missed_doses=row["monthly_missed_doses"],
                pct_adherence=row["pct_adherence"],
                smear_tb_lamp=row["smear_tb_lamp"],
                weight=row["weight"],
                xpert_mtb_rif=row["xpert_mtb_rif"],
            )
        )

    import onnxruntime as ort

    onnx_path = model_dir / "random_forest_temporal.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    print("Random temporal patient trajectory")
    print(f"Height: {height:.1f} cm")
    print(f"Start weight: {month_rows[0].weight:.1f} kg")
    print(f"Threshold (success): {threshold:.4f}")

    for m in (0, 3, 6, 9, 12):
        v = _build_vector(base_static=base_static, temporal_names=temporal_names, temporal_rows=temporal_rows, up_to_month=m)
        outs = sess.run(None, {input_name: v.reshape(1, 399)})
        prob_map = outs[1][0]
        raw_label = int(outs[0][0])

        # Some exports encode class-0 as -p1; derive failure from success if needed.
        success = float(prob_map[1])
        failure = (1.0 - success) if float(prob_map[0]) < 0 else float(prob_map[0])
        derived_label = 1 if success >= threshold else 0

        row = month_rows[m]
        print(f"\nM{m}")
        print(f"  weight: {row.weight:.1f} kg")
        print(f"  cumulative_doses_taken: {row.cumulative_doses_taken:.0f}")
        print(f"  monthly_doses_taken/monthly_missed_doses: {row.monthly_doses_taken:.0f}/{row.monthly_missed_doses:.0f}")
        print(f"  pct_adherence: {row.pct_adherence:.1f}%")
        print(f"  smear_tb_lamp: {row.smear_tb_lamp:.0f}")
        print(f"  xpert_mtb_rif: {row.xpert_mtb_rif:.0f}")
        print(f"  raw_onnx_label: {raw_label}")
        print(f"  failure_probability: {failure:.6f}")
        print(f"  success_probability: {success:.6f}")
        print(f"  derived_label: {derived_label} ({'Success' if derived_label == 1 else 'Failure/Risk'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
