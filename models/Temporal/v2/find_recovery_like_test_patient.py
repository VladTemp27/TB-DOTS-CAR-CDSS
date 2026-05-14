"""Find real test patients whose prediction improves over time.

Searches the held-out test split from the temporal dataset.

Two reports are produced:
1) "Flip" matches (strict): derived label at M0 is risk and at M12 is success.
2) Top improvements (directional): largest drop in failure probability from M0 to M12.

Then prints the first few matches with probabilities at:
M0, M3, M6, M9, M12

This is a more meaningful check than synthetic vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MonthPred:
    month: int
    failure: float
    success: float
    label: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build_features_at_month(
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    *,
    up_to_month: int,
    temporal_names: list[str],
    static_names: list[str],
) -> np.ndarray:
    """Mirror the notebook feature builder and right-pad to M12 width."""

    # static
    feature_list = [X_static]

    # temporal flattened
    valid_months = up_to_month + 1
    temporal_flat = X_temporal[:, :valid_months, :].reshape(X_temporal.shape[0], -1)
    feature_list.append(temporal_flat)

    # aggregates
    temporal_slice = X_temporal[:, :valid_months, :]
    feature_list.extend(
        [
            np.mean(temporal_slice, axis=1),
            np.std(temporal_slice, axis=1),
            np.min(temporal_slice, axis=1),
            np.max(temporal_slice, axis=1),
        ]
    )

    # trends
    n_patients = X_temporal.shape[0]
    n_temporal_feats = X_temporal.shape[2]
    if valid_months >= 2:
        slopes = np.zeros((n_patients, n_temporal_feats), dtype=np.float32)
        x_time = np.arange(valid_months, dtype=np.float32)
        for i in range(n_patients):
            for j in range(n_temporal_feats):
                vals = temporal_slice[i, :, j]
                if np.std(vals) > 1e-8:
                    slopes[i, j] = np.polyfit(x_time, vals, 1)[0]
                else:
                    slopes[i, j] = 0.0
        feature_list.append(slopes)
    else:
        feature_list.append(np.zeros((n_patients, n_temporal_feats), dtype=np.float32))

    # latest
    feature_list.append(X_temporal[:, up_to_month, :])

    # months_available
    feature_list.append(np.full((n_patients, 1), up_to_month, dtype=np.float32))

    X_flat = np.hstack(feature_list).astype(np.float32)
    X_flat = np.nan_to_num(X_flat, nan=0.0)

    # right-pad to full M12 width
    full_width = len(static_names) + (13 * len(temporal_names)) + (4 * len(temporal_names)) + len(temporal_names) + len(temporal_names) + 1
    # The model is trained on full M12 width which should be 399.
    target_width = 399
    if full_width != target_width:
        # Keep padding based on actual current computed width for safety.
        target_width = max(target_width, int(full_width))
    if X_flat.shape[1] < 399:
        X_flat = np.hstack([X_flat, np.zeros((X_flat.shape[0], 399 - X_flat.shape[1]), dtype=np.float32)])
    if X_flat.shape[1] != 399:
        raise RuntimeError(f"Expected 399 features, got {X_flat.shape[1]}")
    return X_flat


def main() -> int:
    root = _repo_root()
    model_dir = root / "models" / "Temporal" / "v2" / "output" / "random_forest"
    meta = _load_json(model_dir / "rf_temporal_feature_metadata.json")
    threshold = float(meta["threshold"])

    import sys

    sys.path.append(str(root / "models" / "Temporal"))
    import model_utils as mu  # type: ignore

    csv_path = root / "dataset" / "temporal" / "output" / "cleaned_human_readable.csv"
    model_data = mu.prepare_default_model_inputs(
        str(csv_path),
        train_frac=0.70,
        val_frac=0.20,
        test_frac=0.10,
        random_state=42,
        drop_feature_cols=mu.get_temporal_v2_drop_feature_cols(),
    )

    X_temporal = model_data["X_temporal"]
    X_static = model_data["X_static"]
    y = np.asarray(model_data["y"], dtype=int)
    static_names = list(model_data["static_feature_names"])
    temporal_names = list(model_data["temporal_feature_names"])
    test_idx = np.asarray(model_data["test_idx"], dtype=int)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_dir / "random_forest_temporal.onnx"), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def pred_for(pid: int, month: int) -> MonthPred:
        X_m = _build_features_at_month(
            X_temporal[[pid]],
            X_static[[pid]],
            up_to_month=month,
            temporal_names=temporal_names,
            static_names=static_names,
        )
        outs = sess.run(None, {input_name: X_m.astype(np.float32)})
        prob_map = outs[1][0]
        success = float(prob_map[1])
        failure = (1.0 - success) if float(prob_map[0]) < 0 else float(prob_map[0])
        label = 1 if success >= threshold else 0
        return MonthPred(month=month, failure=failure, success=success, label=label)

    matches = []
    improvements = []
    for pid in test_idx.tolist():
        p0 = pred_for(pid, 0)
        p12 = pred_for(pid, 12)
        if p0.label == 0 and p12.label == 1:
            matches.append(pid)
        improvements.append((pid, p0.failure - p12.failure, p0, p12))

    print("Recovery-like test patient search")
    print(f"Threshold (success): {threshold:.4f}")
    print(f"Test set size: {len(test_idx)}")
    print(f"Matches (M0 risk -> M12 success): {len(matches)}")

    show = matches[:5]
    if show:
        for pid in show:
            actual = int(y[pid])
            print(f"\nFlip match patient index {pid} | actual y={actual} ({'Success' if actual==1 else 'Failure'})")
            for m in (0, 3, 6, 9, 12):
                pm = pred_for(pid, m)
                print(
                    f"  M{m}: success={pm.success:.6f} failure={pm.failure:.6f} derived_label={pm.label}"
                    f" ({'Success' if pm.label==1 else 'Failure/Risk'})"
                )
    else:
        print("No flip matches found (using derived label + stored threshold).")

    # Always print the top directional improvers.
    improvements.sort(key=lambda t: t[1], reverse=True)
    print("\nTop 5 directional improvers (largest failure drop M0->M12):")
    for pid, delta, p0, p12 in improvements[:5]:
        actual = int(y[pid])
        print(
            f"  patient {pid} | actual={'Success' if actual==1 else 'Failure'} | "
            f"M0 failure={p0.failure:.6f} -> M12 failure={p12.failure:.6f} (delta={delta:+.6f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
