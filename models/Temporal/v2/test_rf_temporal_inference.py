"""Sanity tests for temporal Random Forest ONNX/ORT inference.

Runs three groups of checks:
1) Golden-sample equivalence: feeds saved vectors and compares probabilities.
2) Synthetic low-risk vs high-risk: perturbs a golden vector directionally.
3) Optional ORT file load test.

Exit code is non-zero if any required check fails.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    label: int
    failure_probability: float
    success_probability: float


def _repo_root() -> Path:
    # <repo>/models/Temporal/v2/test_rf_temporal_inference.py
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _soft_equal(a: float, b: float, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= (atol + rtol * abs(b))


def _extract_probs(outputs: list[Any], output_names: list[str]) -> tuple[int | None, float, float]:
    """Handle common sklearn-onnx RF output shapes.

    Typical for skl2onnx RandomForestClassifier:
    - output_label: array([0])
    - output_probability: list[{0: 0.12, 1: 0.88}]
    """

    label: int | None = None
    failure: float | None = None
    success: float | None = None

    def _maybe_label(x: Any) -> int | None:
        if x is None:
            return None
        if isinstance(x, (np.ndarray, list, tuple)):
            arr = np.asarray(x)
            if arr.size >= 1:
                try:
                    return int(arr.reshape(-1)[0])
                except Exception:
                    return None
        return None

    def _maybe_probmap(x: Any) -> tuple[float | None, float | None]:
        # Most common: list[dict[int, float]]
        if isinstance(x, list) and x and isinstance(x[0], dict):
            m = x[0]
            f = m.get(0) if 0 in m else m.get(0.0)
            s = m.get(1) if 1 in m else m.get(1.0)
            f_out = float(f) if f is not None else None
            s_out = float(s) if s is not None else None
            if f_out is not None and s_out is not None:
                # Some exports encode only class-1 probability and store class-0 as -p1.
                # If so, derive failure as (1 - success).
                if f_out < 0 and abs(f_out + s_out) < 1e-6 and 0.0 <= s_out <= 1.0:
                    f_out = 1.0 - s_out
            return (f_out, s_out)

        # Sometimes: dict directly
        if isinstance(x, dict):
            f = x.get(0) if 0 in x else x.get(0.0)
            s = x.get(1) if 1 in x else x.get(1.0)
            f_out = float(f) if f is not None else None
            s_out = float(s) if s is not None else None
            if f_out is not None and s_out is not None:
                if f_out < 0 and abs(f_out + s_out) < 1e-6 and 0.0 <= s_out <= 1.0:
                    f_out = 1.0 - s_out
            return (f_out, s_out)

        # Sometimes: array shape (1,2)
        if isinstance(x, np.ndarray) and x.size >= 2:
            a = x.reshape(-1)
            return (float(a[0]), float(a[1]))
        return (None, None)

    # Try name-based mapping first.
    for name, out in zip(output_names, outputs):
        lname = name.lower()
        if "label" in lname and label is None:
            label = _maybe_label(out)
        if "prob" in lname and (failure is None or success is None):
            f, s = _maybe_probmap(out)
            if f is not None:
                failure = f
            if s is not None:
                success = s

    # Fallback positional mapping.
    if (failure is None or success is None) and len(outputs) >= 2:
        f, s = _maybe_probmap(outputs[1])
        if f is not None:
            failure = f
        if s is not None:
            success = s
    if label is None and outputs:
        label = _maybe_label(outputs[0])

    if failure is None or success is None:
        raise RuntimeError(f"Unable to parse probability output. output_names={output_names}")

    # Clamp tiny numeric noise.
    f2 = float(failure)
    s2 = float(success)
    if 0.0 <= s2 <= 1.0 and (f2 < 0.0 or f2 > 1.0):
        # If success looks like a probability but failure doesn't, derive failure.
        f2 = 1.0 - s2
    f2 = float(min(1.0, max(0.0, f2)))
    s2 = float(min(1.0, max(0.0, s2)))
    return label, f2, s2


def run_onnx(path: Path, vector: list[float]) -> InferenceResult:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    x = np.asarray(vector, dtype=np.float32)
    if x.shape != (399,):
        raise ValueError(f"Expected vector length 399, got {x.shape}")
    outputs = sess.run(None, {input_name: x.reshape(1, 399)})
    label, failure, success = _extract_probs(outputs, out_names)
    if label is None:
        # Derive from probabilities.
        label = 0 if failure >= success else 1
    return InferenceResult(label=int(label), failure_probability=failure, success_probability=success)


def _apply_synthetic_profile(
    base: list[float],
    feature_names: list[str],
    *,
    pct_adherence: float,
    monthly_doses_taken: float,
    monthly_missed_doses: float,
    smear_tb_lamp: float,
    xpert_mtb_rif: float,
    weight: float,
) -> list[float]:
    v = np.asarray(base, dtype=np.float32).copy()
    if v.shape != (399,):
        raise ValueError("base vector must be 399")

    def set_all(suffix: str, value: float) -> None:
        for i, n in enumerate(feature_names):
            if n.endswith(suffix) or n == suffix or n.startswith(suffix):
                v[i] = np.float32(value)

    # Apply to raw monthly values and derived feature blocks.
    for i, n in enumerate(feature_names):
        if n.endswith("_pct_adherence"):
            v[i] = np.float32(pct_adherence)
        elif n.endswith("_monthly_doses_taken"):
            v[i] = np.float32(monthly_doses_taken)
        elif n.endswith("_monthly_missed_doses"):
            v[i] = np.float32(monthly_missed_doses)
        elif n.endswith("_smear_tb_lamp"):
            v[i] = np.float32(smear_tb_lamp)
        elif n.endswith("_xpert_mtb_rif"):
            v[i] = np.float32(xpert_mtb_rif)
        elif n.endswith("_weight"):
            v[i] = np.float32(weight)

    # Also set aggregate/latest/trend features.
    set_all("mean_pct_adherence", pct_adherence)
    set_all("latest_pct_adherence", pct_adherence)
    set_all("min_pct_adherence", pct_adherence)
    set_all("max_pct_adherence", pct_adherence)
    set_all("trend_pct_adherence", 0.0)

    set_all("mean_monthly_missed_doses", monthly_missed_doses)
    set_all("latest_monthly_missed_doses", monthly_missed_doses)
    set_all("trend_monthly_missed_doses", 0.0)

    set_all("mean_monthly_doses_taken", monthly_doses_taken)
    set_all("latest_monthly_doses_taken", monthly_doses_taken)
    set_all("trend_monthly_doses_taken", 0.0)

    set_all("latest_smear_tb_lamp", smear_tb_lamp)
    set_all("mean_smear_tb_lamp", smear_tb_lamp)
    set_all("latest_xpert_mtb_rif", xpert_mtb_rif)
    set_all("mean_xpert_mtb_rif", xpert_mtb_rif)

    set_all("latest_weight", weight)
    set_all("mean_weight", weight)

    return v.astype(float).tolist()


def main() -> int:
    root = _repo_root()
    model_dir = root / "models" / "Temporal" / "v2" / "output" / "random_forest"
    onnx_path = model_dir / "random_forest_temporal.onnx"
    ort_path = model_dir / "random_forest_temporal.ort"
    meta_path = model_dir / "rf_temporal_feature_metadata.json"
    golden_path = model_dir / "rf_temporal_golden_samples.json"

    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not golden_path.exists():
        raise FileNotFoundError(golden_path)

    meta = _load_json(meta_path)
    feature_names = list(meta["feature_names"])
    threshold = float(meta.get("threshold", 0.67))

    golden = _load_json(golden_path)
    if not isinstance(golden, list) or not golden:
        raise RuntimeError("Golden samples missing/empty")

    print(f"Testing model: {onnx_path}")
    print(f"Threshold: {threshold:.4f}")

    failures = 0

    # 1) Golden samples equivalence.
    for sample in golden:
        sid = sample["sample_id"]
        vec = sample["input_vector"]
        exp_f = float(sample["failure_probability"])
        exp_s = float(sample["success_probability"])
        got = run_onnx(onnx_path, vec)

        ok = _soft_equal(got.failure_probability, exp_f, atol=1e-5, rtol=1e-5) and _soft_equal(
            got.success_probability, exp_s, atol=1e-5, rtol=1e-5
        )
        print(f"\nGolden {sid}:")
        print(f"Expected failure={exp_f:.6f} success={exp_s:.6f}")
        print(f"Actual   failure={got.failure_probability:.6f} success={got.success_probability:.6f} label={got.label}")
        print("PASS" if ok else "FAIL")
        if not ok:
            failures += 1

    # 2) Synthetic low/high risk perturbation (directional).
    base = None
    for s in golden:
        if int(s.get("month", -1)) == 12:
            base = s
            break
    if base is None:
        base = golden[0]

    base_vec = base["input_vector"]
    low_vec = _apply_synthetic_profile(
        base_vec,
        feature_names,
        pct_adherence=100.0,
        monthly_doses_taken=30.0,
        monthly_missed_doses=0.0,
        smear_tb_lamp=0.0,
        xpert_mtb_rif=0.0,
        weight=70.0,
    )
    high_vec = _apply_synthetic_profile(
        base_vec,
        feature_names,
        pct_adherence=20.0,
        monthly_doses_taken=5.0,
        monthly_missed_doses=25.0,
        smear_tb_lamp=1.0,
        xpert_mtb_rif=1.0,
        weight=40.0,
    )

    low = run_onnx(onnx_path, low_vec)
    high = run_onnx(onnx_path, high_vec)
    print("\nSynthetic low-risk:")
    print(f"failure={low.failure_probability:.6f} success={low.success_probability:.6f} label={low.label}")
    print("Synthetic high-risk:")
    print(f"failure={high.failure_probability:.6f} success={high.success_probability:.6f} label={high.label}")

    directional_ok = high.failure_probability > low.failure_probability
    # Directional sanity is informative only; synthetic edits may not reflect training distribution.
    print("PASS" if directional_ok else "WARN")

    # 3) Optional ORT load/run.
    if ort_path.exists():
        try:
            _ = run_onnx(ort_path, golden[0]["input_vector"])
            print(f"\nORT load/run: PASS ({ort_path.name})")
        except Exception as e:
            print(f"\nORT load/run: FAIL ({ort_path.name}) {type(e).__name__}: {str(e)[:200]}")
            # ORT export is optional; do not fail the test run for this.

    if failures:
        print(f"\nFAILED checks: {failures}")
        return 1

    print("\nALL REQUIRED CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
