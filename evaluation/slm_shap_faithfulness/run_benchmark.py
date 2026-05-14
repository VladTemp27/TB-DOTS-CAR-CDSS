"""Benchmark runner — mode-invariant entrypoint for slm_shap_faithfulness evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.slm_shap_faithfulness.config import BenchmarkConfig
from evaluation.slm_shap_faithfulness.feature_map import canonicalize_feature
from evaluation.slm_shap_faithfulness.parser import parse_explanation
from evaluation.slm_shap_faithfulness.scorer import score_case
from evaluation.slm_shap_faithfulness.shap_truth import build_truth_rows
from evaluation.slm_shap_faithfulness.io import write_results, read_results
from evaluation.slm_shap_faithfulness.regression import compare_runs
from evaluation.slm_shap_faithfulness.adapters.artifact_adapter import load_cases_from_artifacts
from evaluation.slm_shap_faithfulness.adapters.runtime_adapter import load_cases_from_runtime


def run_benchmark(
    input_dir: str | Path,
    output_dir: str | Path,
    mode: str,
    baseline_path: str | Path | None = None,
    cfg: BenchmarkConfig | None = None,
) -> dict:
    """Run the SLM-to-SHAP faithfulness benchmark.

    Args:
        input_dir: Directory containing input case files.
        output_dir: Directory where results.json will be written.
        mode: Ingestion mode — "artifact" or "runtime".
        baseline_path: Optional path to a previous results.json for regression comparison.
        cfg: BenchmarkConfig instance. Uses defaults if None.

    Returns:
        Dict with keys:
            patients: list of per-patient scoring dicts
            summary: dict with pass_rate and aggregate metrics
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    if mode not in cfg.allowed_modes:
        raise ValueError(f"mode must be one of {cfg.allowed_modes}, got {repr(mode)}")

    # Load cases via the appropriate adapter
    if mode == "artifact":
        cases = load_cases_from_artifacts(input_dir)
    else:
        cases = load_cases_from_runtime(input_dir)

    patient_results = []
    for case in cases:
        shap_series = pd.Series(case["shap_values"])
        truth_rows = build_truth_rows(shap_series, top_k=cfg.top_k) if not shap_series.empty else []
        parsed = parse_explanation(case["explanation"])
        # Canonicalize feature names in claims to match shap_values keys (underscore form)
        claims = []
        for claim in parsed["claims"]:
            canonical = canonicalize_feature(claim["feature"])
            if canonical is not None:
                claims.append({**claim, "feature": canonical})
            else:
                # Keep as-is if no canonical form found (single-word features like Age, BMI)
                claims.append(claim)
        score = score_case(truth_rows, claims, cfg=cfg)
        patient_results.append({
            "patient_id": case["patient_id"],
            "mode": mode,
            **score,
        })

    total = len(patient_results)
    passed = sum(1 for p in patient_results if p["passed"])
    pass_rate = round(passed / total, 6) if total > 0 else 0.0

    summary = {
        "total_cases": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "mode": mode,
    }

    results = {"patients": patient_results, "summary": summary}

    # Regression comparison: if baseline_path provided, compute and attach deltas
    if baseline_path is not None:
        try:
            baseline = read_results(baseline_path)
            baseline_summary = baseline.get("summary", {})
            regression = compare_runs(baseline_summary, summary)
            results["regression"] = regression
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not read baseline from {baseline_path}: {exc}"
            ) from exc

    write_results(output_dir, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SLM-to-SHAP faithfulness benchmark."
    )
    parser.add_argument("--input-dir", required=True, help="Input directory with case files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for results.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["artifact", "runtime"],
        help="Ingestion mode.",
    )
    parser.add_argument(
        "--baseline-path",
        default=None,
        help="Path to previous results.json for regression comparison.",
    )
    args = parser.parse_args()
    results = run_benchmark(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        baseline_path=args.baseline_path,
    )
    summary = results["summary"]
    print(f"Benchmark complete. pass_rate={summary['pass_rate']:.2%} ({summary['passed']}/{summary['total_cases']})")


if __name__ == "__main__":
    main()
