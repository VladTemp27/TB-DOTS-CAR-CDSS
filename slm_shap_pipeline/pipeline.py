from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import time

from slm_shap_pipeline.case_writer import write_case, write_manifest
from slm_shap_pipeline.config import PipelineConfig
from slm_shap_pipeline.data_loader import load_test_patients
from slm_shap_pipeline.feature_aggregator import aggregate_shap, build_groups, validate_coverage
from slm_shap_pipeline.feature_builder import build_features_for_patient
from slm_shap_pipeline.model_loader import load_model_bundle
from slm_shap_pipeline.prompt_builder import build_prompt_for_patient
from slm_shap_pipeline.providers.base import make_provider
from slm_shap_pipeline.shap_runner import compute_shap_for_row
from slm_shap_pipeline.slm_client import call_slm, make_cache_key


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run(
    condition: str,
    config: PipelineConfig | None = None,
    patient_limit: int | None = None,
    dry_run_gemini: bool = False,
) -> Path:
    """Run the pipeline for one condition. Returns path to output condition directory."""
    if config is None:
        config = PipelineConfig()
    if condition not in ("blind", "sighted"):
        raise ValueError(f"condition must be 'blind' or 'sighted', got {condition!r}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = config.output_base / timestamp / condition
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] Loading V2 model bundle ...")
    bundle = load_model_bundle(config)
    feature_groups = build_groups(config.feature_groups_path)
    groups_hash = _file_sha256(config.feature_groups_path)
    prompt_git_sha = _git_short_sha()

    print(f"[pipeline] Initializing SLM provider: {config.provider}")
    provider = make_provider(config)
    print(f"[pipeline]   provider.name = {provider.name}, model = {config.gemini_model}")

    print(f"[pipeline] Loading V2 test cohort from {config.csv_path} ...")
    cohort = load_test_patients(config)          # TestCohort, not a list
    patients = cohort.patients
    if patient_limit is not None:
        patients = patients[:patient_limit]
    print(f"[pipeline] {len(patients)} patients to process (month M{config.month_of_prediction}).")

    stats: dict[str, int | float] = {"cache_hits": 0, "cache_misses": 0, "gemini_failures": 0}
    t0 = time()

    for i, patient in enumerate(patients):
        print(f"[pipeline] ({i+1}/{len(patients)}) {patient.patient_id} ...")

        X_flat, feature_names = build_features_for_patient(patient, up_to_month=config.month_of_prediction)
        model_prediction = float(bundle.booster.predict(X_flat)[0])
        raw_shap = compute_shap_for_row(bundle.booster, X_flat, feature_names)
        validate_coverage(raw_shap, feature_groups)
        aggregated = aggregate_shap(raw_shap, feature_groups, epsilon=config.sign_epsilon)

        prompt = build_prompt_for_patient(
            patient, model_prediction, aggregated,
            condition=condition, month_of_prediction=config.month_of_prediction,
        )

        cache_key = make_cache_key(
            patient.patient_id, condition, config.month_of_prediction, prompt,
            bundle.model_hash, cohort.scaler_static_hash, cohort.scaler_temporal_hash, groups_hash,
        )

        # call_slm uses .json extension for cache files
        cache_path = config.cache_dir / f"{cache_key}.json"
        is_cache_hit = cache_path.exists()

        try:
            explanation = call_slm(
                prompt, cache_key, provider,
                cache_dir=config.cache_dir,
                dry_run=dry_run_gemini,
            )
            if is_cache_hit:
                stats["cache_hits"] += 1
            else:
                stats["cache_misses"] += 1
        except Exception as e:
            print(f"  [WARN] SLM call failed for {patient.patient_id} via {provider.name}: {e}")
            stats["gemini_failures"] += 1
            continue

        metadata = {
            "model_hash": bundle.model_hash,
            "scaler_static_hash": cohort.scaler_static_hash,
            "scaler_temporal_hash": cohort.scaler_temporal_hash,
            "feature_policy_version": "temporal_v2_cleaned_output_facility_v1",
            "feature_groups_hash": groups_hash,
            "prompt_git_sha": prompt_git_sha,
            "gemini_call_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_case(
            patient_id=patient.patient_id,
            condition=condition,
            explanation=explanation,
            aggregated_shap=aggregated,
            model_prediction=model_prediction,
            metadata=metadata,
            out_dir=cases_dir,
            month_of_prediction=config.month_of_prediction,
        )

    stats["duration_seconds"] = int(time() - t0)

    write_manifest(
        out_dir=out_dir,
        condition=condition,
        config_snapshot={
            "model_path": str(config.model_path),
            "model_hash": bundle.model_hash,
            "scaler_static_hash": cohort.scaler_static_hash,
            "scaler_temporal_hash": cohort.scaler_temporal_hash,
            "dataset_path": str(config.csv_path),
            "month_of_prediction": config.month_of_prediction,
            "patient_count": len(patients),
            "feature_policy_version": "temporal_v2_cleaned_output_facility_v1",
            "feature_groups_hash": groups_hash,
            "prompt_git_sha": prompt_git_sha,
            "provider": provider.name,
            "model": config.gemini_model,
        },
        stats=stats,
    )

    # Update 'latest' symlink
    latest_link = config.output_base / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(timestamp)

    print(f"[pipeline] Done. Output: {out_dir}")
    print(f"  cache_hits={stats['cache_hits']}  failures={stats['gemini_failures']}")
    return out_dir
