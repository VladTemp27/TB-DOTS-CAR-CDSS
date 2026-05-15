#!/usr/bin/env python3
"""
One-shot script: reconstructs the V2 model's 399 semantic feature names via
build_features_at_month(up_to_month=12), groups them into ~14 clinical base
names, and writes a draft feature_groups.json for human review.

The V2 booster uses anonymous Column_0..Column_398 names — booster.feature_name()
is intentionally NOT used. Semantic names come from the feature construction
pipeline that the model was trained on.

Must be run and reviewed before the main pipeline can execute.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import models.Temporal.model_utils as mu
from models.Temporal.model_utils import build_features_at_month

MODEL_DIR = REPO_ROOT / "models" / "Temporal" / "v2" / "output" / "lightgbm"
CSV_PATH = REPO_ROOT / "dataset" / "temporal" / "output" / "cleaned_human_readable.csv"
OUT_PATH = REPO_ROOT / "slm_shap_pipeline" / "feature_groups.json"

# One-hot column prefixes (each dummy family collapses to the source column name)
ONE_HOT_PREFIXES: dict[str, str] = {
    "sex_": "Sex",
    "bacteriologic_status_": "Bacteriologic_Status",
    "registration_group_": "Registration_Group",
    "case_registration_group_": "Registration_Group",
    "smear_microscopy_": "Microscopy_Result",
    "facility_": "Site_Factors",  # V2 added facility; not surfaced in prompt
    # Static one-hot families
    "civil_status_": "Civil_Status",
    "diagnosis_": "Diagnosis_Type",
    "drug_resistance_bacteriological_status_": "Drug_Resistance_Status",
    "chest_x_ray_at_case_notification_": "Chest_Xray",
    "treatment_regimen_": "Treatment_Regimen",
    "nationality_": "Nationality",
    "co_morbidities_": "Comorbidities",
    "name_of_treatment_unit_": "Site_Factors",
    "name_of_diagnosing_facility_": "Site_Factors",
    # Missing-indicator families (temporal)
    "is_missing_": "Missing_Indicators",
}

# Raw/aggregate base name → clinical display name
CLINICAL_DISPLAY_NAMES: dict[str, str] = {
    # Temporal features
    "pct_adherence": "Treatment_Adherence",
    "weight": "Body_Weight",
    "height": "Height",
    "monthly_doses_taken": "Monthly_Doses",
    "cumulative_doses_taken": "Cumulative_Doses",
    "monthly_missed_doses": "Missed_Doses",
    "smear_tb_lamp": "Smear_Result",
    "xpert_mtb_rif": "Xpert_Result",
    "months_available": "Months_Available",
    # Static scalar features
    "age": "Age",
    "days_to_treatment": "Days_To_Treatment",
    "bp_systolic": "Vital_Signs",
    "bp_diastolic": "Vital_Signs",
    "heart_rate": "Vital_Signs",
    "o2_sat": "Vital_Signs",
    "height_cm": "Height",
    "weight_kg": "Body_Weight",
    "smear_microscopy": "Microscopy_Result",
    # Date columns (kept as numeric after encoding)
    "date_of_diagnosis": "Dates",
    "date_of_notification": "Dates",
    "treatment_start_date": "Dates",
    "intensive_phase_start_date": "Dates",
}

_TEMPORAL_ENGINEERED = re.compile(r"^(?:mean|std|min|max|trend|latest)_(.+)$")
_TEMPORAL_RAW = re.compile(r"^M\d+_(.+)$")


def classify(name: str) -> str:
    """Return clinical group name for a reconstructed feature name."""
    # Step 1: check direct static one-hot prefixes
    for prefix, group in ONE_HOT_PREFIXES.items():
        if prefix == "is_missing_":
            continue  # handle separately below
        if name.startswith(prefix):
            return group

    # Step 2: temporal raw M<n>_<base>
    m = _TEMPORAL_RAW.match(name)
    if m:
        base = m.group(1)
        # temporal missing indicators
        if base.startswith("is_missing_"):
            return "Missing_Indicators"
        return CLINICAL_DISPLAY_NAMES.get(base, f"UNMAPPED_{base}")

    # Step 3: temporal engineered (mean_/std_/min_/max_/trend_/latest_)
    m = _TEMPORAL_ENGINEERED.match(name)
    if m:
        base = m.group(1)
        if base.startswith("is_missing_"):
            return "Missing_Indicators"
        return CLINICAL_DISPLAY_NAMES.get(base, f"UNMAPPED_{base}")

    # Step 4: static missing indicators
    if name.startswith("is_missing_"):
        return "Missing_Indicators"

    # Step 5: remaining static scalars / one-hot dummies
    return CLINICAL_DISPLAY_NAMES.get(name, f"STATIC_{name}")


def main() -> None:
    print(f"Loading V2 cohort from {CSV_PATH} ...")
    model_data = mu.prepare_default_model_inputs(
        str(CSV_PATH),
        train_frac=0.70,
        val_frac=0.20,
        test_frac=0.10,
        random_state=42,
        drop_feature_cols=mu.get_temporal_v2_drop_feature_cols(),
    )

    X_static = model_data["X_static"]
    X_temporal = model_data["X_temporal"]
    train_idx = model_data["train_idx"]
    static_names = model_data["static_feature_names"]
    temporal_names = model_data["temporal_feature_names"]

    print("Scaling with train-fitted scalers ...")
    scaled = mu.scale_full_arrays(X_static, X_temporal, train_idx)
    X_static_s = scaled["X_static_scaled"]
    X_temporal_s = scaled["X_temporal_scaled"]

    print("Building feature matrix at M12 ...")
    X_flat, feature_names = build_features_at_month(
        X_temporal_s, X_static_s, up_to_month=12,
        temporal_names=temporal_names, static_names=static_names,
    )
    print(f"Total features: {len(feature_names)}")

    # Group by clinical name
    groups: dict[str, list[str]] = {}
    for name in feature_names:
        group = classify(name)
        groups.setdefault(group, []).append(name)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(groups, f, indent=2, sort_keys=True)

    print(f"\nWrote {OUT_PATH}")
    unmapped = [k for k in groups if k.startswith("UNMAPPED_") or k.startswith("STATIC_")]
    print(f"  Clinical groups:   {len(groups) - len(unmapped)}")
    print(f"  Unmapped/static:   {len(unmapped)}")
    print(f"  Total groups:      {len(groups)}")
    if unmapped:
        print(f"\n  Unmapped features -- add to CLINICAL_DISPLAY_NAMES or ONE_HOT_PREFIXES:")
        for k in sorted(unmapped):
            print(f"    {k}: {groups[k][:3]}")
    else:
        print("\nAll features mapped to clinical groups.")
    print("\n  MANUAL REVIEW REQUIRED: Open slm_shap_pipeline/feature_groups.json and verify")
    print("   each group before running the main pipeline.")


if __name__ == "__main__":
    main()
