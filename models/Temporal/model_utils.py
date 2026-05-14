"""Utilities for loading the combined temporal dataset, patient-level splitting,
and train-only scaling for temporal models.

These helpers are intended to be used by the temporal model notebooks
so that scalers are fitted on the training set only and then applied
to validation and test sets.
"""
from importlib import util as importlib_util
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure the shared preprocessing module can be imported whether this
# helper is run from the notebook folder or as a script.
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]


def _load_preprocessing_module():
    module_path = REPO_ROOT / "dataset" / "temporal" / "preprocessingV2.py"
    spec = importlib_util.spec_from_file_location("preprocessingV2", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load preprocessingV2 from {module_path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules.setdefault("preprocessingV2", module)
    spec.loader.exec_module(module)
    return module


pp = _load_preprocessing_module()

SUCCESS_OUTCOMES = {"Cured", "Treatment Completed"}
TRAINING_DROP_COLUMNS = {"data_year", "source_file"}
TEMPORAL_V2_FEATURE_POLICY = "temporal_v2_cleaned_output_facility_v1"
TEMPORAL_V2_DROP_COLUMNS = TRAINING_DROP_COLUMNS | {
    # Outcome-time leakage: valid for retrospective timelines, not prediction inputs.
    "date_of_outcome",
    "is_missing_outcome",
    # Administrative identifiers can let flexible models memorize records/sites.
    "no",
    "name_of_diagnosing_facility",
    "name_of_treatment_unit",
    # Date of birth is redundant with age and unnecessarily identifying.
    "date_of_birth",
    # Late/final treatment process fields are not available at early prediction months.
    "regimen_type_at_end_of_treatment",
    "regimen_type_at_6th_month_of_treatment",
    "intensive_phase_end_date",
    "continuation_phase_start_date",
    "continuation_phase_end_date",
    # Raw duplicated/high-cardinality strings are parsed into numeric columns above.
    "blood_pressure",
    "height",
    "weight",
    # Near-empty/free-text fields are unstable in small medical cohorts.
    "tuberculosis_culture",
    "tuberculin_skin_test",
    "other_lab_test",
    "other",
    "others",
    "dat_supported_dup",
    "risk_factors_for_drug_resistance_tuberculosis",
}


def get_temporal_v2_drop_feature_cols() -> set[str]:
    """Columns excluded from v2 temporal predictors for leakage control."""
    return set(TEMPORAL_V2_DROP_COLUMNS)


def _resolve_source_csv(csv_path: str | None) -> Path:
    """Resolve the cleaned temporal dataset path used by the model notebooks.

    The v2 notebooks historically passed ``combined_complete_dataset.csv``.
    Keep that call stable but route it to the audited missing-data output.
    """
    default_path = REPO_ROOT / "dataset" / "temporal" / "output" / "cleaned_human_readable.csv"
    if csv_path is None:
        return default_path

    requested_path = Path(csv_path)
    if not requested_path.is_absolute():
        requested_path = REPO_ROOT / requested_path
    if requested_path.exists():
        return requested_path
    if requested_path.name in {"combined_complete_dataset.csv", "cleaned_human_readable.csv"}:
        return default_path
    return requested_path


def _build_label_array(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create binary labels from harmonized outcomes.

    Success is defined as {Cured, Treatment Completed}. Any other non-missing
    harmonized outcome is treated as Failure. Missing outcomes are excluded.
    """
    if "outcome" not in df.columns:
        raise KeyError("Expected 'outcome' column after preprocessing")

    outcome_series = df["outcome"]
    valid_mask = outcome_series.notna()
    y = outcome_series.loc[valid_mask].map(lambda value: 1.0 if value in SUCCESS_OUTCOMES else 0.0)
    return y.to_numpy(dtype=np.float32), valid_mask.to_numpy(dtype=bool)


def _filter_observed_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only patients whose outcome was observed, not imputed."""
    if "is_missing_outcome" not in df.columns:
        return df

    observed_mask = pd.to_numeric(df["is_missing_outcome"], errors="coerce").fillna(1).eq(0)
    return df.loc[observed_mask].reset_index(drop=True)


def _convert_datetime_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to numeric day offsets so they can be modeled."""
    converted = df.copy()
    datetime_cols = converted.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    epoch = pd.Timestamp("1970-01-01")
    for col in datetime_cols:
        series = pd.to_datetime(converted[col], errors="coerce")
        converted[col] = ((series - epoch) / pd.Timedelta(days=1)).astype(float)
    return converted


def _fill_static_categoricals(train_df: pd.DataFrame, split_df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    filled = split_df.copy()
    for col in cat_cols:
        pct_missing = train_df[col].isna().mean()
        if pct_missing > 0.90:
            fill_value = "Unknown"
        else:
            mode = train_df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else "Unknown"
        filled[col] = filled[col].fillna(fill_value).astype(str)
    return filled


def _fill_temporal_categoricals(train_df: pd.DataFrame, split_df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    filled = split_df.sort_values(["patient_id", "month"]).copy()
    for col in cat_cols:
        filled[col] = filled.groupby("patient_id")[col].ffill()
        pct_missing = train_df[col].isna().mean()
        if pct_missing > 0.90:
            fill_value = "Unknown"
        else:
            mode = train_df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else "Unknown"
        filled[col] = filled[col].fillna(fill_value).astype(str)
    return filled


def _get_temporal_feature_names(df_temporal: pd.DataFrame) -> list[str]:
    temporal_feature_names = [c for c in df_temporal.columns if c not in {"patient_id", "month"}]
    preferred_order = [
        "cumulative_doses_taken",
        "height",
        "monthly_doses_taken",
        "monthly_missed_doses",
        "pct_adherence",
        "smear_tb_lamp",
        "weight",
        "xpert_mtb_rif",
    ]
    return [feat for feat in preferred_order if feat in temporal_feature_names] + [
        feat for feat in temporal_feature_names if feat not in preferred_order
    ]


def _build_model_arrays(df_static_model: pd.DataFrame, df_temporal_model: pd.DataFrame):
    """Assemble numeric static + temporal arrays after split-safe transforms."""
    patient_ids = df_static_model["patient_id"].to_numpy(dtype=np.int64)
    static_feature_names = [c for c in df_static_model.columns if c != "patient_id"]
    X_static = df_static_model[static_feature_names].to_numpy(dtype=np.float32)

    temporal_feature_names = _get_temporal_feature_names(df_temporal_model)
    n_patients = len(patient_ids)
    n_timesteps = int(df_temporal_model["month"].max()) + 1 if len(df_temporal_model) else len(pp.MONTH_RANGE)
    n_temporal_features = len(temporal_feature_names)
    X_temporal = np.zeros((n_patients, n_timesteps, n_temporal_features), dtype=np.float32)

    df_temporal_sorted = df_temporal_model.sort_values(["patient_id", "month"])
    for j, feat in enumerate(temporal_feature_names):
        pivot = df_temporal_sorted.pivot(index="patient_id", columns="month", values=feat)
        pivot = pivot.reindex(patient_ids)
        pivot = pivot.reindex(columns=range(n_timesteps))
        X_temporal[:, :, j] = pivot.to_numpy(dtype=np.float32)

    X_temporal = np.nan_to_num(X_temporal, nan=0.0)
    X_combined_flat = np.hstack([X_static, X_temporal.reshape(n_patients, -1)]).astype(np.float32)
    return {
        "X_temporal": X_temporal,
        "X_static": X_static,
        "X_combined_flat": X_combined_flat,
        "patient_ids": patient_ids,
        "static_feature_names": static_feature_names,
        "temporal_feature_names": temporal_feature_names,
        "combined_feature_names": static_feature_names
        + [f"M{m}_{feat}" for m in range(n_timesteps) for feat in temporal_feature_names],
        "n_patients": n_patients,
        "n_timesteps": n_timesteps,
        "n_temporal_features": n_temporal_features,
    }


def load_cleaned_csv(csv_path: str):
    """Load, clean, harmonize, label, and structure the temporal dataset.

    This returns split-ready DataFrames plus trustworthy labels. Model-specific
    encoding, imputation, and scaling should happen after patient-level split.
    """
    p = _resolve_source_csv(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(p)
    df.columns = [pp.standardize_column_name(c) for c in df.columns]
    df = _filter_observed_outcomes(df)
    df = pp.clean_and_coerce_types(df)
    df = pp.harmonize_categoricals(df)
    df = pp.drop_uninformative_columns(df)

    y, valid_mask = _build_label_array(df)
    df = df.loc[valid_mask].reset_index(drop=True)
    outcome_labels = df["outcome"].astype(str).to_numpy()

    df_static, df_temporal = pp.structure_temporal_data(df)
    df_static = df_static.reset_index(drop=True)
    df_temporal = df_temporal.sort_values(["patient_id", "month"]).reset_index(drop=True)

    return {
        "df": df,
        "df_static": df_static,
        "df_temporal": df_temporal,
        "y": y,
        "outcome_labels": outcome_labels,
        "patient_ids": df_static["patient_id"].to_numpy(dtype=np.int64),
        "label_summary": {
            "n_patients": int(len(df_static)),
            "n_success": int(y.sum()),
            "n_failure": int(len(y) - y.sum()),
        },
    }


def build_split_model_inputs(
    df_static: pd.DataFrame,
    df_temporal: pd.DataFrame,
    train_idx,
    val_idx=None,
    test_idx=None,
    drop_feature_cols=None,
    random_state: int = 42,
):
    """Fit train-only imputers/encoders and return full arrays in patient order."""
    if val_idx is None:
        val_idx = np.array([], dtype=int)
    if test_idx is None:
        test_idx = np.array([], dtype=int)
    drop_feature_cols = set(TRAINING_DROP_COLUMNS if drop_feature_cols is None else drop_feature_cols)

    static_features = df_static.drop(columns=["outcome"], errors="ignore").copy()
    applied_drop_feature_cols = sorted(c for c in drop_feature_cols if c in static_features.columns)
    static_features = static_features.drop(columns=applied_drop_feature_cols)
    static_features = _convert_datetime_columns_to_numeric(static_features)
    bool_cols = static_features.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        static_features[col] = static_features[col].astype(np.int8)
    static_features["__row_pos"] = np.arange(len(static_features))

    split_indices = {
        "train": np.asarray(train_idx, dtype=int),
        "val": np.asarray(val_idx, dtype=int),
        "test": np.asarray(test_idx, dtype=int),
    }
    static_splits = {name: static_features.iloc[idx].copy() for name, idx in split_indices.items()}
    train_static = static_splits["train"]
    static_missing_cols = [
        c for c in train_static.columns
        if c not in {"patient_id", "__row_pos"} and train_static[c].isna().any()
    ]
    static_missing_indicator_names = [f"{c}__missing" for c in static_missing_cols]

    static_cat_cols = [
        c for c in train_static.select_dtypes(include=["object", "category"]).columns.tolist()
        if c not in {"patient_id", "__row_pos"}
    ]
    static_num_cols = [
        c for c in train_static.columns
        if c not in set(static_cat_cols) | {"__row_pos"}
        and pd.api.types.is_numeric_dtype(train_static[c])
    ]
    static_num_impute_cols = [c for c in static_num_cols if train_static[c].notna().any()]
    static_num_all_missing = [c for c in static_num_cols if c not in static_num_impute_cols]

    static_num_imputer = IterativeImputer(max_iter=20, random_state=random_state, sample_posterior=False)
    if static_num_impute_cols:
        static_num_imputer.fit(train_static[static_num_impute_cols])

    static_split_frames = {}
    train_dummy_cols = None
    for name, split_df in static_splits.items():
        split_out = split_df[["__row_pos"]].copy()
        for col in static_missing_cols:
            split_out[f"{col}__missing"] = split_df[col].isna().astype(np.float32)
        if static_num_impute_cols:
            transformed = static_num_imputer.transform(split_df[static_num_impute_cols])
            split_out = pd.concat(
                [split_out, pd.DataFrame(transformed, columns=static_num_impute_cols, index=split_df.index)],
                axis=1,
            )
        for col in static_num_all_missing:
            split_out[col] = 0.0
        if static_cat_cols:
            cat_filled = _fill_static_categoricals(train_static, split_df[static_cat_cols], static_cat_cols)
            cat_encoded = pd.get_dummies(cat_filled, columns=static_cat_cols, dummy_na=False, dtype=np.float32)
            if name == "train":
                train_dummy_cols = cat_encoded.columns.tolist()
            else:
                cat_encoded = cat_encoded.reindex(columns=train_dummy_cols, fill_value=0.0)
            split_out = pd.concat([split_out, cat_encoded], axis=1)
        static_split_frames[name] = split_out

    df_static_model = pd.concat(static_split_frames.values(), axis=0).sort_values("__row_pos").drop(columns=["__row_pos"])
    if "patient_id" in df_static_model.columns:
        df_static_model["patient_id"] = df_static_model["patient_id"].round().astype(np.int64)

    train_patient_ids = static_features.iloc[split_indices["train"]]["patient_id"].tolist()
    val_patient_ids = static_features.iloc[split_indices["val"]]["patient_id"].tolist()
    test_patient_ids = static_features.iloc[split_indices["test"]]["patient_id"].tolist()
    temporal_splits = {
        "train": df_temporal[df_temporal["patient_id"].isin(train_patient_ids)].copy(),
        "val": df_temporal[df_temporal["patient_id"].isin(val_patient_ids)].copy(),
        "test": df_temporal[df_temporal["patient_id"].isin(test_patient_ids)].copy(),
    }

    train_temporal = temporal_splits["train"].sort_values(["patient_id", "month"]).copy()
    temporal_cat_cols = [
        c for c in train_temporal.select_dtypes(include=["object", "category"]).columns.tolist()
        if c not in {"patient_id", "month"}
    ]
    temporal_num_cols = [
        c for c in train_temporal.columns
        if c not in set(temporal_cat_cols) | {"patient_id", "month"}
        and pd.api.types.is_numeric_dtype(train_temporal[c])
    ]
    temporal_missing_cols = [
        c for c in temporal_num_cols + temporal_cat_cols
        if train_temporal[c].isna().any()
    ]
    temporal_missing_indicator_names = [f"{c}__missing" for c in temporal_missing_cols]
    temporal_num_impute_cols = [c for c in temporal_num_cols if train_temporal[c].notna().any()]
    temporal_num_all_missing = [c for c in temporal_num_cols if c not in temporal_num_impute_cols]

    train_temporal_num = train_temporal[["month"] + temporal_num_impute_cols].copy()
    for col in temporal_num_impute_cols:
        train_temporal_num[col] = train_temporal.groupby("patient_id")[col].ffill()
    temporal_imputer = IterativeImputer(max_iter=20, random_state=random_state, sample_posterior=False)
    if temporal_num_impute_cols:
        temporal_imputer.fit(train_temporal_num[["month"] + temporal_num_impute_cols])

    processed_temporal_splits = {}
    for name, split_df in temporal_splits.items():
        split_sorted = split_df.sort_values(["patient_id", "month"]).copy()
        processed = split_sorted[["patient_id", "month"]].copy()
        for col in temporal_missing_cols:
            processed[f"{col}__missing"] = split_sorted[col].isna().astype(np.float32)
        if temporal_num_impute_cols:
            num_frame = split_sorted[["month"] + temporal_num_impute_cols].copy()
            for col in temporal_num_impute_cols:
                num_frame[col] = split_sorted.groupby("patient_id")[col].ffill()
            num_frame[["month"] + temporal_num_impute_cols] = temporal_imputer.transform(num_frame[["month"] + temporal_num_impute_cols])
            processed[temporal_num_impute_cols] = num_frame[temporal_num_impute_cols]
        for col in temporal_num_all_missing:
            processed[col] = 0.0
        if temporal_cat_cols:
            cat_frame = _fill_temporal_categoricals(train_temporal, split_sorted[temporal_cat_cols], temporal_cat_cols)
            for col in temporal_cat_cols:
                processed[col] = cat_frame[col]
        processed["month"] = processed["month"].round().astype(int)
        processed_temporal_splits[name] = processed

    df_temporal_model = pd.concat(processed_temporal_splits.values(), axis=0).sort_values(["patient_id", "month"]).reset_index(drop=True)
    df_static_model, df_temporal_model = pp._post_imputation_clinical_clipping(df_static_model, df_temporal_model)
    arrays = _build_model_arrays(df_static_model, df_temporal_model)
    arrays.update({
        "feature_policy_version": TEMPORAL_V2_FEATURE_POLICY,
        "dropped_feature_cols": applied_drop_feature_cols,
        "static_missing_indicator_names": static_missing_indicator_names,
        "temporal_missing_indicator_names": temporal_missing_indicator_names,
    })
    return arrays


def patient_level_split(n_patients: int, y=None, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state: int = 42):
    """Return (train_idx, val_idx, test_idx) arrays of patient indices.

    Indices are integers in [0, n_patients).
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Fractions must sum to 1.0")

    all_idx = np.arange(n_patients)
    stratify = np.asarray(y) if y is not None else None

    train_idx, rest_idx = train_test_split(
        all_idx,
        test_size=(1.0 - train_frac),
        random_state=random_state,
        stratify=stratify,
    )

    rest_stratify = stratify[rest_idx] if stratify is not None else None
    val_share = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=(1.0 - val_share),
        random_state=random_state,
        stratify=rest_stratify,
    )
    return train_idx, val_idx, test_idx


def prepare_default_model_inputs(
    csv_path: str,
    train_frac=0.7,
    val_frac=0.2,
    test_frac=0.1,
    random_state: int = 42,
    drop_feature_cols=None,
):
    """Load the temporal dataset, create trustworthy labels, split patients,
    and build train-fit model arrays in one call.
    """
    data = load_cleaned_csv(csv_path)
    train_idx, val_idx, test_idx = patient_level_split(
        len(data["y"]),
        y=data["y"],
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    arrays = build_split_model_inputs(
        data["df_static"],
        data["df_temporal"],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        drop_feature_cols=drop_feature_cols,
        random_state=random_state,
    )
    data.update(arrays)
    data["train_idx"] = train_idx
    data["val_idx"] = val_idx
    data["test_idx"] = test_idx
    return data


def scale_train_val_test(X_static, X_temporal, train_idx, val_idx, test_idx):
    """Fit scalers on training set only and transform all splits.

    - X_static: (n_patients, n_static_features)
    - X_temporal: (n_patients, n_timesteps, n_temporal_features)

    Returns dict with scaled arrays and fitted scalers.
    """
    # Static scaler
    scaler_static = StandardScaler()
    Xs_train = X_static[train_idx]
    scaler_static.fit(Xs_train)
    X_static_scaled = dict()
    X_static_scaled["train"] = scaler_static.transform(X_static[train_idx])
    X_static_scaled["val"] = scaler_static.transform(X_static[val_idx]) if len(val_idx) > 0 else np.empty((0, X_static.shape[1]))
    X_static_scaled["test"] = scaler_static.transform(X_static[test_idx]) if len(test_idx) > 0 else np.empty((0, X_static.shape[1]))

    # Temporal scaler: fit on reshaped train data (stacked timesteps)
    n_timesteps = X_temporal.shape[1]
    n_temp_feats = X_temporal.shape[2]
    Xtemp_train_2d = X_temporal[train_idx].reshape(-1, n_temp_feats)
    scaler_temporal = StandardScaler()
    scaler_temporal.fit(Xtemp_train_2d)

    def transform_temporal(split_idx):
        if len(split_idx) == 0:
            return np.empty((0, n_timesteps, n_temp_feats))
        arr = X_temporal[split_idx]
        arr2 = arr.reshape(-1, n_temp_feats)
        arr2_t = scaler_temporal.transform(arr2)
        return arr2_t.reshape(len(split_idx), n_timesteps, n_temp_feats)

    X_temporal_scaled = dict()
    X_temporal_scaled["train"] = transform_temporal(train_idx)
    X_temporal_scaled["val"] = transform_temporal(val_idx)
    X_temporal_scaled["test"] = transform_temporal(test_idx)

    return {
        "X_static_scaled": X_static_scaled,
        "X_temporal_scaled": X_temporal_scaled,
        "scaler_static": scaler_static,
        "scaler_temporal": scaler_temporal,
    }


def scale_full_arrays(X_static, X_temporal, train_idx):
    """Fit on train indices only and return full-length scaled arrays.

    The returned arrays preserve the original patient order, so existing
    notebook code that indexes by `train_idx`, `val_idx`, `test_idx`
    continues to work unchanged.
    """
    scaler_static = StandardScaler()
    scaler_static.fit(X_static[train_idx])
    X_static_scaled = scaler_static.transform(X_static)

    n_timesteps = X_temporal.shape[1]
    n_temp_feats = X_temporal.shape[2]
    scaler_temporal = StandardScaler()
    scaler_temporal.fit(X_temporal[train_idx].reshape(-1, n_temp_feats))
    X_temporal_scaled = scaler_temporal.transform(
        X_temporal.reshape(-1, n_temp_feats)
    ).reshape(X_temporal.shape[0], n_timesteps, n_temp_feats)

    return {
        "X_static_scaled": X_static_scaled,
        "X_temporal_scaled": X_temporal_scaled,
        "scaler_static": scaler_static,
        "scaler_temporal": scaler_temporal,
    }


def build_features_at_month(X_temporal, X_static, up_to_month, temporal_names, static_names):
    """
    Build a flat feature matrix using data up to `up_to_month`.

    Features include:
    1. Static baseline features (always the same)
    2. Flattened raw temporal features up to month t
    3. Aggregate stats (mean, std, min, max) over available months
    4. Trend (linear slope) of key features
    5. Latest month's values
    6. Month indicator (how much data is available)

    Returns: X_flat (n_patients, n_features), feature_name_list
    """
    n_patients = X_temporal.shape[0]
    n_temporal_feats = X_temporal.shape[2]  # 8
    valid_months = up_to_month + 1

    feature_list = []
    feature_name_list = []

    # 1. Static features (always included)
    feature_list.append(X_static)
    feature_name_list.extend(static_names)

    # 2. Flattened temporal features for months 0..up_to_month
    temporal_flat = X_temporal[:, :valid_months, :].reshape(n_patients, -1)
    for m in range(valid_months):
        for feat in temporal_names:
            feature_name_list.append(f"M{m}_{feat}")
    feature_list.append(temporal_flat)

    # 3. Aggregate statistics over available months
    temporal_slice = X_temporal[:, :valid_months, :]  # (N, valid_months, 8)
    for agg_name, agg_fn in [("mean", np.mean), ("std", np.std),
                              ("min", np.min), ("max", np.max)]:
        agg = agg_fn(temporal_slice, axis=1)  # (N, 8)
        feature_list.append(agg)
        for feat in temporal_names:
            feature_name_list.append(f"{agg_name}_{feat}")

    # 4. Trend features (linear slope) for key temporal features
    if valid_months >= 2:
        slopes = np.zeros((n_patients, n_temporal_feats))
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
        feature_list.append(np.zeros((n_patients, n_temporal_feats)))
        for feat in temporal_names:
            feature_name_list.append(f"trend_{feat}")

    # 5. Latest month's values
    latest = X_temporal[:, up_to_month, :]  # (N, 8)
    feature_list.append(latest)
    for feat in temporal_names:
        feature_name_list.append(f"latest_{feat}")

    # 6. Month indicator
    month_indicator = np.full((n_patients, 1), up_to_month, dtype=np.float32)
    feature_list.append(month_indicator)
    feature_name_list.append("months_available")

    X_flat = np.hstack(feature_list).astype(np.float32)
    X_flat = np.nan_to_num(X_flat, nan=0.0)

    return X_flat, feature_name_list


def evaluate_binary_predictions(y_true, y_prob, threshold=0.5):
    """Return standard binary metrics for a probability threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=int),
        "threshold": float(threshold),
    }


def find_optimal_threshold(
    y_true,
    y_prob,
    thresholds=None,
    objective="balanced",
    min_precision=0.0,
):
    """Search for a probability cutoff that matches the clinical tradeoff.

    Parameters
    ----------
    y_true : array-like
        True binary labels where 1 is the positive class.
    y_prob : array-like
        Predicted probabilities for the positive class.
    thresholds : iterable, optional
        Candidate thresholds to evaluate. Defaults to 0.10..0.90.
    objective : str
        - "balanced": favor a mix of F1, recall, and specificity.
        - "failure_recall": favor detecting failures while keeping success recall.
        - "specificity": favor reducing false positives.
        - "f1": pure F1 search.
        - "accuracy": choose the threshold with the highest accuracy.
    min_precision : float
        Reject thresholds with precision below this floor.

    Returns
    -------
    dict
        Best threshold, score, metrics, and a dataframe of all candidates.
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    rows = []
    best_row = None
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        failure_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        success_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = failure_recall
        balanced_accuracy = (failure_recall + success_recall) / 2.0

        if precision < min_precision:
            score = -np.inf
        elif objective == "failure_recall":
            score = (0.55 * failure_recall) + (0.25 * balanced_accuracy) + (0.20 * f1)
        elif objective == "specificity":
            score = (0.50 * specificity) + (0.30 * f1) + (0.20 * recall)
        elif objective == "f1":
            score = f1
        elif objective == "accuracy":
            score = accuracy
        else:
            score = (0.40 * f1) + (0.30 * recall) + (0.30 * specificity)

        row = {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "failure_recall": float(failure_recall),
            "success_recall": float(success_recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_accuracy),
            "score": float(score),
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best_row = row

    results = pd.DataFrame(rows)
    if best_row is None:
        best_row = results.sort_values(["f1", "specificity", "recall"], ascending=False).iloc[0].to_dict()

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_score": float(best_row["score"]),
        "best_metrics": best_row,
        "search_results": results,
    }


def save_scalers(scaler_static, scaler_temporal, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    s1 = out / "scaler_static.pkl"
    s2 = out / "scaler_temporal.pkl"
    joblib.dump(scaler_static, s1)
    joblib.dump(scaler_temporal, s2)
    return str(s1), str(s2)


def export_tree_model_to_onnx(
    fitted_pipeline,
    feature_names,
    model_name,
    output_dir,
    X_dummy_sample=None,
):
    """Export a fitted tree-based model to ONNX and ORT formats.
    
    Supports LightGBM, XGBoost, and scikit-learn Random Forest models.
    Uses model-specific converters for best compatibility.
    
    Parameters
    ----------
    fitted_pipeline : LGBMClassifier, XGBClassifier, RandomForestClassifier, or Pipeline
        A fitted tree-based model or pipeline.
    feature_names : list of str
        List of input feature names for the model.
    model_name : str
        Name for the exported model (e.g., "lightgbm_temporal").
    output_dir : str or Path
        Directory to save ONNX and ORT model files.
    X_dummy_sample : ndarray, optional
        Dummy input array for ONNX export. If None, uses a random array.
        Expected shape: (1, n_features) for a single sample.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - "onnx_path": Path to saved .onnx file
        - "ort_path": Path to saved .ort file
        - "feature_names": Feature names used during export
        - "model_name": Model name used during export
    """
    import shutil
    import numpy as np
    import onnxruntime as ort
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input if not provided
    if X_dummy_sample is None:
        X_dummy_sample = np.random.randn(1, len(feature_names)).astype(np.float32)
    else:
        X_dummy_sample = np.asarray(X_dummy_sample, dtype=np.float32)
        if X_dummy_sample.shape[0] != 1:
            X_dummy_sample = X_dummy_sample[:1]
    
    onnx_path = output_dir / f"{model_name}.onnx"
    ort_path = output_dir / f"{model_name}.ort"
    onnx_model = None
    
    # Detect model type and use appropriate converter
    model_type_name = type(fitted_pipeline).__name__
    model_type_module = type(fitted_pipeline).__module__
    
    is_lightgbm = ("LGBM" in model_type_name) or ("lightgbm" in model_type_module.lower())
    is_xgboost = ("XGB" in model_type_name) or ("xgboost" in model_type_module.lower())
    is_random_forest = ("RandomForest" in model_type_name) or ("sklearn.ensemble" in model_type_module)
    
    print(f"Converting {model_name} to ONNX (detected: {model_type_name} from {model_type_module})...")
    
    # ── LightGBM Export ──────────────────────────────────────────
    if is_lightgbm:
        try:
            import onnxmltools
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            print(f"  Using onnxmltools for LightGBM conversion...")
            initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
            
            onnx_model = onnxmltools.convert_lightgbm(
                fitted_pipeline,
                initial_types=initial_type,
                target_opset=12,
            )
        except Exception as e:
            print(f"  ⚠ onnxmltools approach failed: {type(e).__name__}: {str(e)[:100]}")
            print(f"  Attempting fallback with skl2onnx...")
            
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                
                initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
                onnx_model = convert_sklearn(fitted_pipeline, initial_types=initial_type, target_opset=12)
            except Exception as e2:
                print(f"  ✗ Both LightGBM converters failed. Saving PyTorch checkpoint instead...")
                checkpoint_path = output_dir / f"{model_name}_checkpoint.joblib"
                joblib.dump(fitted_pipeline, checkpoint_path)
                print(f"  ✓ Model checkpoint saved: {checkpoint_path.name}")
                print(f"  Note: Use joblib.load() for inference instead of ONNX Runtime")
                return {
                    "onnx_path": None,
                    "ort_path": None,
                    "feature_names": feature_names,
                    "model_name": model_name,
                    "checkpoint_path": str(checkpoint_path),
                }
    
    # ── XGBoost (convert via binary with proper handling) ─────────
    elif is_xgboost:
        try:
            import xgboost as xgb_module
            import onnxmltools
            from skl2onnx.common.data_types import FloatTensorType
            
            print(f"  Using onnxmltools for {model_type_name} conversion...")
            
            # Step 1: Save to binary format, then reload as pure Booster
            import tempfile
            import os
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_model_path = os.path.join(tmpdir, "temp_model.bin")
                
                # Save classifier's underlying model to binary
                if hasattr(fitted_pipeline, 'get_booster'):
                    fitted_pipeline.get_booster().save_model(temp_model_path)
                else:
                    fitted_pipeline.save_model(temp_model_path)
                
                # Reload as pure Booster from binary
                booster = xgb_module.Booster(model_file=temp_model_path)
                
                # Now convert the Booster - onnxmltools should handle Booster objects
                initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
                onnx_model = onnxmltools.convert_xgboost(
                    booster,
                    initial_types=initial_type,
                    target_opset=12
                )
        except Exception as e:
            print(f"  ⚠ onnxmltools failed: {type(e).__name__}: {str(e)[:100]}")
            print(f"  Attempting alternative: use sklearn wrapper directly on dummy data...")
            
            try:
                # Try wrapping the model to make it sklearn-compatible
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                
                # Test if we can convert the classifier directly with better type hints
                initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
                
                # Create a simple wrapper to avoid sklearn-specific issues
                class XGBONNXWrapper:
                    def __init__(self, model):
                        self.model = model
                    def predict(self, X):
                        return self.model.predict(X)
                    def predict_proba(self, X):
                        return self.model.predict_proba(X)
                
                wrapper = XGBONNXWrapper(fitted_pipeline)
                onnx_model = convert_sklearn(wrapper, initial_types=initial_type, target_opset=12)
            except Exception as e2:
                print(f"  ⚠ Wrapper approach failed: {type(e2).__name__}: {str(e2)[:100]}")
                print(f"  Saving checkpoint instead...")
                checkpoint_path = output_dir / f"{model_name}_checkpoint.joblib"
                joblib.dump(fitted_pipeline, checkpoint_path)
                print(f"  ✓ Model checkpoint saved: {checkpoint_path.name}")
                print(f"  Note: Use joblib.load() for inference instead of ONNX Runtime")
                return {
                    "onnx_path": None,
                    "ort_path": None,
                    "feature_names": feature_names,
                    "model_name": model_name,
                    "checkpoint_path": str(checkpoint_path),
                }
    
    # ── Random Forest (skl2onnx) ────────────────────────────────────
    elif is_random_forest:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            print(f"  Using skl2onnx for {model_type_name} conversion...")
            initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
            
            onnx_model = convert_sklearn(
                fitted_pipeline,
                initial_types=initial_type,
                target_opset=12,
            )
        except Exception as e:
            print(f"  ⚠ skl2onnx conversion failed: {type(e).__name__}: {str(e)[:100]}")
            print(f"  Saving model checkpoint for Python inference...")
            checkpoint_path = output_dir / f"{model_name}_checkpoint.joblib"
            joblib.dump(fitted_pipeline, checkpoint_path)
            print(f"  ✓ Model checkpoint saved: {checkpoint_path.name}")
            print(f"  Note: Use joblib.load() for inference instead of ONNX Runtime")
            return {
                "onnx_path": None,
                "ort_path": None,
                "feature_names": feature_names,
                "model_name": model_name,
                "checkpoint_path": str(checkpoint_path),
            }
    
    else:
        print(f"  ⚠ Unknown model type: {model_type_name}")
        print(f"  Saving model checkpoint instead...")
        checkpoint_path = output_dir / f"{model_name}_checkpoint.joblib"
        joblib.dump(fitted_pipeline, checkpoint_path)
        return {
            "onnx_path": None,
            "ort_path": None,
            "feature_names": feature_names,
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path),
        }
    
    # ── Save ONNX ────────────────────────────────────────────────
    if onnx_model is not None:
        try:
            with open(str(onnx_path), "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"✓ ONNX model saved: {onnx_path.name}")
        except Exception as e:
            print(f"✗ Failed to save ONNX: {e}")
            raise
        
        # ── Verify ONNX ──────────────────────────────────────────
        print(f"Verifying ONNX model...")
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            ort_inputs = {sess.get_inputs()[0].name: X_dummy_sample}
            ort_outputs = sess.run(None, ort_inputs)
            print(f"✓ ONNX model verified (output shape: {ort_outputs[0].shape})")
        except Exception as e:
            print(f"⚠ ONNX verification failed: {e}")
            raise
        
        # ── Save ORT ─────────────────────────────────────────────
        try:
            shutil.copy(str(onnx_path), str(ort_path))
            print(f"✓ ORT model saved: {ort_path.name}")
        except Exception as e:
            print(f"⚠ ORT save failed: {e}")
    
    return {
        "onnx_path": str(onnx_path) if onnx_model is not None else None,
        "ort_path": str(ort_path) if onnx_model is not None else None,
        "feature_names": feature_names,
        "model_name": model_name,
    }
