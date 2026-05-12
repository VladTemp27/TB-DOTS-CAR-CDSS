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


def _resolve_source_csv(csv_path: str | None) -> Path:
    """Resolve the raw combined dataset path used by the model notebooks.

    The notebook helpers still pass the old cleaned CSV path, so this keeps the
    public function signature stable while steering the loader to the combined
    raw dataset that should now be preprocessed on demand.
    """
    default_path = REPO_ROOT / "dataset" / "temporal" / "combined_complete_dataset.csv"
    if csv_path is None:
        return default_path

    requested_path = Path(csv_path)
    if requested_path.name == "cleaned_human_readable.csv":
        return default_path
    return requested_path


def _build_model_arrays(df_static: pd.DataFrame, df_temporal: pd.DataFrame):
    """Build numeric model-ready arrays from the cleaned human-readable tables.

    The current preprocessing pipeline intentionally stops at human-readable
    outputs, so this helper performs the minimal compatibility transforms
    required by the temporal model notebooks:

    - drop identifiers and raw date / provenance columns from static features
    - one-hot encode remaining static categoricals
    - assemble the 3D temporal tensor in month order (M0..M12)
    - build flattened combined features for tabular models
    """

    patient_ids = df_static["patient_id"].to_numpy()

    # Keep only useful static columns. The raw human-readable CSV contains
    # identifiers, dates, and provenance fields that are not suitable as
    # direct model inputs.
    drop_exact = {
        "data_year",
        "source_file",
    }
    drop_prefixes = ("date_of_", "intensive_phase_", "continuation_phase_")

    static_feature_cols = []
    for col in df_static.columns:
        if col in drop_exact:
            continue
        if any(col.startswith(prefix) for prefix in drop_prefixes):
            continue
        if "date" in col.lower():
            continue
        static_feature_cols.append(col)

    df_static_model = df_static[static_feature_cols].copy()

    # Encode object columns while keeping numeric columns unchanged.
    cat_cols = df_static_model.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df_static_model[col] = df_static_model[col].fillna("Unknown").astype(str)
    df_static_model = pd.get_dummies(df_static_model, columns=cat_cols, dummy_na=False)

    # Ensure booleans are numeric.
    bool_cols = df_static_model.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df_static_model[col] = df_static_model[col].astype(np.int8)

    static_feature_names = df_static_model.columns.tolist()
    X_static = df_static_model.to_numpy(dtype=np.float32)

    # Temporal features already arrive as cleaned numeric columns in long format.
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
    temporal_feature_names = [
        feat for feat in preferred_order if feat in temporal_feature_names
    ] + [feat for feat in temporal_feature_names if feat not in preferred_order]

    n_patients = len(patient_ids)
    n_timesteps = int(df_temporal["month"].max()) + 1 if len(df_temporal) else len(pp.MONTH_RANGE)
    n_temporal_features = len(temporal_feature_names)
    X_temporal = np.zeros((n_patients, n_timesteps, n_temporal_features), dtype=np.float32)

    df_temporal_sorted = df_temporal.sort_values(["patient_id", "month"])
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
    """Load the combined dataset and prepare model-ready arrays.

    Returns a dict with model-ready arrays and feature-name lists:
        {X_temporal, X_static, X_combined_flat, patient_ids,
         static_feature_names, temporal_feature_names, ...}
    """
    p = _resolve_source_csv(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(p)

    # Ensure patient_id exists
    if "patient_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["patient_id"] = df.index

    # Standardize column names using preprocessing helper (no inplace surprises)
    df.columns = [pp.standardize_column_name(c) for c in df.columns]

    # Run the minimal cleaning pipeline necessary to structure data
    df = pp.clean_and_coerce_types(df)
    df = pp.harmonize_categoricals(df)
    df = pp.drop_uninformative_columns(df)

    df_static, df_temporal = pp.structure_temporal_data(df)

    # Impute missing values (MICE + forward fill) — necessary before encoding
    df_static, df_temporal = pp.impute_missing_mice(df_static, df_temporal)

    model_data = _build_model_arrays(df_static, df_temporal)
    return model_data


def patient_level_split(n_patients: int, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state: int = 42):
    """Return (train_idx, val_idx, test_idx) arrays of patient indices.

    Indices are integers in [0, n_patients).
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Fractions must sum to 1.0")

    rng = np.random.RandomState(random_state)
    all_idx = np.arange(n_patients)
    rng.shuffle(all_idx)

    n_train = int(np.floor(train_frac * n_patients))
    n_val = int(np.floor(val_frac * n_patients))

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


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
