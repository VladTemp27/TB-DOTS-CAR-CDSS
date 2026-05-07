======================================================================
TB-DOTS CAR CDSS — Output File Descriptions
======================================================================

This folder contains all outputs produced by the
preprocessing pipeline (preprocessing_pipeline.py).
Below is a description of each file and its intended use.

----------------------------------------------------------------------
HUMAN-READABLE FILES
----------------------------------------------------------------------

cleaned_combined_dataset.csv
  The fully cleaned version of the original combined_dataset.csv.
  One row per patient, wide format (M0–M12 columns preserved).
  Contains human-readable categorical labels (e.g., 'Male',
  'Treatment Completed', 'Regimen 1') and numeric values in
  natural units (kg, cm, mmHg, %, etc.).
  Missing values have been imputed (MICE for numerics, mode
  for categoricals, forward-fill for temporal columns).
  NOT encoded or scaled — suitable for manual review, EDA,
  descriptive statistics, and research documentation.

static_features.csv
  Baseline (non-temporal) patient features after full
  preprocessing: imputed, one-hot encoded, and scaled.
  Categorical variables are split into binary indicator
  columns (e.g., sex_Male, sex_Female). Numeric values
  are standardized (zero mean, unit variance).
  Includes a 'patient_id' column linking to temporal data.

temporal_features.csv
  Monthly monitoring data in long format: one row per
  patient per month (patient_id × month). Contains
  treatment adherence, doses, weight, height, smear, and
  Xpert results — all imputed and scaled.
  Columns: patient_id, month (0–12), plus temporal features.

preprocessing_summary.txt
  A brief text summary of the pipeline execution, including
  dataset dimensions, tensor shapes, and lists of feature
  names used in static and temporal components.

OUTPUT_README.txt
  This file — describes every output file in the folder.

----------------------------------------------------------------------
MODEL-READY FILES (NumPy arrays)
----------------------------------------------------------------------

X_temporal.npy
  Shape: (599, 13, 16)
  3D NumPy array: (n_patients, n_timesteps, n_features).
  Each patient has 13 time steps (M0 through M12), each with
  16 features (doses, adherence, weight, height,
  smear result, Xpert result, etc.).
  Ready for direct input into RNN, LSTM, or GRU models.
  Usage: X = np.load('X_temporal.npy')

X_static.npy
  Shape: (599, 69)
  2D NumPy array: (n_patients, n_static_features).
  Contains baseline demographics, diagnostics, and clinical
  indicators — all encoded and scaled.
  Use for: hybrid models (e.g., concatenate with LSTM output),
  or standalone tabular models like XGBoost/Random Forest.
  Usage: X = np.load('X_static.npy')

X_combined_flat.npy
  Shape: (599, 277)
  2D NumPy array: static features + flattened temporal features
  concatenated side by side. Each patient is one row.
  Designed for tabular ML models (XGBoost, LightGBM, etc.)
  that don't natively handle 3D sequential input.
  Usage: X = np.load('X_combined_flat.npy')

patient_ids.npy
  Shape: (599,)
  1D array of integer patient IDs (0-indexed row numbers).
  Maps each row in X_temporal / X_static / X_combined_flat
  back to the corresponding patient.
  Usage: ids = np.load('patient_ids.npy')

feature_names.npz
  Compressed archive containing three string arrays:
    - 'static':   names of columns in X_static
    - 'temporal': names of columns in the flattened temporal
                  portion (M0_feature, M1_feature, ...)
    - 'combined': names of all columns in X_combined_flat
  Useful for model interpretability and feature importance.
  Usage: fn = np.load('feature_names.npz', allow_pickle=True)
         print(fn['static'])  # static feature names

----------------------------------------------------------------------
NOTES
----------------------------------------------------------------------

- All .npy files can be loaded with: np.load('filename.npy')
- All .csv files can be opened in Excel, Google Sheets, or
  loaded with: pd.read_csv('filename.csv')
- The cleaned_combined_dataset.csv is best for reviewing
  the data manually or generating summary tables.
- The .npy files are best for feeding directly into ML models.
