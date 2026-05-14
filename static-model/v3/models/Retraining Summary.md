# Non-Temporal v3 — Model Pipeline Summary

## Overview

Three independent classification pipelines built to predict TB treatment outcome at the point of registration (non-temporal, baseline snapshot). All models share identical preprocessing and differ only in classifier, hyperparameters, and feature importance method.

**Notebooks:**
- [`../logistic_regression_v3.ipynb`](../logistic_regression_v3.ipynb)
- [`../svm_v3.ipynb`](../svm_v3.ipynb)
- [`../xgboost_v3.ipynb`](../xgboost_v3.ipynb)

---

## Dataset

| Property | Value |
|---|---|
| Source | `dataset/temporal/output/cleaned_human_readable.csv` |
| Raw rows | 599 |
| Retained after outcome filtering | 597 (2 "Not Evaluated" excluded) |
| Success (0) | 536 — Cured + Treatment Completed |
| Failure (1) | 61 — Died + Lost to Follow-Up + Failed |
| Imbalance ratio | ~8.8 : 1 (Success : Failure) |

---

## Target Variable

```
outcome → treatment_outcome_binary
  Cured / Treatment Completed  →  0  (Success)
  Died / Lost to Follow-Up / Failed  →  1  (Failure)
```

Rows with `Not Evaluated` or `On Treatment` are excluded before modelling.

---

## Shared Preprocessing Pipeline

### Columns Dropped (207 total)

| Category | Count | Columns |
|---|---|---|
| Identifiers / target leakage | 5 | `no`, `patient_id`, `source_file`, `outcome`, `date_of_outcome` |
| Raw date columns | 8 | `date_of_birth`, `date_of_diagnosis`, `date_of_notification`, `treatment_start_date`, `intensive_phase_start_date`, `intensive_phase_end_date`, `continuation_phase_start_date`, `continuation_phase_end_date` |
| Raw duplicates | 2 | `height`, `weight` (superseded by `height_cm`, `weight_kg`) |
| M1–M12 temporal | 192 | All columns matching `^M([1-9]\|1[0-2])_` |

### Derived Features (from date columns before dropping)

| Feature | Description | Clip range |
|---|---|---|
| `diagnosis_delay_days` | `date_of_diagnosis` − `date_of_notification` | 0–365 days |
| `days_to_treatment` | `treatment_start_date` − `date_of_diagnosis` | 0–365 days |
| `intensive_phase_duration_days` | `intensive_phase_end_date` − `intensive_phase_start_date` | ≥ 0 |
| `continuation_phase_duration_days` | `continuation_phase_end_date` − `continuation_phase_start_date` | ≥ 0 |

### Final Feature Set

| Type | Count | Examples |
|---|---|---|
| Numeric | 50 | `age`, `weight_kg`, `height_cm`, `bp_systolic`, `bp_diastolic`, `heart_rate`, `o2_sat`, `data_year`, `diagnosis_delay_days`, `days_to_treatment`, all `M0_` continuous features, all `is_missing_*` flags |
| Categorical | 13 | `sex`, `civil_status`, `nationality`, `diagnosis`, `bacteriologic_status`, `case_registration_group`, `treatment_regimen`, `name_of_diagnosing_facility`, `name_of_treatment_unit`, etc. |

### Scaling & Encoding

Applied inside a `ColumnTransformer` **fit on training data only**, then applied to the test set:

- **Numeric** → `StandardScaler()`
- **Categorical** → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

No imputation is applied — the dataset has already been imputed upstream in `preprocessingV2.py`.

### Train / Test Split

| Set | Samples | Success (0) | Failure (1) |
|---|---|---|---|
| Train (80%) | 477 | 428 | 49 |
| Test (20%) | 120 | 108 | 12 |

Stratified by `treatment_outcome_binary`, `random_state=42`.

---

## Model Configurations

### Logistic Regression

```python
LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=42
)
```

- **Imbalance handling:** `class_weight='balanced'` (sklearn re-weights loss by inverse class frequency)
- **Feature importance:** Signed coefficients — positive values predict Failure (1)

### Support Vector Machine (RBF)

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=42
)
```

- **Imbalance handling:** `class_weight='balanced'`
- **Feature importance:** Permutation importance on the test set (15 repeats, scored by ROC-AUC) — RBF kernel has no direct coefficients

### XGBoost

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=8.73,   # (y_train==0).sum() / (y_train==1).sum()
    n_jobs=-1,
    eval_metric='logloss',
    random_state=42
)
```

- **Imbalance handling:** `scale_pos_weight=8.73` (majority-to-minority ratio computed from `y_train` only)
- **Feature importance:** Gain-based `feature_importances_` + XGBoost native `plot_importance`

---

## Results (Test Set, n=120)

| Model | Accuracy | ROC-AUC | Precision (Failure) | Recall (Failure) | F1 (Failure) |
|---|---|---|---|---|---|
| Logistic Regression | 0.7750 | 0.8573 | 0.26 | 0.67 | 0.37 |
| SVM (RBF) | 0.8417 | 0.8549 | 0.32 | 0.50 | 0.39 |
| **XGBoost** | **0.8917** | **0.9398** | **0.47** | **0.75** | **0.58** |

Primary metric is **ROC-AUC** and **Recall for Failure (1)** — identifying patients at risk of treatment failure is the clinical priority.

XGBoost achieves the best performance across all metrics. Logistic Regression achieves the highest Failure recall (0.67) at the cost of accuracy and precision.

---

## Notebook Structure (all three)

| Section | Description |
|---|---|
| 1. Load Data & Derive Target | Read CSV, filter to Success/Failure rows, encode `treatment_outcome_binary` |
| 2. Feature Engineering | Parse date columns, compute 4 duration features, then date columns dropped |
| 3. Drop Leakage Columns | Remove identifiers, date cols, raw dupes, M1–M12 temporal (207 cols total) |
| 4. Train / Test Split | 80/20 stratified split — performed before any scaling |
| 5. Preprocessing | `ColumnTransformer`: `StandardScaler` for numeric, `OneHotEncoder` for categorical, fit on train only |
| 6. Build & Train Model | Wrap preprocessor + classifier in `sklearn.Pipeline`, fit on `X_train` |
| 7. Evaluation | Accuracy, ROC-AUC, classification report, confusion matrix heatmap, ROC curve |
| 8. Feature Importance | LR: coefficients · SVM: permutation importance · XGBoost: gain + native plot |
