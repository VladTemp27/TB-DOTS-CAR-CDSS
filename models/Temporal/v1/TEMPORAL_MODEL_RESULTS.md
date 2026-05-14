# TB-DOTS CAR CDSS: Temporal Model Results Summary

## Overview

Four temporal models were developed for predicting TB treatment outcomes (Success vs Failure) using monthly patient monitoring data and baseline clinical features from 205 patients in the Cordillera Administrative Region (CAR). All models share the same 70/20/10 patient-level stratified split (seed=42), progressive temporal training (13 samples per patient, M0 to M12), and threshold optimization on the validation set (maximizing 0.6 x F1 + 0.4 x Specificity).

**Binary Labels:**
- **Success (1):** Cured, Treatment Completed (~90%)
- **Failure (0):** Died, Lost to Follow-Up, Not Evaluated (~10%)

**Dataset Split (Patient-Level, Stratified):**

| Split | Patients | Purpose |
|-------|----------|---------|
| Train | 143 (70%) | Model training with progressive expansion (143 x 13 = 1,859 samples) |
| Validation | 41 (20%) | Threshold optimization, early stopping |
| Test | 21 (10%) | Final held-out evaluation (19 Success, 2 Failure) |

---

## Model Configurations

### 1. Hybrid Bi-LSTM (Baseline and Augmented)

| Parameter | Value |
|-----------|-------|
| Architecture | Bidirectional LSTM + Masked Attention + Static FC |
| LSTM Hidden Units | 64 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Temporal Features | 8 |
| Static Features | 51 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |
| Epochs | 150 (early stopping, patience=20) |

**Baseline variant:** Weighted BCEWithLogitsLoss, no data augmentation, WeightedRandomSampler. Threshold = 0.15.

**Augmented variant:** Focal Loss (alpha=0.75, gamma=2.0), Gaussian noise injection (5 copies, std=0.05), Mixup (Beta 0.4), WeightedRandomSampler. Threshold = 0.86.

### 2. XGBoost (Baseline and SMOTE-ENN)

| Parameter | Value |
|-----------|-------|
| n_estimators | 300 (with early stopping, patience=20) |
| max_depth | 4 |
| learning_rate | 0.05 |
| min_child_weight | 3 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha / reg_lambda | 0.1 / 1.0 |
| Features | 204 |
| scale_pos_weight | Auto-calculated from class ratio |

**Baseline variant:** No SMOTE-ENN, native `scale_pos_weight` only. Threshold = 0.10.

**SMOTE-ENN variant:** SMOTE synthetic oversampling + ENN cleaning applied on progressive training data. Threshold = 0.34.

### 3. LightGBM (SMOTE-ENN)

| Parameter | Value |
|-----------|-------|
| n_estimators | 300 (early stopping at iteration 60) |
| max_depth | 4 |
| learning_rate | 0.05 |
| num_leaves | 31 |
| min_child_samples | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha / reg_lambda | 0.1 / 1.0 |
| is_unbalance | True |
| Features | 204 |
| Threshold | 0.43 |

Augmented with SMOTE-ENN on progressive training data.

### 4. Random Forest (SMOTE-ENN)

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| max_depth | 8 |
| min_samples_split | 5 |
| min_samples_leaf | 3 |
| max_features | sqrt |
| class_weight | balanced |
| oob_score | 0.9979 |
| Features | 204 |
| Threshold | 0.70 |

Augmented with SMOTE-ENN on progressive training data.

---

## Training Data Distribution: Before and After Augmentation

The original training split contains **143 patients** (131 Success, 12 Failure), reflecting the real-world class imbalance of ~8.4% Failure. Progressive temporal expansion generates **13 samples per patient** (M0 through M12), yielding **1,859 training samples** (1,703 Success, 156 Failure). Each model variant handles this imbalance differently.

### Original Training Data (All Models)

| Metric | Value |
|--------|-------|
| Training Patients | 143 |
| Success Patients | 131 (91.6%) |
| Failure Patients | 12 (8.4%) |
| Progressive Samples (x13) | 1,859 |
| Success Samples | 1,703 |
| Failure Samples | 156 |

### Bi-LSTM Baseline (No Augmentation)

| Metric | Value |
|--------|-------|
| Training Patients | 143 |
| Progressive Samples | 1,859 |
| Success / Failure Samples | 1,703 / 156 |
| Imbalance Handling | WeightedRandomSampler (balanced batches) |

No synthetic data is added. The model relies on a weighted sampler to balance class exposure during training.

### Bi-LSTM Augmented (Gaussian Noise + Mixup)

Augmentation is applied at the **patient level** before progressive expansion.

| Metric | Value |
|--------|-------|
| Original Failure Patients | 12 |
| Gaussian Noise Copies (5x) | 60 |
| Mixup Pairs | 24 |
| Total Synthetic Patients | 84 |
| **Final Training Patients** | **227** (131 Success, 96 Failure) |
| Failure Ratio | 42.3% (was 8.4%) |
| **Progressive Samples (x13)** | **2,951** |
| Imbalance Handling | WeightedRandomSampler (balanced batches) |

### XGBoost Baseline (No Augmentation)

| Metric | Value |
|--------|-------|
| Progressive Samples | 1,859 |
| Success / Failure Samples | 1,703 / 156 |
| Imbalance Handling | scale_pos_weight = 0.09 |

No resampling is applied. The model uses a low `scale_pos_weight` to adjust for class imbalance internally.

### Tree Models with SMOTE-ENN (XGBoost, LightGBM, Random Forest)

SMOTE-ENN is applied to the **progressive training samples** (after temporal expansion). SMOTE generates synthetic minority (Failure) samples, then ENN removes noisy/borderline majority samples.

| Metric | Before SMOTE-ENN | After SMOTE-ENN |
|--------|-------------------|-----------------|
| Total Samples | 1,859 | 2,874 |
| Success Samples | 1,703 | 1,193 |
| Failure Samples | 156 | 1,681 |
| Failure Ratio | 8.4% | 58.5% |

SMOTE-ENN reduced the majority class by 510 samples (ENN cleaning) while increasing the minority class by 1,525 samples (SMOTE oversampling), resulting in a roughly balanced dataset. XGBoost SMOTE-ENN uses `scale_pos_weight = 1.41`; LightGBM uses `scale_pos_weight = 0.71`; Random Forest uses `class_weight = balanced`.

---

## Test Set Evaluation Results

All results are from the held-out 10% test set (21 patients: 19 Success, 2 Failure) using full M0-M12 data with optimized thresholds.

### Primary Metrics Comparison

| Metric | Bi-LSTM BL | Bi-LSTM Aug | XGBoost BL | XGBoost SE | LightGBM SE | Random Forest SE |
|--------|-----------|-------------|-----------|-----------|------------|-----------------|
| **Accuracy** | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | **0.7143** |
| **Precision** | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.8824 |
| **Recall** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7895 |
| **F1 Score** | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.8333 |
| **Specificity** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **ROC-AUC** | 0.4737 | 0.3684 | **0.5789** | 0.1053 | 0.2105 | 0.5263 |
| **PR-AUC** | 0.8847 | 0.9099 | **0.9501** | 0.8356 | 0.8500 | 0.9419 |
| Threshold | 0.15 | 0.86 | 0.10 | 0.34 | 0.43 | 0.70 |

**Confusion Matrix (all models except RF):**
```
              Predicted
              Failure  Success
Actual Failure    0       2
Actual Success    0      19
```

**Confusion Matrix (Random Forest):**
```
              Predicted
              Failure  Success
Actual Failure    0       2
Actual Success    4      15
```

> **Key Observations:**
> - Most models achieve 90.48% accuracy by predicting all patients as **Success** (reflecting the ~90/10 class imbalance).
> - **Specificity = 0.0** across all models means **no Failure cases were correctly identified** on the 21-patient test set.
> - The test set contains only **2 Failure patients**, making per-split evaluation highly sensitive to individual cases.
> - **XGBoost Baseline** achieved the highest ROC-AUC (0.5789) and PR-AUC (0.9501), suggesting better probability calibration despite identical hard predictions.
> - **Random Forest** was the only model to predict some patients as Failure (4 false negatives on Success), resulting in lower accuracy (0.7143) but no better specificity.

### 5-Fold Cross-Validation (Robustness Check)

Cross-validation uses M12 features across all 205 patients with native class weighting (no SMOTE-ENN during CV). This provides a more reliable estimate than the 21-patient test set.

| Metric | LightGBM | Random Forest |
|--------|----------|---------------|
| **Accuracy** | **0.9122 +/- 0.0249** | 0.9073 +/- 0.0284 |
| **F1** | **0.9530 +/- 0.0135** | 0.9508 +/- 0.0154 |
| **ROC-AUC** | **0.7292 +/- 0.1320** | 0.7250 +/- 0.1464 |
| **Precision** | **0.9336 +/- 0.0129** | 0.9248 +/- 0.0218 |
| **Recall** | 0.9734 +/- 0.0166 | **0.9787 +/- 0.0198** |

> LightGBM slightly outperforms Random Forest across most CV metrics. Both models show reasonable ROC-AUC (~0.73) in cross-validation, suggesting the poor test-set ROC-AUC is due to the very small test set (only 2 Failure patients).

---

## Feature Importance

### Feature Engineering (Tree-Based Models)

The 3D temporal data (N x 13 x 8) is transformed into 204 flat 2D features for tree-based models:

| Category | Description | Example Features |
|----------|-------------|-----------------|
| **Static (baseline)** | 51 encoded clinical features | age, sex, disease_classification, treatment_regimen, etc. |
| **Temporal (raw monthly)** | Flattened raw values for months 0 to 12 | M0_weight, M5_pct_adherence, M12_monthly_doses_taken |
| **Aggregates** | Mean, std, min, max over available months | mean_pct_adherence, std_weight, max_monthly_missed_doses |
| **Trends (slopes)** | Linear regression slope over time | trend_weight, trend_pct_adherence |
| **Latest month** | Most recent month's values | latest_weight, latest_monthly_doses_taken |
| **Month indicator** | How many months of data are available | months_available |

### 8 Temporal Features Tracked Monthly (M0-M12)

| Feature | Description |
|---------|-------------|
| cumulative_doses_taken | Total doses taken up to this month |
| height | Patient height |
| monthly_doses_taken | Doses taken this month |
| monthly_missed_doses | Doses missed this month |
| pct_adherence | Adherence percentage this month |
| smear_tb_lamp | Smear/TB-LAMP test result |
| weight | Patient weight |
| xpert_mtb_rif | Xpert MTB/RIF test result |

### Feature Importance by Category (Tree-Based Models)

Each tree-based model reports feature importance by category, showing the relative contribution of static baseline features vs temporal/engineered features. Top-25 feature importance plots are saved to:
- `models/xgb_feature_importance.png`
- `models/lgb_feature_importance.png`
- `models/rf_feature_importance.png`

### Bi-LSTM Attention Weights

The Hybrid Bi-LSTM uses **masked attention** over the temporal sequence to learn which treatment months are most important for the prediction. Unlike tree-based models that provide feature-level importance, the Bi-LSTM provides **temporal attention weights** (per month, M0-M12), indicating which months the model attends to most when making its prediction. The attention mechanism automatically masks out future months when predicting from partial sequences.

---

## Per-Month Prediction Performance (M0-M12)

All models support **progressive prediction** at any month during treatment. The per-month evaluation on the test set (21 patients) shows how prediction accuracy changes as more monthly data becomes available.

### Accuracy by Month

| Month | Bi-LSTM BL | Bi-LSTM Aug | XGBoost BL | XGBoost SE | LightGBM SE | Random Forest SE |
|-------|-----------|-------------|-----------|-----------|------------|-----------------|
| M0 | 0.7143 | 0.9048 | 0.9048 | 0.8571 | 0.8571 | 0.5238 |
| M1 | 0.8571 | 0.9048 | 0.9048 | 0.8571 | 0.8571 | 0.6667 |
| M2 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7619 |
| M3 | 0.9048 | 0.9048 | 0.9048 | 0.8571 | 0.9048 | 0.7143 |
| M4 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |
| M5 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |
| M6 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |
| M7 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |
| M8 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |
| M9 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |
| M10 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |
| M11 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |
| M12 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.7143 |

### ROC-AUC by Month

| Month | Bi-LSTM BL | Bi-LSTM Aug | XGBoost BL | XGBoost SE | LightGBM SE | Random Forest SE |
|-------|-----------|-------------|-----------|-----------|------------|-----------------|
| M0 | 0.6842 | 0.2105 | 0.3684 | 0.1842 | 0.3158 | 0.2368 |
| M1 | 0.5526 | 0.2105 | 0.4737 | 0.1842 | 0.3158 | 0.3158 |
| M2 | 0.5000 | 0.2632 | 0.4211 | 0.2105 | 0.2632 | 0.3158 |
| M3 | 0.5000 | 0.2632 | 0.4474 | 0.1053 | 0.2632 | 0.4211 |
| M4 | 0.5000 | 0.2895 | 0.4474 | 0.0789 | 0.1842 | 0.4211 |
| M5 | 0.4737 | 0.3421 | 0.4474 | 0.1316 | 0.3947 | 0.4211 |
| M6 | 0.4737 | 0.3684 | 0.5263 | 0.1053 | 0.2105 | 0.5000 |
| M7 | 0.4737 | 0.3684 | 0.5526 | 0.1053 | 0.2105 | 0.5000 |
| M8 | 0.4737 | 0.3684 | 0.5526 | 0.1053 | 0.2105 | 0.4737 |
| M9 | 0.4737 | 0.3684 | 0.5789 | 0.1053 | 0.2105 | 0.5526 |
| M10 | 0.4737 | 0.3684 | 0.5789 | 0.1053 | 0.2105 | 0.5263 |
| M11 | 0.4737 | 0.3684 | 0.5789 | 0.1053 | 0.2105 | 0.5263 |
| M12 | 0.4737 | 0.3684 | 0.5789 | 0.1053 | 0.2105 | 0.5263 |

> **Key Observations:**
> - **Specificity = 0.0** at every month for all models, meaning no model correctly identifies either Failure patient at any point during treatment.
> - Most models converge to the same accuracy (~0.9048) by M2, as they learn to predict all patients as Success.
> - **Bi-LSTM Baseline** shows initial variability (0.7143 at M0, rising to 0.9048 by M2), while the **Augmented** variant is already at 0.9048 from M0.
> - **Random Forest** is the only model with consistently different accuracy (0.7143 from M3 onward), as it predicts 4 Success patients as Failure.
> - **XGBoost Baseline** has the best ROC-AUC trend, gradually improving from 0.3684 to 0.5789.

> Per-month performance plots are saved to:
> - `models/per_month_performance.png` (Bi-LSTM)
> - `models/xgb_per_month_performance.png`
> - `models/lgb_per_month_performance.png`
> - `models/rf_per_month_performance.png`

---

## Class Imbalance Handling

| Model | Strategy |
|-------|----------|
| Bi-LSTM Baseline | Weighted BCEWithLogitsLoss + WeightedRandomSampler |
| Bi-LSTM Augmented | Gaussian noise (5 copies, std=0.05) + Mixup (Beta 0.4) + Focal Loss (alpha=0.75, gamma=2.0) + WeightedRandomSampler |
| XGBoost Baseline | `scale_pos_weight` (native class weighting) |
| XGBoost SMOTE-ENN | SMOTE + ENN cleaning + `scale_pos_weight` |
| LightGBM SMOTE-ENN | SMOTE + ENN cleaning + `is_unbalance=True` |
| Random Forest SMOTE-ENN | SMOTE + ENN cleaning + `class_weight='balanced'` |

---

## Model Artifacts

Each notebook saves the following to the `models/` directory:

| File | Model |
|------|-------|
| `baseline_model.pt` | Bi-LSTM Baseline weights |
| `best_model.pt` | Bi-LSTM Augmented weights |
| `model_config.json` | Bi-LSTM config |
| `lstm_metrics.json` | Bi-LSTM metrics (both variants) |
| `evaluation_report.txt` | Bi-LSTM evaluation report |
| `predictions.csv` | Bi-LSTM per-patient predictions |
| `xgb_baseline_model.json` | XGBoost Baseline model |
| `xgb_smote_model.json` | XGBoost SMOTE-ENN model |
| `xgb_model_config.json` | XGBoost config |
| `xgb_metrics.json` | XGBoost metrics (both variants) |
| `xgb_evaluation_report.txt` | XGBoost evaluation report |
| `xgb_predictions.csv` | XGBoost per-patient predictions |
| `lgb_smoteenn_model.txt` | LightGBM SMOTE-ENN model |
| `lgb_model_config.json` | LightGBM config |
| `lgb_metrics.json` | LightGBM metrics |
| `lgb_evaluation_report.txt` | LightGBM evaluation report |
| `rf_smoteenn_model.pkl` | Random Forest SMOTE-ENN model |
| `rf_model_config.json` | Random Forest config |
| `rf_metrics.json` | Random Forest metrics |
| `rf_evaluation_report.txt` | Random Forest evaluation report |

### Visualization Outputs

| Plot | Description |
|------|-------------|
| `training_curves.png` | Bi-LSTM train/val loss curves (baseline vs augmented) |
| `evaluation_plots.png` | Bi-LSTM ROC, PR curves, confusion matrices |
| `per_month_performance.png` | Bi-LSTM per-month accuracy/F1/specificity/AUC |
| `xgb_training_curves.png` | XGBoost train/val loss (baseline vs SMOTE-ENN) |
| `xgb_evaluation_plots.png` | XGBoost ROC, PR curves, confusion matrices |
| `xgb_feature_importance.png` | XGBoost top-25 features (Gain) |
| `xgb_per_month_performance.png` | XGBoost per-month performance |
| `lgb_training_curve.png` | LightGBM validation loss curve |
| `lgb_evaluation_plots.png` | LightGBM ROC, PR curve, confusion matrix |
| `lgb_feature_importance.png` | LightGBM top-25 features (Split Count) |
| `lgb_per_month_performance.png` | LightGBM per-month performance |
| `rf_evaluation_plots.png` | Random Forest ROC, PR curve, confusion matrix |
| `rf_feature_importance.png` | Random Forest top-25 features (Gini) |
| `rf_per_month_performance.png` | Random Forest per-month performance |
