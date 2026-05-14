# TB-DOTS CAR CDSS - Temporal Model Results (v2)

**Last updated**: May 14, 2026

This file summarizes the current saved artifacts in `models/Temporal/v2/output/`.

## Current Run

- Dataset source: `dataset/temporal/output/cleaned_human_readable.csv`
- Label filter: `is_missing_outcome == 0`
- Cohort: 431 patients
- Class distribution: 368 Success, 63 Failure
- Split: 70/20/10 patient-level stratified
- Test set: 44 patients, 38 Success and 6 Failure
- Feature policy: `temporal_v2_cleaned_output_facility_v1`
- Feature count: 399 for tree models
- Facility: included as normalized `facility_*` one-hot features

## Reason For The Update

The earlier v2 runs used the wrong dataset source path. The corrected pipeline uses the audited missing-data output CSV and excludes rows where the outcome was originally missing. This prevents training on imputed labels. The normalized `facility` column is kept because the study is framed as holistic risk prediction under the DOH-CAR TB-DOTS program, where facility/site context may reflect operational differences affecting treatment outcomes.

The pipeline still drops leakage or redundant fields such as `date_of_outcome`, `is_missing_outcome`, raw facility-name fields, source identifiers, raw `height`/`weight`, and late treatment timeline fields.

## Test Set Metrics (M12)

Source artifacts:

- `output/xgboost/xgb_evaluation_report.txt`, `output/xgboost/xgb_metrics.json`
- `output/random_forest/rf_evaluation_report.txt`, `output/random_forest/rf_metrics.json`
- `output/lightgbm/lgb_evaluation_report.txt`, `output/lightgbm/lgb_metrics.json`
- `output/hybrid_lstm/evaluation_report.txt`, `output/hybrid_lstm/lstm_metrics.json`

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.9730 | 0.9474 | 0.9600 | 0.8333 | **0.9693** | 0.9950 | 0.67 |
| LightGBM (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.9730 | 0.9474 | 0.9600 | 0.8333 | 0.8991 | 0.9779 | 0.78 |
| XGBoost (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.9730 | 0.9474 | 0.9600 | 0.8333 | 0.9298 | 0.9863 | 0.66 |
| XGBoost (Baseline) | 0.9091 | 0.8772 | 0.9722 | 0.9211 | 0.9459 | 0.8333 | 0.9430 | 0.9895 | 0.81 |
| Hybrid Bi-LSTM (Baseline) | 0.8409 | **0.9079** | **1.0000** | 0.8158 | 0.8986 | **1.0000** | **0.9868** | **0.9981** | 0.55 |
| Hybrid Bi-LSTM (Augmented) | 0.7500 | 0.8553 | **1.0000** | 0.7105 | 0.8308 | **1.0000** | 0.9474 | 0.9910 | 0.89 |

## Confusion Matrices

Random Forest (+SMOTE-ENN), LightGBM (+SMOTE-ENN), and XGBoost (+SMOTE-ENN):

```text
[[ 5  1]
 [ 2 36]]
```

XGBoost (Baseline):

```text
[[ 5  1]
 [ 3 35]]
```

Hybrid Bi-LSTM (Baseline):

```text
[[ 6  0]
 [ 7 31]]
```

Hybrid Bi-LSTM (Augmented):

```text
[[ 6  0]
 [11 27]]
```

## Cross-Validation Summary

Tree models report 5-fold cross-validation in their evaluation reports.

| Model | Balanced Accuracy | F1 | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| LightGBM (+SMOTE-ENN) | **0.8929 +/- 0.0611** | **0.9717 +/- 0.0115** | 0.9508 +/- 0.0410 | **0.9682 +/- 0.0212** | **0.9756 +/- 0.0157** |
| Random Forest (+SMOTE-ENN) | 0.8819 +/- 0.0589 | 0.9676 +/- 0.0099 | **0.9542 +/- 0.0424** | 0.9655 +/- 0.0209 | 0.9702 +/- 0.0179 |

## Interpretation

- The tree models now converge to similar hard-label performance on the test set: 41/44 correct, 5/6 failures correctly identified, and 36/38 successes correctly identified.
- Random Forest has the best tree-model ranking performance on the held-out test set by ROC-AUC and PR-AUC.
- LightGBM has the best reported cross-validation balanced accuracy among the tree models in the current artifacts.
- The Bi-LSTM baseline has the highest balanced accuracy and ranking metrics on the single test split, but it classifies more true successes as failures, reducing accuracy and success recall.
- Because the test set has only 6 Failure patients, balanced accuracy, specificity, per-month performance, and cross-validation should be emphasized over plain accuracy.

## Artifact Index

- Tree models:
  - `output/xgboost/xgb_evaluation_report.txt`
  - `output/random_forest/rf_evaluation_report.txt`
  - `output/lightgbm/lgb_evaluation_report.txt`
- Bi-LSTM:
  - `output/hybrid_lstm/evaluation_report.txt`
  - `output/hybrid_lstm/lstm_metrics.json`
