# TB-DOTS CAR CDSS - Temporal Model Results (v2)

**Last updated**: May 13, 2026

This file is the canonical, thesis-friendly summary for `models/Temporal/v2/`.

Important: the currently saved v2 artifacts were generated on **two different dataset configurations**, so **do not** rank tree models vs the Bi-LSTM run as if they were evaluated on the same split.

---

## What Changed

An older draft of this file contained a single unified ranking across all models.
The latest saved evaluation artifacts show that:

- **Tree models** (XGBoost, Random Forest, LightGBM) were evaluated on a run that reports **431 patients** total and a **44-patient test set**.
- **Bi-LSTM** was evaluated on a run that reports **599 patients** total and a **60-patient test set**.

This report therefore summarizes results in **two tracks**.

---

## Track A - Tree Models (Dataset: 431 patients; Test: 44 patients)

Source artifacts:

- `output/xgboost/xgb_evaluation_report.txt`, `output/xgboost/xgb_metrics.json`
- `output/random_forest/rf_evaluation_report.txt`, `output/random_forest/rf_metrics.json`
- `output/lightgbm/lgb_evaluation_report.txt`, `output/lightgbm/lgb_metrics.json`

### Test Set Metrics (M12)

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost (Baseline) | **0.9545** | 0.9035 | 0.9737 | **0.9737** | **0.9737** | 0.8333 | 0.9561 | 0.9923 | 0.76 |
| XGBoost (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.9730 | 0.9474 | 0.9600 | 0.8333 | 0.9561 | 0.9923 | 0.70 |
| Random Forest (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.9730 | 0.9474 | 0.9600 | 0.8333 | **0.9605** | **0.9934** | 0.61 |
| LightGBM (+SMOTE-ENN) | 0.8636 | 0.8509 | 0.9706 | 0.8684 | 0.9167 | 0.8333 | 0.9167 | 0.9827 | 0.86 |

### Confusion Matrices

XGBoost (Baseline):

```
[[ 5  1]
 [ 1 37]]
```

XGBoost (+SMOTE-ENN):

```
[[ 5  1]
 [ 2 36]]
```

Random Forest (+SMOTE-ENN):

```
[[ 5  1]
 [ 2 36]]
```

LightGBM (+SMOTE-ENN):

```
[[ 5  1]
 [ 5 33]]
```

### Notes (Tree Track)

- XGBoost baseline is the strongest on thresholded metrics (Accuracy/F1/Recall).
- Random Forest is best on ranking metrics in this track (ROC-AUC/PR-AUC) and reports `oob_score=0.9858`.
- LightGBM underperforms the other two tree families on this run.
- All three tree reports include per-month tables (M0..M12) with `n_total=44`.

---

## Artifact Index

- Tree models:
  - `output/xgboost/xgb_evaluation_report.txt`
  - `output/random_forest/rf_evaluation_report.txt`
  - `output/lightgbm/lgb_evaluation_report.txt`
- Bi-LSTM:
  - `output/hybrid_lstm/evaluation_report.txt`
