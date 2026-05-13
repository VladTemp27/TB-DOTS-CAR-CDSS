# TB-DOTS CAR CDSS - Temporal Model Results (v2)

**Last updated**: May 13, 2026

This file is the canonical, thesis-friendly summary for `models/Temporal/v2/`.

Important: this report reflects the current saved artifacts in `models/Temporal/v2/output/`.

---

## What Changed

An older draft of this file contained a single unified ranking across all models.
The latest saved evaluation artifacts show that the models below were evaluated on the same run:

- **Dataset**: 431 patients (70/20/10 patient-level)
- **Test set**: 44 patients (6 Failure, 38 Success)

---

## Test Set Results (Dataset: 431 patients; Test: 44 patients)

Source artifacts:

- `output/xgboost/xgb_evaluation_report.txt`, `output/xgboost/xgb_metrics.json`
- `output/random_forest/rf_evaluation_report.txt`, `output/random_forest/rf_metrics.json`
- `output/lightgbm/lgb_evaluation_report.txt`, `output/lightgbm/lgb_metrics.json`
- `output/hybrid_lstm/evaluation_report.txt`, `output/hybrid_lstm/lstm_metrics.json`

### Test Set Metrics (M12)

| Model | Accuracy | Balanced Acc | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost (Baseline) | **0.9545** | 0.9035 | 0.9737 | **0.9737** | **0.9737** | 0.8333 | 0.9561 | 0.9923 | 0.76 |
| XGBoost (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.9730 | 0.9474 | 0.9600 | 0.8333 | 0.9561 | 0.9923 | 0.70 |
| Random Forest (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.9730 | 0.9474 | 0.9600 | 0.8333 | **0.9605** | **0.9934** | 0.61 |
| LightGBM (+SMOTE-ENN) | 0.8636 | 0.8509 | 0.9706 | 0.8684 | 0.9167 | 0.8333 | 0.9167 | 0.9827 | 0.86 |
| Hybrid Bi-LSTM (Baseline) | 0.9091 | 0.8772 | 0.9722 | 0.9211 | 0.9459 | 0.8333 | 0.9254 | 0.9852 | 0.31 |
| Hybrid Bi-LSTM (Augmented) | 0.8182 | 0.8246 | 0.9688 | 0.8158 | 0.8857 | 0.8333 | 0.9342 | 0.9874 | 0.89 |

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

Hybrid Bi-LSTM (Baseline):

```
[[ 5  1]
 [ 3 35]]
```

Hybrid Bi-LSTM (Augmented):

```
[[ 5  1]
 [ 7 31]]
```

### Notes

- XGBoost baseline is the strongest on thresholded metrics (Accuracy/F1/Recall).
- Random Forest is best on ranking metrics in this track (ROC-AUC/PR-AUC) and reports `oob_score=0.9858`.
- LightGBM underperforms the other two tree families on this run.
- Hybrid Bi-LSTM baseline is competitive on this split; the augmented variant improves ROC-AUC slightly but reduces thresholded metrics.
- All reports include per-month tables (M0..M12) with `n_total=44`.

## Artifact Index

- Tree models:
  - `output/xgboost/xgb_evaluation_report.txt`
  - `output/random_forest/rf_evaluation_report.txt`
  - `output/lightgbm/lgb_evaluation_report.txt`
- Bi-LSTM:
  - `output/hybrid_lstm/evaluation_report.txt`
  - `output/hybrid_lstm/lstm_metrics.json`
