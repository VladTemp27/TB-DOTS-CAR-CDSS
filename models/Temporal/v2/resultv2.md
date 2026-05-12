# TB-DOTS CAR CDSS — Temporal Model Results v2

**Date:** May 12, 2026  
**Dataset:** 599 TB patients  
**Split:** 70/20/10 patient-level, stratified  
**Training Mode:** Progressive temporal, 13 samples per patient (M0–M12)  
**Feature Setup:** 8 temporal signals + encoded static baseline features + model-specific feature engineering

This report summarizes the current v2 temporal model outputs from the saved evaluation artifacts in `models/Temporal/v2/output/`.

## Executive Summary

The current v2 results do **not** match the older draft in `results.md`. The strongest held-out test performance comes from the **Bi-LSTM baseline**, while the **XGBoost SMOTE-ENN** variant is close on accuracy and F1 but weaker on ranking metrics. **Random Forest** is the most conservative model, with the highest specificity and a strong OOB score, while **LightGBM** trails the other tree models on the test split.

The main takeaway is that all v2 temporal models are trained on the same patient-level split and same underlying data, but their feature representations and augmentation strategies differ enough that performance changes materially. In this small and imbalanced dataset, those modeling choices matter more than the raw feature source itself.

## Test Set Metrics

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bi-LSTM Baseline | **0.6500** | 0.7692 | 0.5714 | **0.6557** | 0.7600 | **0.7097** | **0.8184** | 0.45 |
| Bi-LSTM Augmented | 0.5667 | 0.6957 | 0.4571 | 0.5517 | 0.7200 | 0.6206 | 0.7447 | 0.81 |
| XGBoost Baseline | 0.6000 | 0.6897 | 0.5714 | 0.6250 | 0.6400 | 0.6720 | 0.7753 | 0.57 |
| XGBoost + SMOTE-ENN | **0.6500** | **0.7692** | 0.5714 | **0.6557** | 0.7600 | 0.6274 | 0.7456 | 0.54 |
| LightGBM | 0.6000 | 0.7037 | 0.5429 | 0.6129 | 0.6800 | 0.6091 | 0.7330 | 0.49 |
| Random Forest | 0.6333 | **0.8421** | 0.4571 | 0.5926 | **0.8800** | 0.6571 | 0.7870 | 0.59 |

## Model Ranking

If ranking by balanced held-out performance, the current results are:

1. **Bi-LSTM Baseline** — best ROC-AUC and PR-AUC, tied best F1/accuracy.
2. **XGBoost + SMOTE-ENN** — tied best accuracy/F1, but weaker ranking metrics than the Bi-LSTM baseline.
3. **Random Forest** — strongest specificity and very good PR-AUC, but lower recall.
4. **XGBoost Baseline** — solid and simpler, but below the two top performers on accuracy.
5. **LightGBM** — stable, but the weakest tree-based result on this split.
6. **Bi-LSTM Augmented** — augmentation hurt overall performance versus the baseline Bi-LSTM.

## Interpretation By Model

### Bi-LSTM Baseline

The baseline Bi-LSTM is the strongest overall model in this run. It achieves the best ROC-AUC and PR-AUC, and its recall stays at 0.5714 with a better precision-recall balance than the augmented variant. The progressive sequence setup appears to help, but the extra noise/mixup/focal-loss stack did not improve the held-out test result.

### Bi-LSTM Augmented

The augmented Bi-LSTM underperforms the baseline on every major metric. In this dataset, the augmentation and focal-loss combination likely made optimization harder without adding enough discriminative signal. That is a sign the temporal signal is limited and the model is sensitive to training instability.

### XGBoost Baseline

The baseline XGBoost is competitive and has the best PR-AUC among the tree models. It does not reach the Bi-LSTM baseline, but it remains a reasonable alternative if a simpler tabular model is preferred.

### XGBoost + SMOTE-ENN

SMOTE-ENN improves XGBoost accuracy, precision, specificity, and F1 relative to the baseline XGBoost, but it lowers ROC-AUC and PR-AUC. That suggests the thresholded classification output improved, but the ranking quality across all thresholds became slightly worse.

### LightGBM

LightGBM is respectable but not the best performer in this v2 run. Its test metrics are below the stronger models, even though the per-month curves and cross-validation output show it can still be reasonably stable.

### Random Forest

Random Forest is the most conservative model here. It has the highest specificity and a strong OOB score, which means it is good at avoiding false Success predictions, but it misses more Failure cases than the best-ranked models.

## Key Takeaways

- The v2 results are **much lower than the older draft report** in `results.md`; that older summary should not be treated as the current source of truth.
- The **best test-set balance** is currently the **Bi-LSTM baseline**, not the augmented Bi-LSTM.
- Augmentation is **not uniformly beneficial**. It helped XGBoost at the chosen threshold, but it hurt the Bi-LSTM run.
- The tree models are competitive, but none beat the Bi-LSTM baseline on overall ranking metrics in the current artifacts.

## Caveats

- The test set is small: **60 patients** only.
- The class balance is still skewed toward Success, so accuracy alone is not sufficient for model selection.
- Thresholds differ by model, so direct accuracy comparisons should be read together with ROC-AUC, PR-AUC, recall, and specificity.
- These are single held-out split results, not repeated nested cross-validation estimates.

## Output Artifacts

- [Bi-LSTM metrics](output/hybrid_lstm/lstm_metrics.json)
- [XGBoost metrics](output/xgboost/xgb_metrics.json)
- [LightGBM metrics](output/lightgbm/lgb_metrics.json)
- [Random Forest metrics](output/random_forest/rf_metrics.json)

For implementation details and per-month performance, see the notebook reports in the corresponding `output/` subfolders.