# TB-DOTS CAR CDSS - Temporal Model v1 vs v2 Comparison

This document compares the legacy temporal pipeline in `models/Temporal/v1/` with the current v2 pipeline in `models/Temporal/v2/`.

## Executive Summary

v2 is the stronger thesis pipeline because it fixes the data source, label filtering, missing-data handling, leakage control, temporal feature construction, and evaluation strategy before model comparison.

The current v2 artifacts use the audited missing-data output:

`dataset/temporal/output/cleaned_human_readable.csv`

Only patients with observed outcomes are used for supervised learning:

- Filter: `is_missing_outcome == 0`
- Cohort: 431 patients
- Labels: 368 Success, 63 Failure
- Test set: 44 patients, 38 Success and 6 Failure

The normalized `facility` column is included in v2 because the thesis objective is holistic treatment success/failure prediction under the DOH-CAR TB-DOTS program. Raw facility-name fields are still dropped to avoid duplicated/high-cardinality administrative noise.

## Why The v2 Update Was Needed

The previous v2 draft used the wrong dataset source path and risked mixing preprocessing responsibilities. The corrected v2 pipeline now uses the already-audited cleaned CSV, filters out imputed outcome labels, and documents the feature policy used by the saved artifacts.

This matters because medical ML results are not defensible if the target label is imputed or if final outcome fields leak into predictors. The current v2 setup corrects this by:

- excluding rows where `is_missing_outcome == 1`
- dropping `date_of_outcome` and `is_missing_outcome`
- dropping direct identifiers/source fields such as `no`, `source_file`, and `data_year`
- keeping `facility` as the normalized operational context feature
- dropping raw facility fields: `name_of_diagnosing_facility`, `name_of_treatment_unit`
- dropping redundant raw strings: `height`, `weight`
- dropping late treatment timeline fields: `intensive_phase_end_date`, `continuation_phase_start_date`, `continuation_phase_end_date`
- selecting thresholds with failure-aware validation objectives rather than plain accuracy

## Side-by-Side Pipeline Comparison

| Aspect | v1 | Current v2 | Why it matters |
|---|---|---|---|
| Dataset source | Earlier temporal cleaned set | `dataset/temporal/output/cleaned_human_readable.csv` | v2 uses the audited missing-data output |
| Labeled cohort | 205 patients | 431 observed-outcome patients | More Failure examples and better stability |
| Label handling | Outcome labels from old pipeline | Drops `is_missing_outcome == 1` before training | Avoids training on imputed labels |
| Test set | 21 patients, 2 Failure | 44 patients, 6 Failure | v1 metrics swing heavily from one patient |
| Facility feature | Not consistently documented | Included as normalized `facility` | Supports holistic TB-DOTS operational risk prediction |
| Leakage control | Basic outcome removal | Explicit feature policy `temporal_v2_cleaned_output_facility_v1` | More defensible for medical temporal prediction |
| Feature count | About 204 engineered features | 399 tree-model features | Uses temporal values, missingness indicators, aggregates, trends, latest values |
| Thresholding | Validation threshold search | Failure-aware validation threshold search | Reduces misleading accuracy-first selection |
| Metrics | Accuracy-heavy summaries | Balanced accuracy, specificity, ROC-AUC, PR-AUC, confusion matrices, per-month results | Better for imbalanced medical outcome prediction |

## Current v2 Test Results

Current v2 test set: 44 patients, 6 Failure and 38 Success.

| Model | Accuracy | Balanced Acc | Specificity | ROC-AUC | PR-AUC | Confusion Matrix |
|---|---:|---:|---:|---:|---:|---|
| Random Forest (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.8333 | 0.9693 | 0.9950 | `[[5, 1], [2, 36]]` |
| LightGBM (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.8333 | 0.8991 | 0.9779 | `[[5, 1], [2, 36]]` |
| XGBoost (+SMOTE-ENN) | 0.9318 | 0.8904 | 0.8333 | 0.9298 | 0.9863 | `[[5, 1], [2, 36]]` |
| XGBoost (Baseline) | 0.9091 | 0.8772 | 0.8333 | 0.9430 | 0.9895 | `[[5, 1], [3, 35]]` |
| Hybrid Bi-LSTM (Baseline) | 0.8409 | 0.9079 | 1.0000 | 0.9868 | 0.9981 | `[[6, 0], [7, 31]]` |
| Hybrid Bi-LSTM (Augmented) | 0.7500 | 0.8553 | 1.0000 | 0.9474 | 0.9910 | `[[6, 0], [11, 27]]` |

## Model-by-Model Comparison

### XGBoost

- v1: accuracy 0.9048, specificity 0.0000, ROC-AUC 0.5789, PR-AUC 0.9501
- current v2 SMOTE-ENN: accuracy 0.9318, specificity 0.8333, ROC-AUC 0.9298, PR-AUC 0.9863

XGBoost improves because v2 gives it more observed labels, cleaned temporal features, missingness indicators, and better thresholding. The current SMOTE-ENN variant detects 5 of 6 Failure cases and 36 of 38 Success cases.

### Random Forest

- v1: accuracy 0.7143, specificity 0.0000, ROC-AUC 0.5263
- current v2: accuracy 0.9318, specificity 0.8333, ROC-AUC 0.9693, PR-AUC 0.9950

Random Forest is the strongest tree model by held-out ROC-AUC and PR-AUC in the current v2 artifacts. This suggests the corrected feature set and facility-inclusive context are useful for ranking risk.

### LightGBM

- v1: accuracy 0.9048, specificity 0.0000, ROC-AUC 0.2105
- current v2: accuracy 0.9318, specificity 0.8333, ROC-AUC 0.8991, PR-AUC 0.9779

LightGBM now performs competitively on hard-label metrics and has the best reported 5-fold cross-validation balanced accuracy among the current tree artifacts: 0.8929 +/- 0.0611.

### Hybrid Bi-LSTM

- v1: accuracy 0.9048, specificity 0.0000, ROC-AUC 0.4737
- current v2 baseline: accuracy 0.8409, specificity 1.0000, ROC-AUC 0.9868, PR-AUC 0.9981

The v2 Bi-LSTM no longer collapses to predicting Success. It catches all 6 Failure cases on the test split, but it also misclassifies 7 Success cases as Failure, so accuracy is lower than the tree models. This may be clinically acceptable in a screening-oriented setting, but it should be discussed as a sensitivity/specificity tradeoff.

## Why v2 Is Better Overall

### 1. It uses the correct supervised cohort

v2 excludes imputed outcome labels and trains only on observed outcomes. This is the most important correction for medical validity.

### 2. It preserves operational context

The normalized `facility` feature is included because facility-level differences may reflect access, follow-up systems, and local TB-DOTS implementation. This matches the holistic risk-prediction framing of the thesis.

### 3. It controls leakage more explicitly

v2 drops final outcome fields, direct identifiers, source fields, raw duplicated strings, and late treatment timeline variables that can make performance look better than deployable reality.

### 4. It evaluates the minority class more honestly

Failure detection is clinically important. v2 reports balanced accuracy, specificity/failure recall, per-month performance, ROC-AUC, PR-AUC, and confusion matrices instead of relying on plain accuracy.

### 5. It provides temporal interpretation

Per-month M0 through M12 metrics show how predictive performance changes as more treatment monitoring data becomes available. This is more appropriate for a TB-DOTS temporal CDSS than only reporting a final M12 score.

## Caveat

The v2 test set still has only 6 Failure patients. One Failure case changes specificity by 16.7 percentage points. For thesis interpretation, emphasize cross-validation, per-month trends, and confusion matrices together rather than treating one test split as a definitive deployment estimate.

## Conclusion

v2 is stronger than v1 because it corrects the dataset source, filters to observed labels, includes facility as a justified operational predictor, improves leakage control, and evaluates models with metrics appropriate for imbalanced medical prediction.

For the current artifacts, the tree models are strongest on hard-label accuracy and F1, with Random Forest leading held-out ranking metrics. The Bi-LSTM baseline is more conservative and catches all Failure cases in the test split, but at the cost of more false Failure predictions among true Success cases.
