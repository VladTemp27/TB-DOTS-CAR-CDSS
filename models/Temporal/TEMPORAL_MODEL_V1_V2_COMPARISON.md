# TB-DOTS CAR CDSS — Temporal Model v1 vs v2 Comparison

This document compares the earlier temporal pipeline in [v1/TEMPORAL_MODEL_RESULTS.md](v1/TEMPORAL_MODEL_RESULTS.md) and the v1 notebooks ([v1/Hybrid_Bi-LSTM_temporal.ipynb](v1/Hybrid_Bi-LSTM_temporal.ipynb), [v1/lightgbm_temporal.ipynb](v1/lightgbm_temporal.ipynb), [v1/random_forest_temporal.ipynb](v1/random_forest_temporal.ipynb), [v1/xgboost_model_temporal.ipynb](v1/xgboost_model_temporal.ipynb)) against the later v2 pipeline summarized in [v2/results.md](v2/results.md) and implemented in the v2 notebooks under [v2](v2).

## Executive Summary

v2 is better than v1 for the main reason that it is a materially stronger pipeline, not just a different set of model hyperparameters. The later version uses a much larger cohort, cleaner preprocessing, richer temporal encoding, and a more credible evaluation setup. That combination makes the final metrics more stable and more clinically useful, especially for the tree-based models.

The most important shift is that v1 was evaluated on a very small test set with only 2 Failure cases, which made the metrics highly unstable and encouraged majority-class collapse. v2 expands the study to 599 patients, raises the test set to 60 patients with 6 Failure cases, and adds more complete preprocessing and feature engineering. As a result, v2 shows substantially better specificity, ROC-AUC, and PR-AUC for XGBoost, Random Forest, and LightGBM.

## Side-by-Side Pipeline Comparison

| Aspect | v1 | v2 | Why it matters |
|---|---|---|---|
| Dataset size | 205 patients | 599 patients | v2 has much lower variance and more failure examples to learn from |
| Test set | 21 patients, 2 Failure | 60 patients, 6 Failure | v2 metrics are less sensitive to one or two misclassified failures |
| Reported feature count | 204 engineered features | 407 engineered features | v2 has a richer representation of clinical trajectory |
| Static preprocessing | Encoded/scaled features, outcome columns removed | MICE imputation, categorical harmonization, sanity clipping, date cleaning | v2 is less noisy and less likely to carry missing-value artifacts |
| Temporal preprocessing | Progressive monthly features | Progressive monthly features plus expanded monthly structure and missingness handling | v2 captures more signal from month-to-month change |
| Imbalance handling | SMOTE-ENN and class weights/sampler variants | Same ideas, but on a stronger underlying dataset | The same balancing strategy works better when the input data is cleaner |
| Thresholding | Validation-set threshold search | Validation-set threshold search | Similar method, but v2 thresholds are more trustworthy because validation is better populated |
| Evaluation | Small test set, weaker separation | Larger test set, 5-fold CV reported for tree models | v2 gives a more defensible estimate of real performance |

## What Changed in the Pipeline

### 1. Data quality improved first, model second

The v2 notebooks do much more than train a model. The preprocessing stage now includes:

- MICE imputation for missing values
- Clinical sanity checks and clipping for implausible values
- Harmonization of categorical values
- Temporal structuring of the monthly variables into a richer patient-month representation

That matters because tree models and sequence models both learn from the same underlying representation. If the representation is noisy, the model can only memorize noise. v2 reduces that noise before training starts.

### 2. The cohort is much larger

v1 uses 205 patients. v2 uses 599 patients. That is the biggest reason the v2 results are more credible.

With only 205 patients, the test set contains just 2 Failure cases. In that setting, specificity and ROC-AUC are extremely unstable. One or two prediction changes can swing the reported score dramatically. In v2, the test set has 6 Failure cases, and the cross-validation results further reduce the risk of a lucky or unlucky split driving the conclusion.

### 3. Feature engineering is richer

v1 already used progressive temporal features, but v2 expands the representation further. The v2 summary reports 407 engineered features, which means the models can now use more of the patient trajectory, not just a flattened monthly snapshot.

The effect is especially visible for the tree-based models:

- raw monthly values still matter
- aggregates matter because they summarize treatment progression
- trend features matter because they capture direction of change
- latest-month features matter because recent status is clinically important

This is the right direction for TB outcome prediction, because treatment failure is often signaled by a pattern rather than a single isolated month.

### 4. Evaluation is more realistic

v2 is not relying only on a single held-out split. The tree-based models also report 5-fold cross-validation, which is a much better robustness check for a small-to-medium medical dataset.

That is especially important here because test-set ROC-AUC in v1 is not a reliable indicator of model quality when the positive class is so rare. v2’s CV results show the models are actually learning a usable signal rather than overfitting a tiny test fold.

## Model-by-Model Comparison

### XGBoost

XGBoost shows the clearest improvement.

- v1: accuracy 0.9048, specificity 0.0000, ROC-AUC 0.5789, PR-AUC 0.9501
- v2: accuracy 0.9167, specificity 0.8333, ROC-AUC 0.9259, PR-AUC 0.9908

This is the strongest evidence that v2 is better. In v1, the model mostly collapses to predicting Success, so accuracy looks acceptable but failure detection is poor. In v2, XGBoost keeps high accuracy while also recovering Failure cases. That is the outcome you want in a clinical screening model.

### Random Forest

Random Forest also improves sharply.

- v1: accuracy 0.7143, ROC-AUC 0.5263, specificity 0.0000
- v2: accuracy 0.9000, ROC-AUC 0.9383, specificity 0.8333

This suggests the v1 forest was not getting enough signal from the data representation and class balance. The v2 pipeline gives it cleaner inputs and more usable temporal structure, so it becomes a strong second choice rather than a weak fallback.

### LightGBM

LightGBM changes from a weak result in v1 to a competitive result in v2.

- v1: accuracy 0.9048, ROC-AUC 0.2105, specificity 0.0000
- v2: accuracy 0.9000, ROC-AUC 0.9136, specificity 0.8333

The jump in ROC-AUC is the key story here. That means v2 is much better at ranking high-risk vs low-risk patients, even if the hard-label accuracy stays close to the other tree models. The v2 cross-validation result also shows that LightGBM is one of the most stable models in the later pipeline.

### Hybrid Bi-LSTM

The Bi-LSTM is the most nuanced case.

- v1: accuracy 0.9048, specificity 0.0000, ROC-AUC 0.4737
- v2: accuracy 0.8667, specificity 0.3333, ROC-AUC 0.8272

The v2 Bi-LSTM is not the best model by accuracy, but it is clinically more meaningful than v1 because it stops behaving like a pure majority-class predictor. The lower accuracy is not necessarily a regression; it comes from the model making more Failure predictions, which lowers raw accuracy but improves failure detection and ranking.

In other words, v2 makes the Bi-LSTM more conservative and more informative, even if it still does not beat the tree-based models for production use.

## Why v2 Is Better Overall

### 1. It reduces label-skew artifacts

The biggest flaw in v1 is that the evaluation can look good while the model still fails at the clinically important task of catching Failure cases. v2 reduces this problem by using a larger and more balanced evaluation context.

### 2. It improves the input signal

MICE imputation, sanity checks, and harmonization remove avoidable noise. That is a direct quality improvement to the data the models see.

### 3. It uses a richer temporal representation

The later pipeline does a better job of turning monthly monitoring into learning signal. That is why the tree models gain so much in ROC-AUC and specificity.

### 4. It validates robustness more honestly

The cross-validation results in v2 are important because they show the tree models are stable, not just lucky on a tiny split. This is the main reason v2 is a stronger result to present in a thesis or a deployment discussion.

## Important Caveat

v2 is better, but not because every model is universally better in every metric.

- XGBoost is the best overall point estimate
- Random Forest has the best ROC-AUC
- LightGBM is the most deployment-friendly and very stable
- Bi-LSTM captures temporal structure but still lags the tree models for this dataset

So the correct conclusion is not that v2 makes neural models dominate. The correct conclusion is that v2 makes the whole pipeline more trustworthy, and it lets the simpler tree models express the signal much more effectively.

## Conclusion

v2 is stronger than v1 because it fixes the upstream data problem first. The larger cohort, better preprocessing, richer temporal features, and more stable validation scheme all contribute to better generalization. The result is a pipeline that is more useful for clinical decision support, especially for identifying Failure cases instead of just predicting the majority class.

If the goal is the best single model for the thesis, v2 supports XGBoost as the strongest choice, with Random Forest and LightGBM as credible alternatives. If the goal is interpretability plus deployment flexibility, v2 also gives a stronger case for LightGBM than v1 ever did.