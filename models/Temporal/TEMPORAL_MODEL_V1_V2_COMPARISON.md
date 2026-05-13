# TB-DOTS CAR CDSS — Temporal Model v1 vs v2 Comparison

This document compares the earlier temporal pipeline in [v1/TEMPORAL_MODEL_RESULTS.md](v1/TEMPORAL_MODEL_RESULTS.md) and the v1 notebooks ([v1/Hybrid_Bi-LSTM_temporal.ipynb](v1/Hybrid_Bi-LSTM_temporal.ipynb), [v1/lightgbm_temporal.ipynb](v1/lightgbm_temporal.ipynb), [v1/random_forest_temporal.ipynb](v1/random_forest_temporal.ipynb), [v1/xgboost_model_temporal.ipynb](v1/xgboost_model_temporal.ipynb)) against the later v2 pipeline summarized in [v2/results.md](v2/results.md) and implemented in the v2 notebooks under [v2](v2).

Artifacts & outputs

- v1 notebooks (legacy): artifacts are written by the v1 notebooks to the `models/` directory as configured in those notebooks (see `models/Temporal/v1/*` for the notebooks). If you run a v1 notebook in-place it will save checkpoints and plots to `models/` relative to the notebook's working dir.
- v2 notebooks (current): all v2 training artifacts and visualizations are saved inside `models/Temporal/v2/output/` (for example `models/Temporal/v2/output/xgboost/`, `models/Temporal/v2/output/lightgbm/`, `models/Temporal/v2/output/hybrid_lstm/`). This repo now centralizes v2 outputs under that `v2/output` folder.


## Executive Summary

v2 is better than v1 primarily because it is a stronger end-to-end pipeline (data cleaning, temporal encoding, and evaluation), not just different hyperparameters.

The most important methodological shift vs v1 is that v1's held-out test set has only **2 Failure** patients (21 total), which makes specificity/ROC-AUC highly unstable and leads to majority-class collapse. In the current v2 artifacts, all models are evaluated on the same 431-patient split with a 44-patient test set (6 Failure), which makes the reported test metrics far more meaningful.

## Side-by-Side Pipeline Comparison

| Aspect | v1 | v2 | Why it matters |
|---|---|---|---|
| Dataset size | 205 patients | 431 patients | Larger cohorts reduce variance and increase the number of Failure examples |
| Test set | 21 patients, 2 Failure | 44 patients, 6 Failure | v1 metrics can swing drastically based on 1 patient |
| Reported feature count | 204 engineered features | 403 engineered features (tree track) | v2's temporal representation is richer and cleaner |
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

v1 uses 205 patients. The current v2 artifacts use 431 patients. That increase is a key reason v2 results are more credible.

With only 205 patients, the test set contains just 2 Failure cases. In that setting, specificity and ROC-AUC are extremely unstable. One or two prediction changes can swing the reported score dramatically. In v2, the test set has 6 Failure cases (44 total), and the tree models also report 5-fold cross-validation, which further reduces the risk of a lucky or unlucky split driving the conclusion.

### 3. Feature engineering is richer

v1 already used progressive temporal features, but v2 expands the representation further, so the models can use more of the patient trajectory rather than a flattened monthly snapshot.

Note: the latest saved tree-model artifacts report **403 features** (see `models/Temporal/v2/output/*/*_evaluation_report.txt`).

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

- v1 (205p; test=21): accuracy 0.9048, specificity 0.0000, ROC-AUC 0.5789, PR-AUC 0.9501
- v2 (431p; test=44): accuracy 0.9545, specificity 0.8333, ROC-AUC 0.9561, PR-AUC 0.9923

This is the strongest evidence that v2 is better. In v1, the model mostly collapses to predicting Success, so accuracy looks acceptable but failure detection is poor. In v2, XGBoost keeps high accuracy while also recovering Failure cases. That is the outcome you want in a clinical screening model.

### Random Forest

Random Forest also improves sharply.

- v1 (205p; test=21): accuracy 0.7143, ROC-AUC 0.5263, specificity 0.0000
- v2 (431p; test=44): accuracy 0.9318, ROC-AUC 0.9605, specificity 0.8333

This suggests the v1 forest was not getting enough signal from the data representation and class balance. The v2 pipeline gives it cleaner inputs and more usable temporal structure, so it becomes a strong second choice rather than a weak fallback.

### LightGBM

LightGBM changes from a weak result in v1 to a competitive result in v2.

- v1 (205p; test=21): accuracy 0.9048, ROC-AUC 0.2105, specificity 0.0000
- v2 (431p; test=44): accuracy 0.8636, ROC-AUC 0.9167, specificity 0.8333

The jump in ROC-AUC is the key story here. That means v2 is much better at ranking high-risk vs low-risk patients, even if the hard-label accuracy stays close to the other tree models. The v2 cross-validation result also shows that LightGBM is one of the most stable models in the later pipeline.

### Hybrid Bi-LSTM

The Bi-LSTM is the most nuanced case.

- v1 (205p; test=21): accuracy 0.9048, specificity 0.0000, ROC-AUC 0.4737
- v2 (431p; test=44, baseline): accuracy 0.9091, specificity 0.8333, ROC-AUC 0.9254

The v2 Bi-LSTM baseline is clinically more meaningful than v1 in one key respect: it no longer collapses to predicting all patients as Success (specificity is no longer 0.0). In the current v2 artifacts, the baseline Bi-LSTM is also competitive with the tree models on this split.

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

In the current v2 artifacts (431p; test=44), XGBoost has the best Accuracy/F1 point estimate, Random Forest has the best ROC-AUC/PR-AUC, and the Bi-LSTM baseline is competitive. LightGBM remains the most deployment-friendly (ONNX artifacts exist in v2) but is weaker than XGBoost/RF on this run.

So the correct conclusion is not that v2 makes neural models dominate. The correct conclusion is that v2 makes the whole pipeline more trustworthy, and it lets the simpler tree models express the signal much more effectively.

## Conclusion

v2 is stronger than v1 because it fixes the upstream data problem first. The larger cohort, better preprocessing, richer temporal features, and more stable validation scheme all contribute to better generalization. The result is a pipeline that is more useful for clinical decision support, especially for identifying Failure cases instead of just predicting the majority class.

If the goal is the best single model for the thesis, v2 supports XGBoost as the strongest choice, with Random Forest and LightGBM as credible alternatives. If the goal is interpretability plus deployment flexibility, v2 also gives a stronger case for LightGBM than v1 ever did.
