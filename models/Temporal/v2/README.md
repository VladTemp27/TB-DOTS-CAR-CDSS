# v2 Temporal Models

This folder contains the current temporal-model pipeline for TB-DOTS treatment success/failure prediction.

## Current Dataset

The v2 notebooks now use:

`dataset/temporal/output/cleaned_human_readable.csv`

This is the audited missing-data output, not `combined_complete_dataset.csv`.

Only patients with observed outcomes are used for supervised training:

- Filter: `is_missing_outcome == 0`
- Cohort: 431 patients
- Labels: 368 Success, 63 Failure
- Split: 70/20/10 patient-level stratified
- Test set: 44 patients, 38 Success and 6 Failure

## Feature Policy

The current feature policy is `temporal_v2_cleaned_output_facility_v1`.

`facility` is intentionally included as a normalized operational/site-context predictor because the thesis goal is holistic risk prediction under the DOH-CAR TB-DOTS program.

Dropped fields are limited to leakage, identifiers, or redundant raw fields:

- Outcome leakage: `date_of_outcome`, `is_missing_outcome`
- Administrative/source identifiers: `no`, `source_file`, `data_year`
- Raw facility name fields: `name_of_diagnosing_facility`, `name_of_treatment_unit`
- Privacy/redundant fields: `date_of_birth`, raw `height`, raw `weight`
- Late treatment timeline fields: `intensive_phase_end_date`, `continuation_phase_start_date`, `continuation_phase_end_date`

## Contents

- [Hybrid_Bi-LSTM_temporal.ipynb](Hybrid_Bi-LSTM_temporal.ipynb)
- [lightgbm_temporal.ipynb](lightgbm_temporal.ipynb)
- [random_forest_temporal.ipynb](random_forest_temporal.ipynb)
- [xgboost_model_temporal.ipynb](xgboost_model_temporal.ipynb)
- [results.md](results.md)

## Evaluation Notes

- SMOTE/SMOTE-ENN is applied only to training data.
- Validation/test distributions remain natural and imbalanced.
- Thresholds are selected with failure-aware validation objectives, not plain accuracy.
- Main reported metrics include balanced accuracy, specificity/failure recall, ROC-AUC, PR-AUC, and confusion matrices.
- Per-month performance is reported from M0 through M12 to support temporal interpretation.

## Use It For

- Reviewing the final facility-inclusive v2 temporal modeling pipeline.
- Referencing thesis-ready metrics and rationale in [results.md](results.md).
- Regenerating deployment artifacts under `models/Temporal/v2/output/`.
