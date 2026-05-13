# SLM-to-SHAP Faithfulness Benchmark Design

## Context

This design defines how to evaluate whether SLM-generated per-patient explanations faithfully reflect SHAP attributions produced by the prediction model. The benchmark target is explanation faithfulness to SHAP, not clinical truth.

Repository context relevant to this work:
- `non-temporal/experiment_pipeline.py`
- `non-temporal/run_experiments.py`
- Temporal notebooks that already compute model-level feature importance artifacts.

## Scope and Goal

Build a deterministic benchmark that scores each SLM explanation against that same patient's SHAP attribution facts using:
- feature fidelity,
- sign/direction fidelity,
- magnitude fidelity,
- and a conservative hard-gated composite score.

Primary evaluation unit: per-patient explanation.

## Approach Decision

Chosen approach: deterministic rule-based scorer (Approach A).

Rationale:
- auditable and reproducible,
- stable enough for pass/fail gates,
- low operational variance compared with model-as-judge methods.

## Benchmark Architecture

### Inputs
- Per-patient model context including raw features, predicted probability, and SHAP values.
- SLM explanation text generated for that same patient.

### Ground Truth Builder
For each patient, create canonical SHAP facts:
- `feature_name_canonical`
- `sign` (increases/decreases predicted risk)
- `abs_shap`
- `rank`
- `magnitude_band`

### Explanation Parser
Extract structured claims from SLM text:
- mentioned features,
- claimed direction/sign,
- claimed magnitude language.

Hybrid strictness policy:
- strict for feature identity and sign,
- semantically tolerant for narrative wording.

### Deterministic Scorer
Compute:
- feature identity precision/recall/F1,
- sign/direction accuracy on matched features,
- magnitude band accuracy,
- weighted composite score.

### Gate Evaluator
Use conservative hard minima plus composite gate:
- feature identity F1 >= 0.80
- sign/direction accuracy >= 0.90
- magnitude band accuracy >= 0.75
- composite >= 0.82

### Outputs
- per-patient scorecard,
- per-patient pass/fail status,
- deterministic failure tags,
- aggregate summary distributions and pass rates.

## Scoring Specification

### SHAP Truth Slice
Use top-`K` SHAP features per patient (`K=5` default) as truth set.

### Feature Fidelity
Compare SLM-mentioned canonical features to SHAP top-`K` set using precision, recall, and F1.

### Sign Fidelity
For matched features only, check whether explanation direction agrees with SHAP sign.

### Magnitude Fidelity
Map absolute SHAP values to per-patient strength bands (`strong`, `moderate`, `weak`) and compare with SLM strength claims.

### Composite
Weighted score:

`0.45 * feature_F1 + 0.35 * sign_accuracy + 0.20 * magnitude_accuracy`

### Failure Tags
Return deterministic tags for remediation:
- `missing_top_feature`
- `wrong_direction`
- `overstated_magnitude`
- `hallucinated_feature`
- `magnitude_missing`

## Data Flow

1. Select fixed benchmark patient cohort (stratified by outcome class).
2. Compute/serialize per-patient SHAP truth tuples.
3. Generate SLM explanation once per patient using production-format input.
4. Parse explanation to structured claims.
5. Score and gate each patient.
6. Aggregate metrics, pass rate, and failure-pattern frequencies.
7. Compare with prior benchmark snapshots for regression tracking.

## Error Handling and Reliability

- If parser extracts partial claims, mark `parse_partial` and score only parseable claims.
- If parser extracts no valid claims, hard-fail with `parse_failed`.
- Unknown feature mentions become `hallucinated_feature` via strict canonical map.
- Negation handling must prevent accidental sign inversion (for example, "did not increase risk").
- Missing strength language yields `magnitude_missing` and counts as magnitude mismatch.

## Reproducibility Controls

- Fixed benchmark split and random seed.
- Fixed SLM decoding parameters for evaluation mode.
- Versioned scorer config (K, thresholds, weights, alias map).
- Versioned benchmark manifests and artifacts for run-to-run comparability.

## Testing Strategy

- Unit tests for parser extraction (feature/sign/magnitude/negation).
- Unit tests for metric calculations and gate logic.
- Gold fixtures with hand-labeled explanation claims.
- Integration test from input artifacts to final scorecard outputs.
- Regression tests to detect scoring drift after scorer/config changes.

## Non-Goals

- This benchmark does not assert clinical correctness of SHAP itself.
- This benchmark does not optimize model discrimination metrics (AUC/F1).
- This benchmark does not evaluate non-SHAP explanation methods in v1.

## Acceptance Criteria

Benchmark v1 is accepted when:
- deterministic per-patient scoring runs end-to-end,
- conservative gate is enforced exactly,
- parser coverage is high enough to trust results (minimum 95% parseable cases),
- outputs include patient-level failures and aggregate summaries suitable for regression tracking.
