# Temporal SLM-to-SHAP Faithfulness Benchmark Migration Design

## Context

The benchmark goal is to evaluate whether SLM-generated per-patient explanations faithfully reflect SHAP attributions from the temporal prediction model.

Current in-progress implementation is split between:
- `non-temporal/` (tests/docs)
- `non_temporal/` (Python package code)

This split is confusing and mismatched to the intended temporal benchmark scope.

## Goal

Consolidate benchmark implementation into one canonical location:

- `evaluation/slm_shap_faithfulness/`

and support two ingestion modes with one shared deterministic scorer:

1. artifact mode (precomputed SHAP JSON)
2. runtime mode (temporal model outputs with optional SHAP recomputation)

## Architecture Decision

Chosen approach: adapter-based dual input pipeline.

### Why

- Keeps one scoring core for fair cross-mode comparisons.
- Prevents duplication and drift between artifact and runtime flows.
- Maintains clean boundaries between ingestion and evaluation logic.

## Canonical Folder Layout

`evaluation/slm_shap_faithfulness/`

- `config.py`
- `schemas.py`
- `feature_map.py`
- `parser.py`
- `shap_truth.py`
- `scorer.py`
- `regression.py`
- `io.py`
- `run_benchmark.py`
- `benchmark_manifest.json`
- `feature_aliases.json`
- `adapters/artifact_adapter.py`
- `adapters/runtime_adapter.py`
- `tests/` (all benchmark tests)
- `README.md`

## Dual-Mode Data Contract

Both adapters must output the same normalized case schema for the shared scorer.

Required per-case fields:
- `patient_id`
- `shap_map` (`feature -> shap_value`)
- `slm_explanation_text`

Preferred fields for traceability:
- `predicted_probability`
- `raw_features`

## Data Flow

1. `run_benchmark.py` selects adapter by `--mode {artifact|runtime}`.
2. Adapter validates input and emits normalized benchmark cases.
3. Core pipeline canonicalizes feature names.
4. SHAP truth builder derives top-K tuples (`feature`, `sign`, `abs_shap`, `rank`, `magnitude_band`).
5. Parser extracts explanation claims (`feature`, `direction`, `magnitude`).
6. Scorer computes deterministic metrics and pass/fail gate.
7. Aggregator writes per-patient and summary artifacts.
8. Optional regression comparison writes deltas versus baseline run.

## Scoring and Gate (Unchanged)

- feature F1 >= 0.80
- sign accuracy >= 0.90
- magnitude accuracy >= 0.75
- composite >= 0.82

Composite:

`0.45 * feature_F1 + 0.35 * sign_accuracy + 0.20 * magnitude_accuracy`

## Error Handling and Validity

- `parse_failed` when no valid claims are extracted.
- `parse_partial` when only some clauses produce claims.
- Unknown features map to `hallucinated_feature`.
- Missing magnitude language maps to `magnitude_missing`.
- Run validity gate requires parser coverage >= 95%; otherwise run is marked invalid.

## Migration and Cleanup Rules

This migration must remove legacy benchmark paths in the same change.

Delete:
- `non_temporal/` benchmark package
- `non-temporal/tests/` benchmark tests
- `non-temporal/faithfulness/README.md`

Update:
- root `README.md` to point only to `evaluation/slm_shap_faithfulness/`

No compatibility wrappers or transitional stubs are kept.

## Temporal Scope Enforcement

- Runtime adapter must validate temporal model context before accepting cases.
- Documentation and fixtures must use temporal SHAP naming and examples.
- Benchmark output schema remains mode-invariant to enable apples-to-apples comparisons.

## Testing Strategy

- Unit tests for core modules (`feature_map`, `parser`, `shap_truth`, `scorer`, `regression`).
- Unit tests for each adapter (`artifact`, `runtime`).
- Integration tests for full benchmark execution in both modes.
- Migration test that asserts no stale imports/references to `non_temporal.faithfulness` remain.

## Acceptance Criteria

Migration is complete when:

- all benchmark code and tests live only under `evaluation/slm_shap_faithfulness/`
- both modes emit identical output schema
- deterministic scoring and conservative gate are enforced exactly
- parser coverage validity gate is enforced
- no benchmark files remain under `non-temporal/` or `non_temporal/`
