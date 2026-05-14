# SLM-SHAP Producer Pipeline

Runs the 44-patient V2 test cohort through the LightGBM model, computes SHAP values, calls Gemini for natural-language explanations under two experimental conditions, and writes `case_*.json` files for the faithfulness evaluator.

## Prerequisites

All paths are relative to the repo root. Run everything from there.

```
models/Temporal/v2/output/lightgbm/lgb_smoteenn_model.txt  ← V2 LightGBM model
dataset/temporal/output/cleaned_human_readable.csv           ← V2 patient data
slm_shap_pipeline/feature_groups.json                        ← SHAP group map
```

**Provider options:**

| Provider | Requirement |
|---|---|
| `cli` (default) | `gemini` CLI installed and logged in (`gemini auth login`) |
| `api` | `GOOGLE_API_KEY` env var set; `google-generativeai` package installed |

## Quickstart

### 1. Smoke test (no real Gemini calls)

```bash
python -m slm_shap_pipeline.run_pipeline \
  --condition sighted \
  --patient-limit 2 \
  --dry-run-gemini
```

Writes 2 case files to `outputs/slm_shap_pipeline/<timestamp>/sighted/cases/`. Check `manifest.json` to confirm the run metadata.

### 2. Full production run

```bash
# Sighted condition first (SHAP values shown to the model)
python -m slm_shap_pipeline.run_pipeline --condition sighted

# Then blind (no SHAP — model infers from demographics only)
python -m slm_shap_pipeline.run_pipeline --condition blind
```

Each run produces 44 `case_*.json` files. Both conditions write to the same `outputs/slm_shap_pipeline/latest/` symlink — they land in separate `sighted/` and `blind/` subdirectories so they don't overwrite each other.

Expected runtime: ~10–15 min per condition (44 Gemini calls, sequential).

## CLI reference

```
python -m slm_shap_pipeline.run_pipeline --help

  --condition {blind,sighted}   Required. Experimental condition.
  --patient-limit N             Stop after N patients (for testing).
  --dry-run-gemini              Skip real Gemini calls; write stub explanations.
  --provider {cli,api}          SLM backend. Default: cli.
  --model MODEL                 Override Gemini model name.
  --output-base PATH            Override output directory.
  --model-path PATH             Override model file path.
  --csv-path PATH               Override CSV data path.
```

### Switch to API provider

```bash
export GOOGLE_API_KEY="your-key"
python -m slm_shap_pipeline.run_pipeline \
  --condition sighted \
  --provider api \
  --model gemini-2.5-pro
```

API provider bypasses the CLI subprocess and calls the SDK directly. The cache key is provider-agnostic — a CLI-warmed cache is reusable under API and vice versa.

## Output structure

```
outputs/slm_shap_pipeline/
  latest -> 2026-05-14T10-00-00/     ← symlink to most recent run
  2026-05-14T10-00-00/
    sighted/
      cases/
        case_<patient_id>.json        ← one per patient
      manifest.json                   ← run metadata + stats
    blind/
      cases/
        case_<patient_id>.json
      manifest.json
```

### case_*.json schema

```json
{
  "patient_id": "...",
  "condition": "sighted",
  "month_of_prediction": 12,
  "model_prediction": 0.73,
  "explanation": "...",           ← raw Gemini output
  "shap_values": {
    "Treatment_Adherence": -0.41,
    "Age": 0.12,
    ...
  },
  "prediction_metadata": {
    "model_hash": "...",
    "scaler_static_hash": "...",
    "scaler_temporal_hash": "...",
    "feature_policy_version": "temporal_v2_cleaned_output_facility_v1",
    "feature_groups_hash": "...",
    "prompt_git_sha": "...",
    "gemini_call_timestamp": "..."
  }
}
```

## Scoring output with the evaluator

```bash
# Score sighted condition
python -m evaluation.slm_shap_faithfulness.run_benchmark \
  --input-dir outputs/slm_shap_pipeline/latest/sighted/cases \
  --output-dir outputs/slm_shap_pipeline/latest/sighted \
  --mode artifact

# Score blind condition (compare against sighted baseline)
python -m evaluation.slm_shap_faithfulness.run_benchmark \
  --input-dir outputs/slm_shap_pipeline/latest/blind/cases \
  --output-dir outputs/slm_shap_pipeline/latest/blind \
  --mode artifact \
  --baseline-path outputs/slm_shap_pipeline/latest/sighted/results.json

# View headline result
cat outputs/slm_shap_pipeline/latest/sighted/results.json | python -m json.tool
```

Expected: `sighted.pass_rate` substantially higher than `blind.pass_rate`. The `regression.pass_rate_delta` is the headline number quantifying the SHAP-exposure advantage.

## Validate case files before scoring

Checks that every case file has required fields and that SHAP keys are in the evaluator's alias map:

```bash
python scripts/validate_case_schema.py \
  outputs/slm_shap_pipeline/latest/sighted/cases/
```

Expected output: `44 cases checked. 0 with issues.`

If you get `shap_values key not in alias map` errors, add the missing key to `evaluation/slm_shap_faithfulness/feature_aliases.json`.

## Caching

Responses are cached in `slm_shap_pipeline/cache/` keyed on patient ID, condition, prompt text, model hash, scaler hashes, and feature-groups hash. Re-running after a partial failure picks up where it left off. Changing `backend/prompt.py`, `feature_groups.json`, the model file, or the scalers automatically invalidates the cache for affected patients.

To force a fresh run, delete `slm_shap_pipeline/cache/`.

## Tests

```bash
# Unit + integration tests (fast, no Gemini calls)
pytest slm_shap_pipeline/tests/ -v

# E2E smoke tests (runs 2 patients against real model, dry-run Gemini)
pytest slm_shap_pipeline/tests/test_pipeline_e2e.py -v

# Evaluator tests
pytest evaluation/slm_shap_faithfulness/tests/ -v
```

Expected: 61 pipeline tests + 35 evaluator tests, all green.

## Module map

| File | Responsibility |
|---|---|
| `config.py` | `PipelineConfig` dataclass with all defaults |
| `data_loader.py` | Load 44-patient test cohort via `model_utils`; fit+apply scalers |
| `model_loader.py` | Load LightGBM booster; compute model hash |
| `feature_builder.py` | Reconstruct semantic feature names via `build_features_at_month` |
| `shap_runner.py` | TreeExplainer SHAP values → `dict[feature_name, float]` |
| `feature_aggregator.py` | Collapse 399 raw SHAP values → ~25 clinical groups |
| `prompt_builder.py` | Build sighted/blind prompt strings |
| `providers/` | `GeminiCLIProvider` + `GoogleAPIProvider` (hotswap via `make_provider`) |
| `slm_client.py` | Call provider with disk-based caching |
| `case_writer.py` | Write `case_*.json` and `manifest.json` |
| `pipeline.py` | Orchestrator: wires all modules for one condition |
| `run_pipeline.py` | CLI entrypoint |
