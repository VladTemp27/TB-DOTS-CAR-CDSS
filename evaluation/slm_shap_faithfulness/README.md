# SLM-to-SHAP Faithfulness Benchmark

Evaluates whether SLM (Small Language Model) explanations faithfully reflect SHAP feature importance values.

## Overview

This benchmark measures three dimensions of explanation faithfulness:

| Metric | Description | Weight | Threshold |
|--------|-------------|--------|-----------|
| Feature F1 | SLM mentions the correct top-k features | 45% | ≥ 0.80 |
| Sign accuracy | Direction (increase/decrease) is correct | 35% | ≥ 0.90 |
| Magnitude accuracy | Magnitude (strong/moderate) is correct | 20% | ≥ 0.75 |

A weighted composite score ≥ 0.82 **and** all individual thresholds passing determines whether a case passes.

---

## Prerequisites

```bash
pip install pandas numpy pytest
```

All other dependencies (`json`, `re`, `pathlib`, `argparse`, `dataclasses`) are in the standard library.

---

## Input File Format

### Artifact mode (`case_*.json`)

Used when you have pre-computed SLM explanations and SHAP values stored as files.

```json
{
  "patient_id": "P001",
  "explanation": "Age strongly increases risk while Days To Treatment moderately reduces risk.",
  "shap_values": {
    "Age": 0.82,
    "Days_To_Treatment": -0.54,
    "Province": 0.31,
    "BMI": -0.18,
    "Sex": 0.09
  }
}
```

Place files in a directory and name them `case_<anything>.json`.

### Runtime mode (`runtime_*.json`)

Used when cases are generated at inference time. Same schema plus an optional `timestamp`.

```json
{
  "patient_id": "P002",
  "explanation": "Province decreases risk significantly.",
  "shap_values": { "Province": -0.65 },
  "timestamp": "2026-05-13T10:30:00"
}
```

Place files in a directory and name them `runtime_<anything>.json`.

---

## Running the Benchmark

### Basic run (artifact mode)

```bash
cd /path/to/slm-benchmark

python -m evaluation.slm_shap_faithfulness.run_benchmark \
  --input-dir /path/to/input/cases \
  --output-dir /path/to/output \
  --mode artifact
```

### Runtime mode

```bash
python -m evaluation.slm_shap_faithfulness.run_benchmark \
  --input-dir /path/to/runtime/cases \
  --output-dir /path/to/output \
  --mode runtime
```

### With regression comparison (compare against a previous run)

```bash
python -m evaluation.slm_shap_faithfulness.run_benchmark \
  --input-dir /path/to/cases \
  --output-dir /path/to/output \
  --mode artifact \
  --baseline-path /path/to/previous/output/results.json
```

The output will include a `regression` key with deltas like `pass_rate_delta`.

---

## Output

Results are written to `<output-dir>/results.json`:

```json
{
  "patients": [
    {
      "patient_id": "P001",
      "mode": "artifact",
      "feature_f1": 1.0,
      "sign_accuracy": 1.0,
      "magnitude_accuracy": 1.0,
      "composite": 1.0,
      "passed": true,
      "failure_tags": []
    }
  ],
  "summary": {
    "total_cases": 1,
    "passed": 1,
    "pass_rate": 1.0,
    "mode": "artifact"
  }
}
```

If `--baseline-path` is provided, a `regression` key is also included:

```json
{
  "regression": {
    "pass_rate_delta": 0.05,
    "total_cases_delta": 0.0
  }
}
```

### `failure_tags` reference

| Tag | Meaning |
|-----|---------|
| `low_feature_f1` | Feature overlap below 0.80 |
| `low_sign_accuracy` | Sign correctness below 0.90 |
| `low_magnitude_accuracy` | Magnitude correctness below 0.75 |
| `low_composite` | Weighted composite below 0.82 |

---

## Quick Example

```bash
# Create a sample input directory
mkdir -p /tmp/benchmark_demo

cat > /tmp/benchmark_demo/case_P001.json << 'EOF'
{
  "patient_id": "P001",
  "explanation": "Age strongly increases risk while Days To Treatment moderately reduces risk.",
  "shap_values": {"Age": 0.82, "Days_To_Treatment": -0.54}
}
EOF

# Run
python -m evaluation.slm_shap_faithfulness.run_benchmark \
  --input-dir /tmp/benchmark_demo \
  --output-dir /tmp/benchmark_demo \
  --mode artifact

# View results
cat /tmp/benchmark_demo/results.json
```

---

## Recognized Features

The parser recognizes these feature names in SLM explanation text:

| Text form | Canonical (SHAP key) |
|-----------|----------------------|
| `Days To Treatment` | `Days_To_Treatment` |
| `Treatment Category` | `Treatment_Category` |
| `Patient Category` | `Patient_Category` |
| `Smear Result` | `Smear_Result` |
| `Age` | `Age` |
| `Sex` | `Sex` |
| `BMI` | `BMI` |
| `Province` | `Province` |
| `Weight` | `Weight` |
| `Height` | `Height` |

Add new aliases in `feature_aliases.json`.

---

## Configuration

Default thresholds and weights are defined in `config.py` as a frozen dataclass:

```python
from evaluation.slm_shap_faithfulness.config import BenchmarkConfig

cfg = BenchmarkConfig(
    top_k=5,                        # top-k SHAP features to evaluate against
    threshold_feature_f1=0.80,
    threshold_sign_accuracy=0.90,
    threshold_magnitude_accuracy=0.75,
    threshold_composite=0.82,
    weight_feature_f1=0.45,
    weight_sign_accuracy=0.35,
    weight_magnitude_accuracy=0.20,
)
```

Pass a custom `cfg` when calling `run_benchmark()` programmatically.

---

## Running as a Library

```python
from evaluation.slm_shap_faithfulness.run_benchmark import run_benchmark

results = run_benchmark(
    input_dir="/path/to/cases",
    output_dir="/path/to/output",
    mode="artifact",           # or "runtime"
    baseline_path=None,        # optional: path to previous results.json
)

print(f"Pass rate: {results['summary']['pass_rate']:.1%}")
for patient in results["patients"]:
    status = "PASS" if patient["passed"] else "FAIL"
    print(f"  {patient['patient_id']}: {status} (composite={patient['composite']:.3f})")
```

---

## Running Tests

```bash
# From the repo root
pytest evaluation/slm_shap_faithfulness/tests/ -v
```

Expected output: 25 tests, all passing.
