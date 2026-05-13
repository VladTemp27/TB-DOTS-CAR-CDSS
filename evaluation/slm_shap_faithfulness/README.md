# SLM-to-SHAP Faithfulness Benchmark

Evaluates whether SLM (Small Language Model) explanations faithfully reflect SHAP feature importance values.

## Overview

This benchmark measures three dimensions of explanation faithfulness:
- **Feature F1**: Whether the SLM mentions the correct features
- **Sign accuracy**: Whether direction (increase/decrease) is correct
- **Magnitude accuracy**: Whether the magnitude (strong/moderate/weak) is correct

A composite score (weighted average) determines whether each case passes.

## Usage

```bash
python -m evaluation.slm_shap_faithfulness.run_benchmark \
  --input-dir /path/to/cases \
  --output-dir /path/to/output \
  --mode artifact
```

## Modes

- `artifact` — loads pre-computed JSON files matching `case_*.json`
- `runtime` — loads inference-time JSON files matching `runtime_*.json`

## Configuration

See `config.py` for threshold and weight defaults (`BenchmarkConfig`).

## Tests

```bash
pytest evaluation/slm_shap_faithfulness/tests/
```
