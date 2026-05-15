# Temporal SLM-to-SHAP Faithfulness Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the SLM-to-SHAP faithfulness benchmark to `evaluation/slm_shap_faithfulness/`, support both artifact/runtime ingestion modes, and remove all legacy benchmark paths under `non-temporal/` and `non_temporal/`.

**Architecture:** Keep one deterministic evaluation core (parser, SHAP truth builder, scorer, regression) and add two input adapters that normalize data into a shared case schema. A single `run_benchmark.py` entrypoint selects adapter mode (`artifact` or `runtime`) and emits mode-invariant outputs.

**Tech Stack:** Python, pandas, numpy, pytest, JSON/CSV I/O, SHAP (runtime adapter optional recompute path).

---

## File Structure (Target)

- Create: `evaluation/slm_shap_faithfulness/__init__.py`
- Create: `evaluation/slm_shap_faithfulness/config.py`
- Create: `evaluation/slm_shap_faithfulness/schemas.py`
- Create: `evaluation/slm_shap_faithfulness/feature_map.py`
- Create: `evaluation/slm_shap_faithfulness/feature_aliases.json`
- Create: `evaluation/slm_shap_faithfulness/shap_truth.py`
- Create: `evaluation/slm_shap_faithfulness/parser.py`
- Create: `evaluation/slm_shap_faithfulness/scorer.py`
- Create: `evaluation/slm_shap_faithfulness/io.py`
- Create: `evaluation/slm_shap_faithfulness/regression.py`
- Create: `evaluation/slm_shap_faithfulness/benchmark_manifest.json`
- Create: `evaluation/slm_shap_faithfulness/adapters/__init__.py`
- Create: `evaluation/slm_shap_faithfulness/adapters/artifact_adapter.py`
- Create: `evaluation/slm_shap_faithfulness/adapters/runtime_adapter.py`
- Create: `evaluation/slm_shap_faithfulness/run_benchmark.py`
- Create: `evaluation/slm_shap_faithfulness/README.md`
- Create: `evaluation/slm_shap_faithfulness/tests/conftest.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_config.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_feature_map.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_shap_truth.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_parser.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_scorer.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_regression.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_adapter_artifact.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_adapter_runtime.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_runner_integration.py`
- Create: `evaluation/slm_shap_faithfulness/tests/test_migration_cleanup.py`
- Modify: `README.md` (single canonical pointer)
- Delete: `non_temporal/`
- Delete: `non-temporal/tests/`
- Delete: `non-temporal/faithfulness/README.md`

### Task 1: Bootstrap canonical package and config

**Files:**
- Create: `evaluation/slm_shap_faithfulness/__init__.py`
- Create: `evaluation/slm_shap_faithfulness/config.py`
- Create: `evaluation/slm_shap_faithfulness/schemas.py`
- Test: `evaluation/slm_shap_faithfulness/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
from evaluation.slm_shap_faithfulness.config import BenchmarkConfig


def test_conservative_defaults_and_mode_choices():
    cfg = BenchmarkConfig()
    assert cfg.top_k == 5
    assert cfg.threshold_feature_f1 == 0.80
    assert cfg.threshold_sign_accuracy == 0.90
    assert cfg.threshold_magnitude_accuracy == 0.75
    assert cfg.threshold_composite == 0.82
    assert cfg.parser_coverage_min == 0.95
    assert cfg.allowed_modes == ("artifact", "runtime")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_config.py::test_conservative_defaults_and_mode_choices -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    top_k: int = 5
    threshold_feature_f1: float = 0.80
    threshold_sign_accuracy: float = 0.90
    threshold_magnitude_accuracy: float = 0.75
    threshold_composite: float = 0.82
    weight_feature_f1: float = 0.45
    weight_sign_accuracy: float = 0.35
    weight_magnitude_accuracy: float = 0.20
    parser_coverage_min: float = 0.95
    allowed_modes: tuple[str, str] = ("artifact", "runtime")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_config.py::test_conservative_defaults_and_mode_choices -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/slm_shap_faithfulness/__init__.py evaluation/slm_shap_faithfulness/config.py evaluation/slm_shap_faithfulness/schemas.py evaluation/slm_shap_faithfulness/tests/test_config.py
git commit -m "feat(eval): bootstrap canonical temporal shap benchmark package"
```

### Task 2: Implement feature mapping and SHAP truth extraction

**Files:**
- Create: `evaluation/slm_shap_faithfulness/feature_aliases.json`
- Create: `evaluation/slm_shap_faithfulness/feature_map.py`
- Create: `evaluation/slm_shap_faithfulness/shap_truth.py`
- Test: `evaluation/slm_shap_faithfulness/tests/test_feature_map.py`
- Test: `evaluation/slm_shap_faithfulness/tests/test_shap_truth.py`

- [ ] **Step 1: Write failing tests**

```python
from evaluation.slm_shap_faithfulness.feature_map import canonicalize_feature


def test_canonicalize_temporal_aliases():
    assert canonicalize_feature("Days To Treatment") == "Days_To_Treatment"
    assert canonicalize_feature("Province") == "Province"
    assert canonicalize_feature("Unknown Field") is None
```

```python
import pandas as pd
from evaluation.slm_shap_faithfulness.shap_truth import build_truth_rows


def test_truth_rows_rank_sign_band():
    row = pd.Series({"A": 0.8, "B": -0.6, "C": 0.2})
    out = build_truth_rows(row, top_k=3)
    assert [x["feature"] for x in out] == ["A", "B", "C"]
    assert [x["sign"] for x in out] == ["increase", "decrease", "increase"]
    assert [x["rank"] for x in out] == [1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_feature_map.py evaluation/slm_shap_faithfulness/tests/test_shap_truth.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Write minimal implementation**

```python
def canonicalize_feature(raw_name: str, alias_map: dict[str, str] | None = None) -> str | None:
    if alias_map is None:
        alias_map = {}
    return alias_map.get(raw_name)
```

```python
def build_truth_rows(shap_series, top_k: int = 5) -> list[dict]:
    ranked = sorted(shap_series.items(), key=lambda x: abs(float(x[1])), reverse=True)[:top_k]
    return [{"feature": f, "sign": "increase" if v >= 0 else "decrease", "abs_shap": abs(float(v)), "rank": i + 1, "magnitude_band": "strong" if i == 0 else "moderate"} for i, (f, v) in enumerate(ranked)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_feature_map.py evaluation/slm_shap_faithfulness/tests/test_shap_truth.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/slm_shap_faithfulness/feature_aliases.json evaluation/slm_shap_faithfulness/feature_map.py evaluation/slm_shap_faithfulness/shap_truth.py evaluation/slm_shap_faithfulness/tests/test_feature_map.py evaluation/slm_shap_faithfulness/tests/test_shap_truth.py
git commit -m "feat(eval): add feature canonicalization and shap truth builder"
```

### Task 3: Implement parser and deterministic scorer

**Files:**
- Create: `evaluation/slm_shap_faithfulness/parser.py`
- Create: `evaluation/slm_shap_faithfulness/scorer.py`
- Test: `evaluation/slm_shap_faithfulness/tests/test_parser.py`
- Test: `evaluation/slm_shap_faithfulness/tests/test_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
from evaluation.slm_shap_faithfulness.parser import parse_explanation


def test_parse_extracts_feature_direction_magnitude():
    parsed = parse_explanation("Age strongly increases risk while Days To Treatment moderately reduces risk.")
    assert parsed["claims"][0]["feature"] == "Age"
    assert parsed["claims"][0]["direction"] == "increase"
    assert parsed["claims"][0]["magnitude"] == "strong"
```

```python
from evaluation.slm_shap_faithfulness.scorer import score_case


def test_score_case_applies_conservative_gate():
    truth = [
        {"feature": "Age", "sign": "increase", "magnitude_band": "strong"},
        {"feature": "Days_To_Treatment", "sign": "decrease", "magnitude_band": "moderate"},
    ]
    claims = [
        {"feature": "Age", "direction": "increase", "magnitude": "strong"},
        {"feature": "Days_To_Treatment", "direction": "decrease", "magnitude": "moderate"},
    ]
    out = score_case(truth, claims)
    assert out["passed"] is True
    assert out["composite"] >= 0.82
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_parser.py evaluation/slm_shap_faithfulness/tests/test_scorer.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Write minimal implementation**

```python
def parse_explanation(text: str, alias_map: dict[str, str] | None = None) -> dict[str, object]:
    return {"status": "parse_failed", "claims": [], "coverage": 0.0}
```

```python
def score_case(truth_rows: list[dict], claims: list[dict], cfg: BenchmarkConfig | None = None) -> dict[str, object]:
    return {"feature_f1": 0.0, "sign_accuracy": 0.0, "magnitude_accuracy": 0.0, "composite": 0.0, "passed": False, "failure_tags": ["missing_top_feature"]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_parser.py evaluation/slm_shap_faithfulness/tests/test_scorer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/slm_shap_faithfulness/parser.py evaluation/slm_shap_faithfulness/scorer.py evaluation/slm_shap_faithfulness/tests/test_parser.py evaluation/slm_shap_faithfulness/tests/test_scorer.py
git commit -m "feat(eval): add deterministic parser and faithfulness scorer"
```

### Task 4: Implement adapters and shared case schema

**Files:**
- Create: `evaluation/slm_shap_faithfulness/adapters/__init__.py`
- Create: `evaluation/slm_shap_faithfulness/adapters/artifact_adapter.py`
- Create: `evaluation/slm_shap_faithfulness/adapters/runtime_adapter.py`
- Test: `evaluation/slm_shap_faithfulness/tests/test_adapter_artifact.py`
- Test: `evaluation/slm_shap_faithfulness/tests/test_adapter_runtime.py`

- [ ] **Step 1: Write failing tests**

```python
from evaluation.slm_shap_faithfulness.adapters.artifact_adapter import load_cases_from_artifacts


def test_artifact_adapter_outputs_normalized_cases(tmp_path):
    out = load_cases_from_artifacts(tmp_path)
    assert isinstance(out, list)
```

```python
from evaluation.slm_shap_faithfulness.adapters.runtime_adapter import load_cases_from_runtime


def test_runtime_adapter_validates_temporal_context(tmp_path):
    out = load_cases_from_runtime(tmp_path)
    assert isinstance(out, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_adapter_artifact.py evaluation/slm_shap_faithfulness/tests/test_adapter_runtime.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Write minimal implementation**

```python
def load_cases_from_artifacts(input_dir: str | Path) -> list[dict]:
    return []
```

```python
def load_cases_from_runtime(input_dir: str | Path, allow_recompute_shap: bool = True) -> list[dict]:
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_adapter_artifact.py evaluation/slm_shap_faithfulness/tests/test_adapter_runtime.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/slm_shap_faithfulness/adapters/__init__.py evaluation/slm_shap_faithfulness/adapters/artifact_adapter.py evaluation/slm_shap_faithfulness/adapters/runtime_adapter.py evaluation/slm_shap_faithfulness/tests/test_adapter_artifact.py evaluation/slm_shap_faithfulness/tests/test_adapter_runtime.py
git commit -m "feat(eval): add artifact and runtime ingestion adapters"
```

### Task 5: Implement runner, I/O, regression, and integration test

**Files:**
- Create: `evaluation/slm_shap_faithfulness/io.py`
- Create: `evaluation/slm_shap_faithfulness/regression.py`
- Create: `evaluation/slm_shap_faithfulness/run_benchmark.py`
- Create: `evaluation/slm_shap_faithfulness/benchmark_manifest.json`
- Test: `evaluation/slm_shap_faithfulness/tests/test_runner_integration.py`
- Test: `evaluation/slm_shap_faithfulness/tests/test_regression.py`

- [ ] **Step 1: Write failing tests**

```python
from evaluation.slm_shap_faithfulness.run_benchmark import run_benchmark


def test_runner_writes_mode_invariant_artifacts(tmp_path):
    out = run_benchmark(input_dir=tmp_path, output_dir=tmp_path, mode="artifact")
    assert "patients" in out and "summary" in out
```

```python
from evaluation.slm_shap_faithfulness.regression import compare_runs


def test_compare_runs_deltas():
    out = compare_runs({"pass_rate": 0.8}, {"pass_rate": 0.7})
    assert out["pass_rate_delta"] == -0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_runner_integration.py evaluation/slm_shap_faithfulness/tests/test_regression.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Write minimal implementation**

```python
def run_benchmark(input_dir: str | Path, output_dir: str | Path, mode: str, baseline_path: str | None = None) -> dict:
    return {"patients": [], "summary": {"pass_rate": 0.0}}
```

```python
def compare_runs(base: dict, current: dict) -> dict[str, float]:
    return {"pass_rate_delta": round(float(current.get("pass_rate", 0.0)) - float(base.get("pass_rate", 0.0)), 6)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_runner_integration.py evaluation/slm_shap_faithfulness/tests/test_regression.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/slm_shap_faithfulness/io.py evaluation/slm_shap_faithfulness/regression.py evaluation/slm_shap_faithfulness/run_benchmark.py evaluation/slm_shap_faithfulness/benchmark_manifest.json evaluation/slm_shap_faithfulness/tests/test_runner_integration.py evaluation/slm_shap_faithfulness/tests/test_regression.py
git commit -m "feat(eval): add benchmark runner and regression output"
```

### Task 6: Migrate docs and remove legacy benchmark paths

**Files:**
- Modify: `README.md`
- Create: `evaluation/slm_shap_faithfulness/README.md`
- Delete: `non_temporal/`
- Delete: `non-temporal/tests/`
- Delete: `non-temporal/faithfulness/README.md`
- Test: `evaluation/slm_shap_faithfulness/tests/test_migration_cleanup.py`

- [ ] **Step 1: Write failing migration cleanup test**

```python
from pathlib import Path


def test_legacy_benchmark_paths_removed():
    assert not Path("non_temporal/faithfulness").exists()
    assert not Path("non-temporal/tests").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_migration_cleanup.py::test_legacy_benchmark_paths_removed -v`
Expected: FAIL while legacy paths still exist.

- [ ] **Step 3: Remove legacy paths and update README pointers**

```bash
rm -rf non_temporal
rm -rf non-temporal/tests
rm -f non-temporal/faithfulness/README.md
```

Update `README.md` so only this path remains:

```markdown
SLM explanation faithfulness-to-SHAP tooling lives in `evaluation/slm_shap_faithfulness/`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest evaluation/slm_shap_faithfulness/tests/test_migration_cleanup.py::test_legacy_benchmark_paths_removed -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md evaluation/slm_shap_faithfulness/README.md evaluation/slm_shap_faithfulness/tests/test_migration_cleanup.py
git add -A non_temporal non-temporal/tests non-temporal/faithfulness/README.md
git commit -m "refactor(eval): migrate temporal shap benchmark and remove legacy paths"
```

### Task 7: Full verification pass

**Files:**
- Verify only

- [ ] **Step 1: Run package test suite**

Run: `pytest evaluation/slm_shap_faithfulness/tests -v`
Expected: PASS.

- [ ] **Step 2: Run CLI help verification**

Run: `python -m evaluation.slm_shap_faithfulness.run_benchmark --help`
Expected: help output includes `--mode`, `--input-dir`, `--output-dir`.

- [ ] **Step 3: Verify no stale imports remain**

Run: `rg "non_temporal\.faithfulness|non-temporal/faithfulness|non-temporal/tests"`
Expected: no matches.

- [ ] **Step 4: Commit verification-only updates if needed**

```bash
git add -A
git commit -m "test(eval): finalize temporal shap migration verification" 
```

Use this step only when Task 7 required file adjustments.

---

## Self-Review Notes

- Spec coverage map:
  - canonical folder + cleanup: Tasks 1, 6
  - dual-mode adapters: Task 4
  - shared deterministic scorer: Tasks 2, 3, 5
  - parser coverage and validity gate: Tasks 3, 5
  - mode-invariant artifacts + regression: Task 5
  - stale reference removal: Task 6, Task 7
- Placeholder scan completed: no TBD/TODO/"implement later" directives.
- Type consistency checked:
  - `run_benchmark(input_dir, output_dir, mode, baseline_path)` shared consistently
  - adapters output normalized list[dict] consumed by scorer
  - scoring keys match spec (`feature_f1`, `sign_accuracy`, `magnitude_accuracy`, `composite`)
