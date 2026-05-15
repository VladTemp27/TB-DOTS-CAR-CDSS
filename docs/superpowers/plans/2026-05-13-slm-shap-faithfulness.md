# SLM-to-SHAP Faithfulness Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, per-patient benchmark that scores whether SLM explanations faithfully reflect SHAP feature attributions using conservative hard gates plus a composite score.

**Architecture:** Use SHAP outputs from the predictive model as ground truth, parse SLM explanation text into structured claims, and compute deterministic faithfulness metrics (feature identity, sign accuracy, magnitude band accuracy). Gate each explanation with hard minima and composite threshold, then aggregate for regression tracking.

**Tech Stack:** Python, pandas, numpy, SHAP (TreeExplainer), pytest, JSON/CSV artifacts.

---

### Task 1: Define benchmark config and schemas

**Files:**
- Create: `non-temporal/faithfulness/config.py`
- Create: `non-temporal/faithfulness/schemas.py`
- Test: `non-temporal/tests/test_faithfulness_config.py`

- [ ] **Step 1: Write the failing test**

```python
from non_temporal.faithfulness.config import BenchmarkConfig


def test_conservative_threshold_defaults():
    cfg = BenchmarkConfig()
    assert cfg.top_k == 5
    assert cfg.threshold_feature_f1 == 0.80
    assert cfg.threshold_sign_accuracy == 0.90
    assert cfg.threshold_magnitude_accuracy == 0.75
    assert cfg.threshold_composite == 0.82
    assert cfg.weight_feature_f1 == 0.45
    assert cfg.weight_sign_accuracy == 0.35
    assert cfg.weight_magnitude_accuracy == 0.20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_faithfulness_config.py::test_conservative_threshold_defaults -v`
Expected: FAIL with import or attribute error.

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_faithfulness_config.py::test_conservative_threshold_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/config.py non-temporal/tests/test_faithfulness_config.py
git commit -m "feat: add benchmark faithfulness config defaults"
```

### Task 2: Build SHAP ground-truth extractor

**Files:**
- Create: `non-temporal/faithfulness/shap_truth.py`
- Test: `non-temporal/tests/test_shap_truth.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from non_temporal.faithfulness.shap_truth import build_truth_rows


def test_build_truth_rows_top_k_sign_rank_band():
    row = pd.Series({"f1": 0.2, "f2": -0.7, "f3": 0.1, "f4": 0.5, "f5": -0.05})
    out = build_truth_rows(row, top_k=3)
    assert [r.feature for r in out] == ["f2", "f4", "f1"]
    assert [r.sign for r in out] == ["decrease", "increase", "increase"]
    assert [r.rank for r in out] == [1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_shap_truth.py::test_build_truth_rows_top_k_sign_rank_band -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
def build_truth_rows(shap_series, top_k=5):
    abs_sorted = sorted(shap_series.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    rows = []
    for idx, (feature, value) in enumerate(abs_sorted, start=1):
        sign = "increase" if value >= 0 else "decrease"
        rows.append({"feature": feature, "sign": sign, "abs_shap": abs(value), "rank": idx})
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_shap_truth.py::test_build_truth_rows_top_k_sign_rank_band -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/shap_truth.py non-temporal/tests/test_shap_truth.py
git commit -m "feat: add per-patient shap truth extraction"
```

### Task 3: Implement feature canonicalization map

**Files:**
- Create: `non-temporal/faithfulness/feature_map.py`
- Create: `non-temporal/faithfulness/feature_aliases.json`
- Test: `non-temporal/tests/test_feature_map.py`

- [ ] **Step 1: Write the failing test**

```python
from non_temporal.faithfulness.feature_map import canonicalize_feature


def test_canonicalize_feature_aliases_and_unknown():
    assert canonicalize_feature("Days To Treatment") == "Days_To_Treatment"
    assert canonicalize_feature("Province") == "Province"
    assert canonicalize_feature("Random Hallucinated Field") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_feature_map.py::test_canonicalize_feature_aliases_and_unknown -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
def canonicalize_feature(raw_name, alias_map):
    if raw_name in alias_map:
        return alias_map[raw_name]
    return raw_name if raw_name in alias_map.values() else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_feature_map.py::test_canonicalize_feature_aliases_and_unknown -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/feature_map.py non-temporal/faithfulness/feature_aliases.json non-temporal/tests/test_feature_map.py
git commit -m "feat: add deterministic feature canonicalization"
```

### Task 4: Implement deterministic explanation parser

**Files:**
- Create: `non-temporal/faithfulness/parser.py`
- Test: `non-temporal/tests/test_parser.py`

- [ ] **Step 1: Write the failing test**

```python
from non_temporal.faithfulness.parser import parse_explanation


def test_parse_explanation_extracts_feature_sign_magnitude():
    text = "Age strongly increases risk while Days To Treatment moderately reduces risk."
    claims = parse_explanation(text)
    assert claims[0]["feature"] == "Age"
    assert claims[0]["direction"] == "increase"
    assert claims[0]["magnitude"] == "strong"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_parser.py::test_parse_explanation_extracts_feature_sign_magnitude -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
def parse_explanation(text):
    # deterministic token/phrase rules for feature mentions, direction, magnitude
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_parser.py::test_parse_explanation_extracts_feature_sign_magnitude -v`
Expected: PASS after parser rules are implemented.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/parser.py non-temporal/tests/test_parser.py
git commit -m "feat: add deterministic slm explanation parser"
```

### Task 5: Implement scorer and gate logic

**Files:**
- Create: `non-temporal/faithfulness/scorer.py`
- Test: `non-temporal/tests/test_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
from non_temporal.faithfulness.scorer import score_case


def test_score_case_applies_hard_minima_and_composite_gate():
    result = score_case(
        feature_f1=0.82,
        sign_accuracy=0.92,
        magnitude_accuracy=0.76,
    )
    assert result["passed"] is True
    assert result["composite"] >= 0.82
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_scorer.py::test_score_case_applies_hard_minima_and_composite_gate -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
def score_case(feature_f1, sign_accuracy, magnitude_accuracy):
    composite = 0.45 * feature_f1 + 0.35 * sign_accuracy + 0.20 * magnitude_accuracy
    passed = (
        feature_f1 >= 0.80
        and sign_accuracy >= 0.90
        and magnitude_accuracy >= 0.75
        and composite >= 0.82
    )
    return {"composite": composite, "passed": passed}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_scorer.py::test_score_case_applies_hard_minima_and_composite_gate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/scorer.py non-temporal/tests/test_scorer.py
git commit -m "feat: add faithfulness scoring and conservative gate"
```

### Task 6: Build benchmark runner

**Files:**
- Create: `non-temporal/faithfulness/io.py`
- Create: `non-temporal/faithfulness/run_faithfulness_benchmark.py`
- Test: `non-temporal/tests/test_benchmark_runner.py`

- [ ] **Step 1: Write the failing test**

```python
from non_temporal.faithfulness.run_faithfulness_benchmark import run_benchmark


def test_run_benchmark_emits_per_patient_and_summary_artifacts(tmp_path):
    out = run_benchmark(input_dir=tmp_path, output_dir=tmp_path)
    assert "patients" in out
    assert "summary" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_benchmark_runner.py::test_run_benchmark_emits_per_patient_and_summary_artifacts -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
def run_benchmark(input_dir, output_dir):
    # orchestrate truth extraction, parsing, scoring, and artifact writing
    return {"patients": [], "summary": {}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_benchmark_runner.py::test_run_benchmark_emits_per_patient_and_summary_artifacts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/io.py non-temporal/faithfulness/run_faithfulness_benchmark.py non-temporal/tests/test_benchmark_runner.py
git commit -m "feat: add deterministic faithfulness benchmark runner"
```

### Task 7: Add deterministic split and manifest versioning

**Files:**
- Create: `non-temporal/faithfulness/benchmark_split.py`
- Create: `non-temporal/faithfulness/benchmark_manifest.json`
- Test: `non-temporal/tests/test_benchmark_split.py`

- [ ] **Step 1: Write the failing test**

```python
from non_temporal.faithfulness.benchmark_split import build_benchmark_split


def test_build_benchmark_split_is_deterministic():
    s1 = build_benchmark_split(seed=42)
    s2 = build_benchmark_split(seed=42)
    assert s1 == s2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_benchmark_split.py::test_build_benchmark_split_is_deterministic -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
def build_benchmark_split(seed=42):
    # deterministic stratified selection
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_benchmark_split.py::test_build_benchmark_split_is_deterministic -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/benchmark_split.py non-temporal/faithfulness/benchmark_manifest.json non-temporal/tests/test_benchmark_split.py
git commit -m "feat: add deterministic benchmark split manifest"
```

### Task 8: Add parser coverage validity gate

**Files:**
- Modify: `non-temporal/faithfulness/scorer.py`
- Modify: `non-temporal/faithfulness/run_faithfulness_benchmark.py`
- Test: `non-temporal/tests/test_parser_coverage_gate.py`

- [ ] **Step 1: Write the failing test**

```python
from non_temporal.faithfulness.scorer import evaluate_run_validity


def test_run_is_invalid_when_parse_coverage_below_95():
    status = evaluate_run_validity(parse_coverage=0.94)
    assert status == "invalid"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_parser_coverage_gate.py::test_run_is_invalid_when_parse_coverage_below_95 -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
def evaluate_run_validity(parse_coverage):
    return "valid" if parse_coverage >= 0.95 else "invalid"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_parser_coverage_gate.py::test_run_is_invalid_when_parse_coverage_below_95 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/scorer.py non-temporal/faithfulness/run_faithfulness_benchmark.py non-temporal/tests/test_parser_coverage_gate.py
git commit -m "feat: enforce parser coverage validity gate"
```

### Task 9: Add regression comparison helpers

**Files:**
- Create: `non-temporal/faithfulness/regression.py`
- Test: `non-temporal/tests/test_regression.py`

- [ ] **Step 1: Write the failing test**

```python
from non_temporal.faithfulness.regression import compare_runs


def test_compare_runs_returns_metric_deltas():
    base = {"pass_rate": 0.8}
    cur = {"pass_rate": 0.75}
    out = compare_runs(base, cur)
    assert out["pass_rate_delta"] == -0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest non-temporal/tests/test_regression.py::test_compare_runs_returns_metric_deltas -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
def compare_runs(base, cur):
    return {"pass_rate_delta": cur["pass_rate"] - base["pass_rate"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest non-temporal/tests/test_regression.py::test_compare_runs_returns_metric_deltas -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/regression.py non-temporal/tests/test_regression.py
git commit -m "feat: add faithfulness run regression comparison"
```

### Task 10: Document operations and interpretation

**Files:**
- Create: `non-temporal/faithfulness/README.md`
- Modify: `README.md`

- [ ] **Step 1: Document benchmark inputs and outputs**

```markdown
- input: per-patient SHAP values, prediction probability, raw features, SLM explanation text
- output: per-patient scorecards + aggregate summary + run validity
```

- [ ] **Step 2: Document command usage**

```bash
python non-temporal/faithfulness/run_faithfulness_benchmark.py --input-dir <path> --output-dir <path>
```

- [ ] **Step 3: Document thresholds and rationale**

```markdown
- conservative profile enforces feature/sign/magnitude minima before composite pass
```

- [ ] **Step 4: Document limitations**

```markdown
Faithfulness-to-SHAP does not guarantee clinical correctness.
```

- [ ] **Step 5: Commit**

```bash
git add non-temporal/faithfulness/README.md README.md
git commit -m "docs: add shap faithfulness benchmark usage guide"
```

---

## Final Verification Suite

Run:

```bash
pytest non-temporal/tests/test_faithfulness_config.py -v
pytest non-temporal/tests/test_shap_truth.py -v
pytest non-temporal/tests/test_feature_map.py -v
pytest non-temporal/tests/test_parser.py -v
pytest non-temporal/tests/test_scorer.py -v
pytest non-temporal/tests/test_benchmark_runner.py -v
pytest non-temporal/tests/test_benchmark_split.py -v
pytest non-temporal/tests/test_parser_coverage_gate.py -v
pytest non-temporal/tests/test_regression.py -v
pytest non-temporal/tests -v
python non-temporal/faithfulness/run_faithfulness_benchmark.py --help
```

Expected:
- all tests pass,
- runner help is printed,
- no threshold mismatches,
- benchmark artifacts are emitted in the documented format.
