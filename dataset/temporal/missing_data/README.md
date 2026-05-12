# Missing Data Handling — Diagnostic-First Pipeline

This package implements the missing data strategy described in the project research document *"Handling Missing Data: Imputation vs. Deletion"* for the TB-DOTS CAR CDSS preprocessing pipeline.

It replaces the original `impute_missing_mice()` function in `preprocessing.py` with a modular, audit-ready pipeline that diagnoses each column's missingness mechanism before deciding how to handle it.

---

## Why This Exists

The original `impute_missing_mice()` had several problems that motivated this rewrite:

| Problem | Impact |
|---|---|
| No pre-imputation diagnostics | MCAR/MAR/MNAR unknown — one-size-fits-all imputation is statistically incorrect |
| `BayesianRidge` + `sample_posterior=False` | Deterministic regression, not stochastic MICE — underestimates uncertainty |
| `max_iter=20` | Far below the FMI requirement (~70–100 for high-sparsity features per Von Hippel's rule) |
| Mode-fill for all categoricals | Collapses variance — 83% of patients getting the same imputed value destroys signal |
| No missing indicators | Drops the predictive signal of absence — in TB dropout data, a missing M9 weight IS informative |
| No audit trail | No record of what was imputed, dropped, or why |

---

## Architecture

```
dataset/temporal/missing_data/
├── __init__.py           ← Public API: handle_missing_data()
├── diagnostics.py        ← Missingness audit, MCAR test, mechanism classification
├── feature_importance.py ← RF importance ranking on available cases
├── strategy.py           ← Per-column pathway routing
├── apply.py              ← Execution: impute / delete / indicator per column
└── report.py             ← JSON audit trail
```

---

## The Four Pathways

Every column with missing data is routed to exactly one of four pathways, in order of aggressiveness:

| Pathway | When used | What it does |
|---|---|---|
| **Alpha** — Drop | >80% missing (hard), or >50% + bottom importance quartile (soft) | Column removed entirely from the dataset |
| **Beta** — Listwise | MCAR proven + <15% missing + N > 500 after deletion | Rows with missing values removed; patient IDs synced to temporal data |
| **Gamma** — Indicator + Fill | MNAR, temporal features, or moderate missingness | Binary `is_missing_{col}` flag added; NaN filled with forward-fill (temporal) or median/mode (static) |
| **Delta** — MICE | MAR + informative feature | Stochastic MICE via `ExtraTreesRegressor` (missForest-style); FMI-scaled iterations |

### Decision Logic

```
For each column with missing data:

  miss_rate > 80% AND not temporal?
    └─ ALPHA (hard threshold)

  miss_rate > 50% AND importance in bottom quartile AND not temporal?
    └─ ALPHA (soft threshold, only when importance scores are available)

  mechanism == MCAR AND miss_rate < 15% AND remaining_N > 500?
    └─ BETA (listwise deletion)

  mechanism == MAR AND (importance scores unavailable OR importance above median)?
    └─ DELTA (MICE imputation)

  otherwise (MNAR, temporal features, or uncertain):
    └─ GAMMA (indicator + fill)
```

### Why temporal features are never Alpha-dropped

Temporal features (monthly observations: weight, adherence, doses, etc.) have structurally high missingness — later months have fewer recorded values simply because patients hadn't reached those timepoints yet. The research document specifies GAMMA as the correct pathway for all temporal features because:

1. The absence of a month-9 weight reading often means the patient stopped attending — that IS the signal.
2. The `is_missing_{col}` indicator created by Gamma captures this dropout pattern as a binary feature.
3. Dropping temporal columns would eliminate time-series structure that LSTM/RNN models are specifically designed to exploit.

The 80% hard threshold still applies to temporal features — if a temporal column exceeds it, that's a data quality problem, not structural missingness.

---

## How to Use

### Basic usage (drop-in replacement)

```python
from missing_data import handle_missing_data

df_static, df_temporal = handle_missing_data(df_static, df_temporal)
```

Both DataFrames must be structured as described in `preprocessing.py`:
- `df_static`: one row per patient, columns include `patient_id` and all static baseline features
- `df_temporal`: long format with columns `[patient_id, month, ...feature columns...]`

### Adjusting thresholds

All routing thresholds can be overridden via the `config` dict:

```python
df_static, df_temporal = handle_missing_data(
    df_static, df_temporal,
    config={
        # Raise hard threshold to rescue clinically important sparse columns
        # e.g. smear_microscopy at 83.8% would survive at 0.90
        "alpha_hard_threshold": 0.90,

        # Raise soft threshold if you want to preserve more sparse features
        "alpha_soft_threshold": 0.60,

        # Tighten Beta eligibility (only use listwise for very low missingness)
        "beta_max_missing": 0.10,

        # Require larger remaining sample for Beta
        "beta_min_remaining_n": 550,

        # Only use MICE for features above the 60th importance percentile
        "delta_min_importance_pct": 60,
    }
)
```

Full list of config keys and their defaults:

| Key | Default | Description |
|---|---|---|
| `alpha_hard_threshold` | `0.80` | Drop column if missing rate exceeds this, regardless of importance |
| `alpha_soft_threshold` | `0.50` | Drop column if missing rate exceeds this AND importance is in bottom quartile |
| `beta_max_missing` | `0.15` | Maximum missing rate for Beta (listwise) to be viable |
| `beta_min_remaining_n` | `500` | Minimum rows that must remain after listwise deletion |
| `delta_min_importance_pct` | `50` | MAR columns must rank above this importance percentile to qualify for MICE. Lower values route more MAR columns to Delta. |
| `n_total` | auto | Set automatically from `len(df_static)` — do not override unless testing |

### Rescuing a clinically important sparse column

If a column was routed to Alpha but carries clinical meaning, raise `alpha_hard_threshold`:

```python
# Rescues smear_microscopy (83.8%) and height_cm (82.1%)
# Columns above 90% are still dropped
df_static, df_temporal = handle_missing_data(
    df_static, df_temporal,
    config={"alpha_hard_threshold": 0.90}
)
```

This routes rescued columns to Gamma — actual values preserved where observed, missing filled with median/mode, and an `is_missing_{col}` indicator captures structured absence.

### Routing more MAR columns to MICE

By default, a MAR column must rank above the 50th importance percentile to qualify for Delta. Lower `delta_min_importance_pct` to widen the net:

```python
# Routes all four MAR cardiovascular vitals to MICE
# (o2_sat, heart_rate, bp_systolic, bp_diastolic all confirmed MAR)
df_static, df_temporal = handle_missing_data(
    df_static, df_temporal,
    config={"delta_min_importance_pct": 30}
)
```

Use this when MAR columns cluster closely in importance and the default 50th-percentile cut would split clinically related measurements (e.g. bp_systolic and bp_diastolic) across different pathways based on a sub-noise importance difference.

### Specifying output directory

By default the audit report is written to `dataset/temporal/output/missing_data_report.json`. Override with:

```python
df_static, df_temporal = handle_missing_data(
    df_static, df_temporal,
    output_dir="path/to/your/output"
)
```

---

## Outputs

### Modified DataFrames

Both returned DataFrames have zero NaN values. Changes from the input:

- **Alpha columns** — removed from whichever DataFrame they belonged to
- **Beta rows** — rows where Beta-routed columns were missing are removed (patient IDs synced across both DataFrames)
- **Gamma columns** — NaNs filled; new binary `is_missing_{col}` columns added alongside each affected column
- **Delta columns** — NaNs filled via stochastic MICE; no indicator added (MAR means absence is not directly informative)

### Audit Report (`missing_data_report.json`)

A JSON file written to `output_dir` after every pipeline run. Structure:

```json
{
  "generated_at": "2025-...",
  "pipeline": "TB-DOTS CAR CDSS — Missing Data Handling",
  "dataset_info": { "n_rows": 599, "n_static_cols": ..., "n_temporal_cols": ... },
  "mcar_test": { "chi2": ..., "p_value": ..., "is_mcar": false, "note": "..." },
  "pathway_summary": {
    "ALPHA_DROP": 15,
    "BETA_LISTWISE": 0,
    "GAMMA_INDICATOR": 25,
    "DELTA_MICE": 4
  },
  "column_decisions": {
    "smear_microscopy": {
      "missing_rate_pct": 83.8,
      "mechanism": "MNAR",
      "temporal_pattern": "N/A",
      "pathway": "GAMMA_INDICATOR",
      "pathway_rationale": "Missing indicator + fill: mechanism=MNAR, moderate-to-high missingness (83.8%)."
    },
    ...
  },
  "post_imputation": {
    "remaining_static_nulls": 0,
    "remaining_temporal_nulls": 0,
    "n_rows": 599
  }
}
```

Every routing decision is recorded with the missing rate, diagnosed mechanism, temporal pattern, assigned pathway, and human-readable rationale. This is the audit trail required for methodological transparency in the research.

---

## Results on `combined_complete_dataset.csv`

Run on 599 patients, M0–M12 monthly observations. Config used: `alpha_hard_threshold=0.90`, `delta_min_importance_pct=30`.

### Alpha — Dropped (15 columns)

All MNAR. Hard-dropped (>90%) or soft-dropped (>50% + bottom importance quartile).

| Column | Missing | Drop reason |
|---|---|---|
| `risk_factors_for_drug_resistance_tuberculosis` | 100.0% | Hard (>90%) |
| `tuberculosis_culture` | 99.8% | Hard (>90%) |
| `other_lab_test` | 99.8% | Hard (>90%) |
| `others` | 99.7% | Hard (>90%) |
| `dat_supported_dup` | 96.2% | Hard (>90%) — duplicate column |
| `other` | 95.2% | Hard (>90%) |
| `tuberculin_skin_test` | 93.5% | Hard (>90%) |
| `regimen_type_at_end_of_treatment` | 93.3% | Hard (>90%) |
| `prior_history_of_tb` | 91.7% | Hard (>90%) |
| `dat_supported` | 91.7% | Hard (>90%) |
| `respiratory_rate` | 87.8% | Hard (>90%) |
| `regimen_type_at_start_of_treatment` | 87.2% | Hard (>90%) |
| `regimen_type_at_6th_month_of_treatment` | 86.5% | Hard (>90%) |
| `temperature` | 84.8% | Hard (>90%) |
| `blood_pressure` | 67.3% | Soft (>50% + low importance) — raw string already parsed into `bp_systolic`/`bp_diastolic` |

### Beta — Listwise (0 columns)

Not triggered. N=599 is too small and MCAR not confirmed on this dataset.

### Gamma — Indicator + Fill (25 columns)

All MNAR. Each column gets a binary `is_missing_{col}` indicator alongside the filled value.

| Column | Missing | Notes |
|---|---|---|
| `smear_tb_lamp` | 99.3% | Temporal feature — absence = signal |
| `xpert_mtb_rif` | 98.7% | Temporal feature — absence = signal |
| `height` | 92.1% | Temporal feature |
| `weight` | 90.8% | Temporal feature |
| `smear_microscopy` | 83.8% | Clinically important; threshold raised to 0.90 to preserve |
| `height_cm` | 82.1% | Relevant for dosing; importance above bottom quartile |
| `weight_kg` | 79.5% | Static baseline weight |
| `pct_adherence` | 73.7% | Temporal feature |
| `monthly_missed_doses` | 66.5% | Temporal feature |
| `cumulative_doses_taken` | 60.0% | Temporal feature |
| `monthly_doses_taken` | 51.8% | Temporal feature |
| `drug_resistance_bacteriological_status` | 49.6% | |
| `name_of_treatment_unit` | 48.4% | |
| `treatment_regimen` | 45.9% | |
| `co_morbidities` | 42.2% | |
| `chest_x_ray_at_case_notification` | 35.6% | |
| `civil_status` | 28.2% | |
| `outcome` | 28.1% | Target variable — indicator is a useful meta-feature |
| `name_of_diagnosing_facility` | 25.0% | |
| `case_registration_group` | 21.0% | |
| `nationality` | 19.9% | |
| `diagnosis` | 16.7% | |
| `bacteriologic_status` | 15.4% | |
| `sex` | 5.7% | |
| `age` | 3.0% | |

### Delta — MICE (4 columns)

All MAR — missingness explained by other observed variables (patients who miss a visit miss all vitals together). All four are cardiovascular/respiratory vitals measured on the same clinical encounter.

| Column | Missing | Importance percentile |
|---|---|---|
| `o2_sat` | 79.8% | 54.5th |
| `heart_rate` | 76.5% | 45.5th |
| `bp_systolic` | 69.3% | 50.0th |
| `bp_diastolic` | 69.3% | 36.4th |

`bp_systolic` and `bp_diastolic` are the same BP reading — they must be treated identically. `delta_min_importance_pct=30` ensures all four route to MICE rather than being split by a sub-noise importance difference.

### Pipeline outputs

| File | Description |
|---|---|
| `output/cleaned_human_readable.csv` | 599 rows × 269 columns, original clinical units |
| `output/missing_data_report.json` | Per-column audit trail |

**Rows retained: 599 / 599 (zero deleted)**

---

## Key Design Decisions and Rationale

### No `bfill()` in temporal imputation

Forward-fill (`ffill`) is applied within each patient's time series before any median/MICE fallback. Backward-fill (`bfill`) is explicitly not used. Reason: `bfill` would fill a missing Month 0 weight with the Month 2 reading — a future measurement the model cannot have at inference time, causing direct temporal data leakage. For start-of-sequence gaps, the global median is used instead.

### `sample_posterior=False` with ExtraTreesRegressor

`sklearn`'s `IterativeImputer` supports `sample_posterior=True` for Bayesian estimators to inject stochasticity. With `ExtraTreesRegressor`, stochasticity comes from random feature subsets and bootstrap sampling within the trees — `sample_posterior=True` is incompatible with non-Bayesian estimators and raises an error. The ExtraTrees approach (missForest-style) is preferred over `BayesianRidge` because it captures non-linear relationships between TB clinical variables without requiring standardization.

### FMI-scaled `max_iter`

Von Hippel's rule: the number of MICE iterations should approximately equal the fraction of missing information (FMI), which is approximately `missing_rate × 100` for a single missing variable. The original pipeline used `max_iter=20` for all columns. This pipeline scales it: `max(20, min(100, int(missing_rate × 100)))`. A column with 70% missingness gets 70 iterations; one with 20% gets 20.

### Available-case analysis for feature importance

The RF importance ranking used to gate Alpha and Delta decisions requires a target variable (`outcome`) and numeric features. With high feature-level missingness, requiring zero NaN across all features leaves zero usable rows. Instead, the pipeline uses available-case analysis: rows where `outcome` is observed (even if features have NaN), with median fill applied to the features only for the purpose of importance ranking. This is a coarse imputation intentionally — the ranking signal matters more than purity here.

### `is_mcar_global` from Little's test

Little's MCAR test produces a single dataset-level verdict. In this dataset, the test reliably returns `is_mcar=False` (MNAR/MAR patterns dominate). This means MCAR-gated Beta pathway is effectively inactive for most runs — which is correct, because deleting rows from a 599-patient TB dataset under MNAR would introduce systematic bias.

---

## Module Reference

### `handle_missing_data(df_static, df_temporal, config=None, output_dir=...)`

The only function you need to call. Orchestrates all five phases and returns `(df_static, df_temporal)` with zero NaN values.

### `diagnostics.py`

| Function | Returns |
|---|---|
| `compute_missing_rates(df)` | `dict[col, float]` — missing fraction per column |
| `littles_mcar_test(df)` | `dict` with `chi2`, `p_value`, `is_mcar`, `note` |
| `detect_temporal_pattern(df_temporal)` | `dict[col, "MNAR_dropout" \| "random"]` — Spearman correlation of monthly miss rate vs month |
| `compute_missingness_correlations(df)` | `dict[col, float]` — max point-biserial correlation of `is_missing_col` with observed columns |
| `classify_mechanism(col, miss_rate, is_mcar_global, temporal_pattern, miss_corrs)` | `"MCAR" \| "MAR" \| "MNAR"` |

### `strategy.py`

| Symbol | Description |
|---|---|
| `Pathway` | Enum: `ALPHA_DROP`, `BETA_LISTWISE`, `GAMMA_INDICATOR`, `DELTA_MICE` |
| `DEFAULT_CONFIG` | Dict of default threshold values |
| `route_columns(diagnostics_result, importance_scores, config, temporal_cols)` | `dict[col, Pathway]` |

### `apply.py`

| Function | Description |
|---|---|
| `apply_alpha(df_static, df_temporal, cols)` | Drops columns from whichever DataFrame they belong to |
| `apply_beta(df_static, df_temporal, cols)` | Listwise deletion; cascades removed patient IDs to temporal data |
| `apply_gamma(df_static, df_temporal, cols)` | Indicator + fill; forward-fill for temporal, median/mode for static |
| `apply_delta(df_static, df_temporal, cols, missing_rates)` | MICE with ExtraTreesRegressor; forward-fill temporal first |

---

## Dependencies

No new dependencies beyond what `preprocessing.py` already uses:

| Library | Usage |
|---|---|
| `scikit-learn >= 1.3` | `IterativeImputer`, `ExtraTreesRegressor`, `RandomForestClassifier` |
| `scipy >= 1.10` | `chi2.cdf` for Little's MCAR test, `spearmanr` for dropout detection |
| `numpy >= 1.24` | Array operations throughout |
| `pandas >= 2.0` | DataFrame manipulation throughout |
