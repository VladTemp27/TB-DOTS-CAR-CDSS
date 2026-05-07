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
| `delta_min_importance_pct` | `50` | Importance percentile threshold for Delta (MICE) eligibility |
| `n_total` | auto | Set automatically from `len(df_static)` — do not override unless testing |

### Rescuing a clinically important sparse column

If a column was routed to Alpha but you know it carries clinical meaning, the recommended approach is to raise `alpha_hard_threshold`:

```python
# Rescues smear_microscopy (83.8%) and height_cm (82.1%)
# Columns above 90% (other, treatment_regimen, etc.) are still dropped
df_static, df_temporal = handle_missing_data(
    df_static, df_temporal,
    config={"alpha_hard_threshold": 0.90}
)
```

This routes rescued columns to Gamma — their actual values are preserved where observed, missing values are filled with the median/mode, and an `is_missing_{col}` indicator captures the structured absence.

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
    "ALPHA_DROP": 9,
    "BETA_LISTWISE": 0,
    "GAMMA_INDICATOR": 20,
    "DELTA_MICE": 1
  },
  "column_decisions": {
    "smear_microscopy": {
      "missing_rate_pct": 83.8,
      "mechanism": "MNAR",
      "temporal_pattern": "N/A",
      "pathway": "ALPHA_DROP",
      "pathway_rationale": "Dropped: 83.8% missing (exceeds hard threshold)."
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

Run on 599 patients, M0–M12 monthly observations.

### Pathway summary

| Pathway | Count | Notes |
|---|---|---|
| Alpha — dropped | 9 static columns | All >80% missing: `other`, `treatment_regimen`, `dat_supported`, `prior_history_of_tb`, `respiratory_rate`, `regimen_type_at_start_of_treatment`, `temperature`, `smear_microscopy`, `height_cm` |
| Beta — listwise | 0 columns | N=599 is too small; Beta was never viable at 15% threshold with 500-row floor |
| Gamma — indicator + fill | 20 static + 8 temporal columns | Includes all temporal features; 28 `is_missing_*` indicators added to model data |
| Delta — MICE | 1 static column | `bp_systolic` (69.3%) classified MAR |

### Final output shapes

| Array | Shape | Description |
|---|---|---|
| `X_temporal` | `(599, 13, 16)` | 13 timesteps × 16 features (8 measurements + 8 absence indicators) |
| `X_static` | `(599, 69)` | Static features after encoding |
| `X_combined_flat` | `(599, 277)` | Flattened temporal + static |

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
