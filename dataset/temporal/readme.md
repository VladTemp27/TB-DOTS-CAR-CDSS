# Preprocessing Pipeline — Technical Documentation

## How `preprocessing.py` Meets Every Requirement

---

### 1. Remove Identifiers

**Where:** Stage 2 (`remove_identifiers`) + Stage 9 (`encode_features`)

| Phase | What happens |
|-------|-------------|
| **Phase A** | Stage 2 **flags** 4 identifier columns (`no`, `source_file`, `name_of_diagnosing_facility`, `name_of_treatment_unit`) but **retains** them in `cleaned_human_readable.csv` for traceability — per the no-drop policy |
| **Phase B** | Stage 9 **removes** those same 4 columns from the model-ready DataFrame via `non_feature_cols` list + `df_static.drop(columns=cols_to_drop)`. They never appear in `static_features.csv`, `X_static.npy`, or any tensor |

**Result:** Identifiers exist only in the human-readable CSV. Zero identifiers in any model output.

---

### 2. Handle Missing Values (Diagnostic-First Pipeline)

**Where:** Stage 7 (`handle_missing_data` from `missing_data/`)

> **Note:** Stage 7 was rewritten from a monolithic `impute_missing_mice()` to a modular diagnostic-first pipeline. See [`missing_data/README.md`](missing_data/README.md) for full documentation.

The pipeline diagnoses each column's missingness mechanism before deciding how to handle it. Every column is routed to one of four pathways:

| Pathway | Condition | Action |
|---|---|---|
| **Alpha — Drop** | >80% missing (hard), or >50% + low importance (soft) | Column removed |
| **Beta — Listwise** | MCAR + <15% missing + N > 500 after deletion | Rows removed, patient IDs synced |
| **Gamma — Indicator + Fill** | MNAR, temporal features, or moderate missingness | `is_missing_{col}` flag added; NaN filled with forward-fill (temporal) or median/mode (static) |
| **Delta — MICE** | MAR + informative feature | Stochastic MICE via `ExtraTreesRegressor` (missForest-style), FMI-scaled iterations |

**Results on `combined_complete_dataset.csv` (599 patients):**

| Pathway | Count |
|---|---|
| Alpha — dropped | 9 static columns (all >80% missing) |
| Beta — listwise | 0 (N=599 too small; MCAR not confirmed) |
| Gamma — indicator + fill | 20 static + 8 temporal columns |
| Delta — MICE | 1 static column (`bp_systolic`, 69.3%, MAR) |

**Rows retained: 599 / 599** — zero patients removed.

**Key design constraints:**
- Temporal features are never Alpha-dropped (structural missingness; absence IS the signal)
- No `bfill()` on temporal data — forward-fill only (no future data leakage)
- `max_iter` scaled to FMI: `max(20, min(100, int(miss_rate × 100)))` per Von Hippel's rule

**Post-imputation sanity clipping** (unchanged from original):
- Weight: [10, 180] kg | Height: [80, 220] cm | Monthly doses: [0, 31] | Smear/Xpert: [0, 1]

**Audit trail:** Every routing decision is written to `dataset/temporal/output/missing_data_report.json`.

**Result:** 0 missing values across all outputs.

---

### 3. Feature Encoding

**Where:** Stage 9 (`encode_features`)

```python
# One-hot encoding via pandas get_dummies
df_static_encoded = pd.get_dummies(
    df_static, columns=cat_cols_static,
    prefix_sep="_", dummy_na=False, dtype=float
)
```

| What gets encoded | Example transformation |
|-------------------|----------------------|
| `sex` → `sex_Male`, `sex_Female` | "Male" → 1.0, 0.0 |
| `outcome` → `outcome_Cured`, `outcome_Died`, `outcome_Treatment Completed`, etc. | "Cured" → 1.0, 0.0, 0.0, ... |
| `diagnosis` → `diagnosis_TB Disease`, `diagnosis_TB Infection` | "TB Disease" → 1.0, 0.0 |
| 16 categorical columns total | All converted to binary indicators |

**Additional handling:**
- Any remaining `object` columns not in the curated list are **dropped** from model data
- Date columns are converted to **ordinal integers** (days since 1970-01-01) so they remain numeric
- Temporal categoricals are also one-hot encoded if any exist

**Result:** 0 object/datetime columns in model-ready data. All features are `float64`.

---

### 4. Scaling/Normalization

**Where:** Stage 10 (`scale_features`)

```python
# StandardScaler on static numerics
scaler_static = StandardScaler()
if static_num_cols:
    df_static[static_num_cols] = scaler_static.fit_transform(
        df_static[static_num_cols]
    )
```

| Dataset | Columns Scaled | Method |
|---------|---------------|--------|
| **Static** | 9 baseline clinical numerics: `age`, `weight_kg`, `height_cm`, `bp_systolic`, `bp_diastolic`, `heart_rate`, `respiratory_rate`, `temperature`, `o2_sat` | `StandardScaler` (zero mean, unit variance) |
| **Temporal** | Continuous temporal features with >2 unique values (doses, adherence, weight, height — excludes binary smear/xpert) | `StandardScaler` |

**Why StandardScaler?** It works well with RNN/LSTM models and preserves outlier signal for the subsequent capping stage. The scalers are returned so inverse transforms are possible.

**Data leakage prevention:** For future train/test splitting, the scaler should be fit on training data only.

**Separation guarantee:** Scaling happens **only in Phase B**. Phase A's `cleaned_human_readable.csv` has age mean ~45.6 (real years), not ~0.0. This is validated by `_validate_cleaned_human_readable()` which checks `age_mean > 1.0`.

---

### 5. Outlier Detection

**Where:** Stage 11 (`detect_and_cap_outliers`)

Two methods applied sequentially:

```python
# IQR method
def cap_outliers_iqr(series, col_name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((series < lower) | (series > upper)).sum()
    capped = series.clip(lower=lower, upper=upper)
    return capped, n_outliers
```

| Method | Threshold | Applied to |
|--------|-----------|-----------|
| **IQR** | Q1 − 1.5×IQR to Q3 + 1.5×IQR | Static: 9 vital columns (`OUTLIER_COLUMNS`). Temporal: all continuous numeric columns |
| **Z-score** | \|z\| > 3.0 (applied after IQR) | Same columns, catching any remaining extremes |

Values are **capped (winsorized)**, not removed — no patients are deleted.

**Note:** This runs on already-scaled data (post-Stage 10), so z-score of 3.0 = ±3 standard deviations from the mean.

---

### 6. Schema Validation

**Where:** Stage 1 (`load_and_validate_schema`) + Stage 13a (`validate_model_ready`)

**Stage 1 — Input validation:**

```python
dupes = df.columns[df.columns.duplicated()].tolist()      # duplicate columns
all_null = df.columns[df.isnull().all()].tolist()          # entirely null columns
print(f"  Data types: {df.dtypes.value_counts().to_dict()}")  # dtype distribution
```

**Stage 13a — Output validation (6 checks):**

| Check | What it verifies | How |
|-------|-----------------|-----|
| 1 | No identifier columns in model data | Scans column names for `name`, `no.`, `source`, `facility_name` |
| 2 | No NaN in temporal tensor | `np.isnan(X_temporal).sum() == 0` |
| 3 | No NaN in static array | `np.isnan(X_static).sum() == 0` |
| 4 | No object/datetime dtypes | `df_static.select_dtypes(include=["object", "datetime64"])` must be empty |
| 5 | Tensor shape consistency | `X_temporal.shape == (n_patients, n_timesteps, n_temporal_features)` i.e. `(205, 13, 8)` |
| 6 | No infinite values | `np.isinf(X_temporal).sum() + np.isinf(X_static).sum() == 0` |

**Phase A validation** (`_validate_cleaned_human_readable`):

| Check | What it verifies |
|-------|-----------------|
| Age not scaled | `age_mean > 1.0` (real years, not standardized) |
| Categorical labels preserved | `sex`, `outcome`, `diagnosis` have `dtype == object` |
| No one-hot columns | No columns matching `sex_Male`, `outcome_Cured` pattern |

---

### 7. Impossible Values Prevention

**Where:** Stage 3 (`clean_and_coerce_types`) + Stage 7 (post-MICE clipping)

| Issue | Fix | Location |
|-------|-----|----------|
| Future DOBs (2063 instead of 1963) | Subtract 100 years if DOB year > data_year | Stage 3 |
| Future clinical dates (2028 instead of 2018) | Subtract 10 years if year > data_year + 2 | Stage 3 |
| Negative weight/height from MICE | Clip to [10, 180] kg and [80, 220] cm | Stage 3 (pre-MICE) + Stage 7 (post-MICE) |
| Negative test results from MICE | Clip smear/xpert to [0, 1], missed doses to [0, ∞) | Stage 7 |
| Monthly doses > 31 | Clip to [0, 31] | Stage 3 + Stage 7 |
| Misplaced CXR findings in tuberculin_skin_test | Regex detect → NaN | Stage 3 |
| Near-empty columns mode-filled with single misleading value | >90% missing → `"Unknown"` instead of mode | Stage 7 |
| Temperature 300°C, BP impossible | Clinical range clipping: temp [30, 45], BP [40/20, 300/200] | Stage 3 |

---

## Two-Phase Architecture Summary

```
PHASE A (Cleaning)                          PHASE B (Model Preparation)
┌─────────────────────────┐                ┌──────────────────────────────┐
│ Stage 1: Schema         │                │ Stage 9:  One-hot encoding   │
│ Stage 2: Flag IDs       │                │ Stage 10: StandardScaler     │
│ Stage 3: Type coercion  │                │ Stage 11: IQR + Z-score cap  │
│ Stage 4: Harmonize cats │   deep copy    │ Stage 12: 3D tensor build    │
│ Stage 5: Flag redundant │ ────────────►  │ Stage 13: Validate + export  │
│ Stage 6: Wide → long    │                │                              │
│ Stage 7: MICE impute    │                │ Outputs:                     │
│ Stage 8: Export + valid  │                │   static_features.csv        │
│                         │                │   temporal_features.csv      │
│ Output:                 │                │   X_temporal.npy (205,13,8)  │
│   cleaned_human_        │                │   X_static.npy   (205,78)   │
│   readable.csv          │                │   X_combined_flat.npy        │
│   (real units, labels)  │                │   (scaled, encoded, capped)  │
└─────────────────────────┘                └──────────────────────────────┘
```

Phase B operates on a **deep copy** of Phase A data, ensuring the cleaned CSV is never retroactively modified by scaling or encoding.

---

## Verification Results

| # | Checklist Item | Status | Details |
|---|---------------|--------|---------|
| 1 | Remove identifiers | **PASS** | 4 ID columns retained in Phase A; 0 in Phase B |
| 2 | Handle missing values (diagnostic-first) | **PASS** | 0 missing across all outputs; 599/599 rows retained |
| 3 | Feature encoding | **PASS** | 38 readable object cols in Phase A; 0 in Phase B |
| 4 | Scaling/Normalization | **PASS** | Original units in Phase A; ~0 mean in Phase B |
| 5 | Outlier detection | **PASS** | 0 Inf, 0 NaN in model arrays |
| 6 | Schema validation | **PASS** | X_temporal (599,13,16), X_static (599,69), X_combined_flat (599,277) |

### Impossible Values Audit — All Clear

| Category | Result |
|----------|--------|
| Future dates | 0 found |
| Negative test results | 0 found |
| Monthly doses > 31 | 0 found |
| Weight outside [10, 180 kg] | 0 found |
| Height outside [80, 220 cm] | 0 found |
| Near-empty mode-fill errors | 0 found |
| Whitespace issues | 0 found |
| **Total missing values** | **0** |