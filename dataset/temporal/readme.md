# Preprocessing Pipeline — Technical Documentation

This document describes `preprocessingV2.py`, the canonical preprocessing script for the TB-DOTS CAR CDSS temporal dataset.

> **Scope:** Pure data cleaning only (Stages 1–8). Encoding, scaling, outlier capping, and tensor preparation are deliberately deferred to model-specific scripts to prevent data leakage across train/test splits.

---

## Pipeline Architecture

```
preprocessingV2.py  (Stages 1–8)
┌──────────────────────────────────────────────────────┐
│  Stage 1: Schema Validation & Column Standardization │
│  Stage 2: Identifier Removal (Patient Privacy)       │
│  Stage 3: Data Cleaning & Type Coercion              │
│  Stage 4: Categorical Harmonization                  │
│  Stage 5: Drop Uninformative Columns                 │
│  Stage 6: Wide → Long Temporal Structuring           │
│  Stage 7: Missing Data Handling (diagnostic-first)   │
│  Stage 7B: Post-imputation Clinical Clipping         │
│  Stage 8: Export & Validate                          │
│                                                      │
│  Output: cleaned_human_readable.csv (599 rows)       │
│          missing_data_report.json                    │
└──────────────────────────────────────────────────────┘
           ↓
  Model-specific scripts (after train/test split)
┌──────────────────────────────────────────────────────┐
│  - One-hot encoding                                  │
│  - StandardScaler (fit on train only)                │
│  - IQR + Z-score outlier capping                     │
│  - 3D tensor preparation for RNN/LSTM                │
└──────────────────────────────────────────────────────┘
```

The separation ensures that scalers and encoders are never fit on the full dataset before splitting — a common source of data leakage.

---

## Stage-by-Stage Reference

### Stage 1 — Schema Validation & Column Standardization

Loads `combined_complete_dataset.csv`, standardizes all column names to `snake_case`, and reports:
- Duplicate columns (auto-deduplicated, keeps first)
- Entirely null columns (flagged, not dropped)
- Data type distribution

### Stage 2 — Identifier Removal

Flags 4 patient/facility-identifying columns (`no`, `source_file`, `name_of_diagnosing_facility`, `name_of_treatment_unit`). They are **retained** in `cleaned_human_readable.csv` for traceability but must be excluded before any model training.

### Stage 3 — Data Cleaning & Type Coercion

Parses all columns to correct types with clinical validation:

| Issue | Fix |
|---|---|
| Future DOBs (e.g. 2063 instead of 1963) | Subtract 100 years if DOB year > data year |
| Future clinical dates | Subtract 10 years if year > data year + 2 |
| Weight/height strings with units | Regex parse → float |
| BP written as `120/80` strings | Split into `bp_systolic` / `bp_diastolic` |
| O2 sat with `%` suffix | Strip and parse |
| Temperature 300°C, BP impossible | Clip: temp [30, 45], BP [60/30, 220/140] |
| Smear/Xpert free-text results | Map to binary 0/1 |
| Monthly doses > 31 | Clip to [0, 31] |
| Misplaced CXR findings in `tuberculin_skin_test` | Regex detect → NaN |

### Stage 4 — Categorical Harmonization

Standardizes inconsistent categorical values to canonical labels:
- `nationality` — merges raw nationality strings with a `nationality_raw` column; normalises to Filipino / Foreign / Unknown
- `outcome` — maps variants to: Cured, Treatment Completed, Died, Treatment Failed, Lost to Follow-Up, Not Evaluated
- `sex` — maps M/F/Male/Female to Male/Female
- `regimen_type_at_start_of_treatment`, `diagnosis`, `civil_status`, comorbidities, prior TB history, DAT support — all standardised to consistent labels

### Stage 5 — Drop Uninformative Columns

Flags columns that are entirely null, near-entirely null, or redundant after harmonisation (e.g. `nationality_raw` after merging into `nationality`). Columns are flagged in the output rather than silently dropped to maintain auditability.

### Stage 6 — Wide → Long Temporal Structuring

Melts the wide-format dataset (one row per patient, columns like `m3_weight`, `m6_weight`) into long format:

```
df_static  — one row per patient  (baseline/demographic features)
df_temporal — one row per patient-month  (monthly observations M0–M12)
              columns: [patient_id, month, weight, height, pct_adherence,
                        monthly_doses_taken, monthly_missed_doses,
                        cumulative_doses_taken, smear_tb_lamp, xpert_mtb_rif]
```

### Stage 7 — Missing Data Handling (Diagnostic-First Pipeline)

> Full documentation: [`missing_data/README.md`](missing_data/README.md)

Replaces the original `impute_missing_mice()` with a modular pipeline that **diagnoses each column's missingness mechanism before deciding how to handle it**.

**Why this matters:** Applying the same imputation to every column regardless of mechanism is statistically incorrect. A missing M9 weight because a patient dropped out (MNAR) should be handled differently from a missing BP reading that is explainable by other observed variables (MAR).

**The four pathways:**

| Pathway | Condition | Action |
|---|---|---|
| **Alpha — Drop** | >90% missing (hard), or >50% + bottom importance quartile (soft) | Column removed |
| **Beta — Listwise** | MCAR proven + <15% missing + N > 500 after deletion | Rows removed, patient IDs synced |
| **Gamma — Indicator + Fill** | MNAR, all temporal features, or moderate missingness | Binary `is_missing_{col}` flag added; NaN filled with forward-fill (temporal) or median/mode (static) |
| **Delta — MICE** | MAR + informative feature | Stochastic MICE via `ExtraTreesRegressor` (missForest-style), FMI-scaled iterations |

> **Why temporal features are never Alpha-dropped:** Monthly observation columns have structurally high missingness because later months have fewer recorded values — patients hadn't reached those timepoints yet. The absence of a Month 9 weight is itself predictive (likely dropout). Gamma preserves this signal via the `is_missing_*` indicator.

**Results on `combined_complete_dataset.csv` (599 patients):**

| Pathway | Count | Columns |
|---|---|---|
| Alpha — dropped | 15 | Hard-dropped (>90%): near-empty fields, regimen columns, `respiratory_rate`, `temperature`. Soft-dropped (>50% + low importance): `blood_pressure` raw string (already split into `bp_systolic`/`bp_diastolic`) |
| Beta — listwise | 0 | N=599 too small; MCAR not confirmed |
| Gamma — indicator + fill | 25 | All 8 temporal features + 17 static. Includes `smear_microscopy` (83.8%), `height_cm` (82.1%), `outcome` (28.1%). Each gets a binary `is_missing_{col}` indicator |
| Delta — MICE | 4 | All MAR cardiovascular/respiratory vitals: `o2_sat` (79.8%), `heart_rate` (76.5%), `bp_systolic` (69.3%), `bp_diastolic` (69.3%) |

**`smear_microscopy` note:** Clinically important despite 83.8% missingness. Threshold raised to 0.90 so it routes to Gamma — the `is_missing_smear_microscopy` indicator captures the structured absence signal (patients not tested via smear often have a distinct diagnostic pathway).

**Key constraints:**
- Temporal features never Alpha-dropped (structural missingness)
- No `bfill()` on temporal data — forward-fill only (no future data leakage)
- `max_iter` scaled to FMI via Von Hippel's rule: `max(20, min(100, int(miss_rate × 100)))`
- Audit trail written to `output/missing_data_report.json`

**Result:** 0 missing values in both `df_static` and `df_temporal`.

### Stage 7B — Post-Imputation Clinical Clipping

Applied immediately after `handle_missing_data()` returns. Handled separately from the `missing_data/` package to keep clinical domain constraints out of the reusable imputation logic.

**Static bounds:**

| Column | Range |
|---|---|
| `weight_kg` | [10, 180] kg |
| `height_cm` | [80, 220] cm |
| `bp_systolic` | [60, 220] mmHg |
| `bp_diastolic` | [30, 140] mmHg |
| `heart_rate` | [20, 250] bpm |
| `respiratory_rate` | [4, 60] breaths/min |
| `temperature` | [30, 45] °C |
| `o2_sat` | [70, 100] % |
| `age` | [0, 110] years (re-rounded to int) |

**Temporal bounds:** weight [10,180], height [80,220], pct_adherence [0,100], monthly_doses_taken [0,31], monthly_missed_doses [0,∞), smear_tb_lamp [0,1], xpert_mtb_rif [0,1]

**Cumulative dose monotonicity:** `cumulative_doses_taken` is repaired via `cummax()` per patient — MICE imputes months independently and can introduce decreasing values (442 violations repaired on current dataset).

### Stage 8 — Export & Validate

Exports `cleaned_human_readable.csv` with all original clinical units and categorical labels (no encoding, no scaling). Validates:

| Check | What it verifies |
|---|---|
| Age not scaled | `age_mean > 1.0` (real years, not standardized) |
| Categorical labels preserved | `sex`, `outcome`, `diagnosis` have `dtype == object` |
| No one-hot columns | No columns matching `sex_Male`, `outcome_Cured` patterns |

---

## Outputs

| File | Description |
|---|---|
| `output/cleaned_human_readable.csv` | 599 rows × 269 columns — cleaned data in original units and labels, suitable for EDA and clinical review |
| `output/missing_data_report.json` | Audit trail of every missing data routing decision — mechanism, pathway, rationale per column |

---

## What This Pipeline Does NOT Do

Deliberately excluded from `preprocessingV2.py` to prevent data leakage:

| Task | Where it belongs |
|---|---|
| One-hot encoding | Model-specific script, after train/test split |
| StandardScaler / MinMaxScaler | Model-specific script, fit on train set only |
| Outlier capping (IQR / Z-score) | Model-specific script |
| 3D tensor preparation for RNN/LSTM | Model-specific script |

Fitting a scaler on the full dataset before splitting leaks test-set distribution into training — a subtle but common bug. The cleaned CSV is the hand-off point.

---

## Running the Pipeline

```bash
python dataset/temporal/preprocessingV2.py
```

From the project root. Reads `dataset/temporal/combined_complete_dataset.csv`, writes to `dataset/temporal/output/`.

---

## Verification Checklist

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Identifiers removed from model data | **PASS** | 4 ID cols in CSV only; excluded in model scripts |
| 2 | Zero NaN in cleaned outputs | **PASS** | Static: 0 / Temporal: 0 (from missing_data_report.json) |
| 3 | No scaled/encoded values in CSV | **PASS** | `age_mean > 1.0`; object dtypes preserved |
| 4 | No impossible clinical values | **PASS** | All columns within bounds post-clipping |
| 5 | Temporal leakage prevented | **PASS** | Forward-fill only; no bfill in temporal imputation |
| 6 | 599 rows retained | **PASS** | Zero rows deleted (Beta pathway not triggered) |
| 7 | `smear_microscopy` preserved | **PASS** | GAMMA_INDICATOR at 83.8% missing |
| 8 | Audit trail generated | **PASS** | `output/missing_data_report.json` written every run |
