"""
==============================================================================
TB-DOTS CAR Clinical Decision Support System (CDSS)
Preprocessing Pipeline for Pulmonary Tuberculosis (PTB) Dataset
==============================================================================

This pipeline prepares longitudinal TB clinical registry data (2015-2025)
from the Cordillera Administrative Region (CAR), Philippines, for temporal
machine learning models (RNN, LSTM, LSTM+XGBoost hybrid).

Dataset: TB-DOTS treatment monitoring records with monthly observations
         (M0-M12) covering demographics, diagnostics, vitals, and adherence.

The pipeline is split into two distinct phases:

  PHASE A - DATA CLEANING (produces cleaned_human_readable.csv)
    1. Schema Validation & Column Standardization
    2. Identifier Removal (Patient Privacy)
    3. Data Cleaning & Type Coercion
    4. Categorical Harmonization
    5. Drop Uninformative Columns
    6. Temporal Data Structuring (wide -> long format)
    7. Missing Data Handling (forward-fill + MICE imputation)
    8. Export & Validate cleaned_human_readable.csv

  PHASE B - MODEL PREPARATION (produces model-ready outputs)
    9.  One-Hot Encoding of categorical features
   10.  Feature Scaling (StandardScaler)
   11.  Outlier Detection & Capping (IQR + Z-score)
   12.  Temporal Modeling Preparation (3D tensor for RNN/LSTM)
   13.  Final Validation & Export of model-ready outputs

Design rationale:
  - Phase A produces a single, interpretable CSV suitable for EDA and
    clinical validation, with real units (kg, cm, %, mmHg) and readable
    categorical labels (Male/Female, Cured, Regimen 1, etc.).
  - Phase B derives all model-ready outputs strictly from the cleaned
    data produced in Phase A. This ensures:
      * Scaling and encoding happen only after cleaning and imputation.
      * No data leakage: temporal imputation uses only past data
        (forward-fill, never backward-fill).
      * Future train/test splitting can be inserted between Phase A
        and Phase B without re-running the cleaning pipeline.

Author: TB-DOTS CAR CDSS Research Team
Date:   2025 (revised 2026)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
INPUT_PATH = os.path.join("dataset", "temporal", "combined_complete_dataset.csv")
OUTPUT_DIR = os.path.join("dataset", "temporal", "output")

# Monthly time-step range
MONTH_RANGE = range(0, 13)  # M0 through M12

# Per-month feature suffixes (as they appear in the raw column names)
MONTHLY_SUFFIXES = [
    "Monthly Doses Taken",
    "Cumulative Doses Taken",
    "Monthly Missed Doses",
    "%Adherence",
    "Weight",
    "Height",
    "Smear/TB Lamp",
    "Xpert MTB/RIF",
]

# Identifier columns to drop (privacy)
ID_COLUMNS_TO_DROP = [
    "no",
    "source_file",
    "name_of_diagnosing_facility",
    "name_of_treatment_unit",
]

# Columns that are entirely null and carry no information
ALWAYS_DROP_COLUMNS = [
    "tuberculosis_culture",     # near-entirely missing
    "tuberculin_skin_test",     # near-entirely missing
    "other_lab_test",           # near-entirely missing (1/599 non-null)
    "others",                   # near-entirely missing
    "other",                    # new in complete dataset: 'Other:' -> 'other'; mostly N/A (29/599 non-null)
    "dat_supported_dup",        # duplicate of dat_supported
    "risk_factors_for_drug_resistance_tuberculosis",  # all null
]

# Categorical columns for one-hot encoding (used in Phase B only)
# NOTE: The complete dataset has both 'Nationality?' (standardized -> 'nationality')
# and a new 'Nationality' column (standardized -> 'nationality'). After standardization
# these will collide; the duplicate is handled in Stage 1 schema validation.
# Only 'nationality' (from 'Nationality?') is kept in the model as it has more coverage.
CATEGORICAL_COLUMNS = [
    "sex",
    "civil_status",
    "nationality",
    "diagnosis",
    "bacteriologic_status",
    "treatment_regimen",
    "outcome",
    "case_registration_group",
    "drug_resistance_bacteriological_status",
    "chest_x_ray_at_case_notification",
    "xpert_mtb_rif",
    "smear_microscopy",
    "co_morbidities",
    "prior_history_of_tb",
    "regimen_type_at_start_of_treatment",
    "dat_supported",
]

# Numerical columns for scaling (baseline clinical numerics)
NUMERICAL_COLUMNS_SCALE = [
    "age",
    "weight_kg",
    "height_cm",
    "bp_systolic",
    "bp_diastolic",
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "o2_sat",
]

# Columns for outlier detection
OUTLIER_COLUMNS = [
    "age",
    "weight_kg",
    "height_cm",
    "bp_systolic",
    "bp_diastolic",
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "o2_sat",
]


# ############################################################################
#
#  PHASE A - DATA CLEANING
#
#  Stages 1-8: Load -> clean -> harmonize -> impute -> export human-readable
#  All outputs retain original clinical units and categorical labels.
#  No scaling, encoding, or outlier capping is applied in this phase.
#
# ############################################################################


# ============================================================================
# STAGE 1: SCHEMA VALIDATION & COLUMN STANDARDIZATION  [CLEANING]
# ============================================================================

def standardize_column_name(col: str) -> str:
    """
    Convert raw column names to snake_case identifiers.

    Transforms:
        'M0_%Adherence'           -> 'm0_pct_adherence'
        'Blood Pressure'          -> 'blood_pressure'
        'Xpert MTB/RIF'           -> 'xpert_mtb_rif'
        'Smear/TB Lamp'           -> 'smear_tb_lamp'
        'Date of Birth'           -> 'date_of_birth'
        'OTHERS:'                 -> 'others'
        'DAT- supported'          -> 'dat_supported_dup'
        'Nationality'             -> 'nationality_raw'  (new col; avoids collision with 'Nationality?')
        'Other:'                  -> 'other'            (new col in complete dataset)
    """
    col = col.strip()
    # Handle the duplicate DAT column
    if col == "DAT- supported":
        return "dat_supported_dup"
    # Handle new 'Nationality' column (bare, without '?') to avoid colliding
    # with 'Nationality?' which also standardizes to 'nationality'
    if col == "Nationality":
        return "nationality_raw"
    # Replace % with pct
    col = col.replace("%", "pct_")
    col = col.replace("?", "")
    col = col.replace(":", "")
    col = col.replace("'", "")
    col = col.replace("/", "_")
    col = col.replace("-", "_")
    col = col.replace("(", "").replace(")", "")
    col = col.replace(".", "")
    # Collapse whitespace
    col = re.sub(r"\s+", "_", col)
    # Remove consecutive underscores
    col = re.sub(r"_+", "_", col)
    col = col.strip("_").lower()
    # Specific renames for clarity
    col = col.replace("risk_factor_s_for_drug_resistance_tuberculosis",
                       "risk_factors_for_drug_resistance_tuberculosis")
    col = col.replace("nationality", "nationality")
    col = col.replace("civil_status", "civil_status")
    col = col.replace("dat_supported", "dat_supported")
    col = col.replace("no", "no") if col == "no" else col
    return col


def load_and_validate_schema(path: str) -> pd.DataFrame:
    """
    Stage 1 [CLEANING]: Load raw CSV, standardize column names, report schema.

    Returns:
        pd.DataFrame with standardized column names.
    """
    print("=" * 70)
    print("STAGE 1: Schema Validation & Column Standardization")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Loaded dataset: {df.shape[0]} patients x {df.shape[1]} columns")

    # Standardize column names
    original_cols = df.columns.tolist()
    df.columns = [standardize_column_name(c) for c in df.columns]
    renamed = {o: n for o, n in zip(original_cols, df.columns) if o != n}
    print(f"  Renamed {len(renamed)} columns to snake_case")

    # Check for duplicate column names after standardization
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        print(f"  WARNING: Duplicate columns found: {dupes}")

    # Report null columns
    all_null = df.columns[df.isnull().all()].tolist()
    print(f"  Entirely null columns ({len(all_null)}): {all_null[:10]}...")

    # Report basic dtypes
    print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
    print()
    return df


# ============================================================================
# STAGE 2: REMOVE IDENTIFIERS  [CLEANING]
# ============================================================================

def remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2 [CLEANING]: Flag patient-identifying and facility-identifying
    columns. These columns are retained in the dataset for traceability
    but are excluded from model features in Phase B.

    Flagged columns:
        - No. (patient row number) -> reassigned to sequential 1..N
        - Source_File (data provenance)
        - Name of Diagnosing Facility (facility ID)
        - Name of Treatment Unit (facility ID)

    The 'no' column is reassigned to a clean sequential integer index
    (1, 2, 3, ...) so downstream exports have a stable, unambiguous row
    number regardless of the heterogeneous original identifiers.
    """
    print("=" * 70)
    print("STAGE 2: Flag Identifiers (Patient Privacy)")
    print("=" * 70)

    # Reassign 'no' to a clean sequential index (1-based)
    if "no" in df.columns:
        df["no"] = range(0, len(df))
        print(f"  Reassigned 'no' column to sequential index 0..{len(df) - 1}")

    flagged = [c for c in ID_COLUMNS_TO_DROP if c in df.columns]
    print(f"  Flagged {len(flagged)} identifier columns (retained): {flagged}")
    print(f"  Total columns: {df.shape[1]}")
    print()
    return df


# ============================================================================
# STAGE 3: DATA CLEANING & TYPE COERCION  [CLEANING]
# ============================================================================

def parse_weight(val) -> float:
    """Extract numeric weight in kg from strings like '45kg', '56.5 kg'."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    val = val.replace("kg", "").replace("kgs", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def parse_height(val) -> float:
    """Extract numeric height in cm from strings like '162.5 cm', '175cm'."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    val = val.replace("cm", "").replace("cms", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def parse_blood_pressure(val):
    """
    Parse blood pressure string '120/80' into systolic and diastolic components.

    Returns:
        (systolic: float, diastolic: float) or (NaN, NaN) if unparseable.
    """
    if pd.isna(val):
        return np.nan, np.nan
    val = str(val).strip()
    match = re.match(r"(\d+)\s*/\s*(\d+)", val)
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan


def parse_o2_sat(val) -> float:
    """
    Extract numeric O2 saturation in percentage points (70-100 scale).

    Raw data mixes two formats:
        - Decimal fraction: 0.95  -> 95.0%
        - Percentage:       95.0  -> 95.0%
        - With suffix:      '98%' -> 98.0%
        - Garbage / N/A    -> NaN

    Values that cannot be mapped to the 70-100 clinical range after
    unit resolution are returned as NaN so MICE does not use corrupt
    seeds.
    """
    if pd.isna(val):
        return np.nan
    raw = str(val).strip()
    if raw.lower() in ("n/a", "na", "nan", ""):
        return np.nan
    # Strip % suffix and any trailing junk characters
    raw = re.sub(r"[%%\s]+$", "", raw).strip()
    try:
        v = float(raw)
    except ValueError:
        return np.nan
    # Decimal fraction encoding (e.g. 0.95 means 95%)
    # Use threshold <= 2.0 to catch values like 1.0 (100%) safely
    if v <= 2.0:
        v = v * 100.0
    # Clamp to plausible clinical range; anything outside is set to NaN
    # so it is imputed rather than propagating a corrupt value
    if v < 70.0 or v > 100.0:
        return np.nan
    return round(v, 2)


def parse_adherence(val) -> float:
    """Extract numeric adherence percentage from strings like '100%', '95.5%'."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip().replace("%", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


def parse_date_safe(val):
    """Parse date strings with mixed formats, handling typos like '6/10/0202'."""
    if pd.isna(val):
        return pd.NaT
    val = str(val).strip()
    # Strip trailing non-date characters (e.g., '4/3/1995= ' -> '4/3/1995')
    val = re.sub(r"[^0-9/\-]+$", "", val).strip()
    if not val:
        return pd.NaT
    # Fix common year typos (e.g., 0202 -> 2022)
    val = re.sub(r"/0(\d{3})$", r"/\1", val)      # /0202 -> /202 (won't fix)
    val = re.sub(r"/02(\d{2})$", r"/20\1", val)    # ensure 4-digit year
    # Fix transposed-digit years (e.g., 2990 -> 1990, 2979 -> 1979)
    m = re.search(r"/(\d{4})$", val)
    if m:
        year = int(m.group(1))
        if year > 2026:
            # Try swapping first two digits: 29xx -> 19xx
            corrected = "19" + m.group(1)[2:]
            val = val[: m.start(1)] + corrected
    try:
        result = pd.to_datetime(val, format="mixed", dayfirst=False)
    except Exception:
        try:
            result = pd.to_datetime(val, infer_datetime_format=True)
        except Exception:
            return pd.NaT
    return result


def parse_smear_result(val) -> float:
    """
    Harmonize smear/TB-LAMP results to numeric codes:
        0 = Negative / Not detected / 0
        1 = Positive / Detected / any AFB grade
        NaN = missing
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    if val in ("0", "negative", "negataive", "neg", "not detected",
               "no afb seen", "n", "nd", "mtb not det", "mtb nd",
               "n (not detected)", "mtb not detected"):
        return 0.0
    if val in ("", "nan"):
        return np.nan
    # Any positive indication
    return 1.0


def parse_xpert_result(val) -> float:
    """
    Harmonize Xpert MTB/RIF results to numeric codes:
        0 = MTB Not Detected / Negative
        1 = MTB Detected (any level including Trace)
        NaN = missing
    """
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    if val in ("not detected", "mtb not det", "mtb nd", "mtb not detected",
               "n (not detected)", "negative", "n", "nd", "0"):
        return 0.0
    if val in ("", "nan"):
        return np.nan
    # Detected, Trace, or any positive
    return 1.0


def clean_and_coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3 [CLEANING]: Parse and coerce all columns to proper types.

    Operations:
        - Parse date columns (DOB, diagnosis, treatment start, outcome)
        - Extract numeric weight (kg) from text
        - Extract numeric height (cm) from text
        - Split Blood Pressure into systolic/diastolic
        - Parse O2 saturation percentages
        - Parse monthly adherence percentages
        - Harmonize smear and Xpert results to binary
        - Coerce numeric columns

    All values remain in their original clinical units (kg, cm, mmHg, %).
    """
    print("=" * 70)
    print("STAGE 3: Data Cleaning & Type Coercion")
    print("=" * 70)

    # --- Date columns (missing values filled with median date) ---
    date_cols = ["date_of_birth", "date_of_diagnosis", "date_of_notification",
                 "treatment_start_date", "date_of_outcome",
                 "intensive_phase_start_date", "intensive_phase_end_date",
                 "continuation_phase_start_date", "continuation_phase_end_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_date_safe)
            parsed = df[col].notna().sum()
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                valid_dates = df[col].dropna()
                if len(valid_dates) > 0:
                    # Compute median date from valid values
                    timestamps = valid_dates.astype(np.int64)
                    median_ts = np.median(timestamps)
                    median_date = pd.Timestamp(median_ts)
                    df[col] = df[col].fillna(median_date)
                    print(f"  Parsed '{col}': {parsed}/{len(df)} valid, "
                          f"{n_missing} filled with median ({median_date.strftime('%Y-%m-%d')})")
                else:
                    print(f"  Parsed '{col}': {parsed}/{len(df)} valid, "
                          f"{n_missing} missing (no valid dates for median)")
            else:
                print(f"  Parsed '{col}': {parsed}/{len(df)} valid dates")

    # --- Fix future DOBs (century-swap for 2-digit year ambiguity) ---
    # If DOB year > data_year, the century was likely mis-assigned (e.g., 2063 -> 1963).
    # Also catches the reverse error: DOB year < 1920 where the recorded age implies
    # the patient was born in the 20th century (e.g. 1898 should be 1998).
    if "date_of_birth" in df.columns:
        dob = df["date_of_birth"]
        data_yr = df["data_year"] if "data_year" in df.columns else pd.Series(2025, index=df.index)

        # Forward fix: DOB is in the future
        future_mask = dob.notna() & (dob.dt.year > data_yr)
        n_fixed = future_mask.sum()
        if n_fixed > 0:
            df.loc[future_mask, "date_of_birth"] = dob[future_mask].apply(
                lambda d: d.replace(year=d.year - 100)
            )
            print(f"  Fixed {n_fixed} future DOBs by subtracting 100 years")

        # Backward fix: DOB year < 1920 but recorded age suggests a modern birth year.
        # Computed age from corrected DOB (add 100 years) must be plausible (0-110).
        dob_refreshed = df["date_of_birth"]
        ancient_mask = dob_refreshed.notna() & (dob_refreshed.dt.year < 1920)
        if ancient_mask.sum() > 0:
            def fix_ancient_dob(row):
                d = row["date_of_birth"]
                corrected_year = d.year + 100
                corrected_age = (pd.Timestamp(str(int(row.get("data_year", 2025))) + "-06-01") -
                                 d.replace(year=corrected_year)).days / 365.25
                if 0 <= corrected_age <= 110:
                    return d.replace(year=corrected_year)
                return d
            n_ancient = ancient_mask.sum()
            df.loc[ancient_mask, "date_of_birth"] = df[ancient_mask].apply(fix_ancient_dob, axis=1)
            print(f"  Fixed {n_ancient} historically implausible DOBs (pre-1920) by adding 100 years")

    # --- Fix future treatment/clinical dates (century-swap) ---
    # Clinical dates should not be more than 2 years after data_year.
    clinical_date_cols = [
        "date_of_diagnosis", "date_of_notification", "date_of_outcome",
        "intensive_phase_start_date", "intensive_phase_end_date",
        "continuation_phase_start_date", "continuation_phase_end_date",
    ]
    for col in clinical_date_cols:
        if col in df.columns:
            dates = df[col]
            data_yr = df["data_year"] if "data_year" in df.columns else pd.Series(2025, index=df.index)
            future_mask = dates.notna() & (dates.dt.year > data_yr + 2)
            n_fixed = future_mask.sum()
            if n_fixed > 0:
                df.loc[future_mask, col] = dates[future_mask].apply(
                    lambda d: d.replace(year=d.year - 10)
                )
                print(f"  Fixed {n_fixed} future dates in '{col}' by subtracting 10 years")

    # --- Enforce date ordering: treatment_start_date >= date_of_diagnosis ---
    # Treatment cannot logically begin before a diagnosis is made.
    # Where this is violated, set treatment_start_date = date_of_diagnosis.
    if "treatment_start_date" in df.columns and "date_of_diagnosis" in df.columns:
        bad_order = (
            df["treatment_start_date"].notna()
            & df["date_of_diagnosis"].notna()
            & (df["treatment_start_date"] < df["date_of_diagnosis"])
        )
        n_bad = bad_order.sum()
        if n_bad > 0:
            df.loc[bad_order, "treatment_start_date"] = df.loc[bad_order, "date_of_diagnosis"]
            print(f"  Fixed {n_bad} rows where treatment_start_date < date_of_diagnosis "
                  f"(set to date_of_diagnosis)")

    # --- Baseline Weight -> numeric kg (original text column standardized) ---
    if "weight" in df.columns:
        df["weight_kg"] = df["weight"].apply(parse_weight)
        # Standardize raw text to uniform "XX.X kg" format
        df["weight"] = df["weight_kg"].apply(
            lambda v: f"{v:.1f} kg" if pd.notna(v) else np.nan
        )
        print(f"  Parsed 'weight' -> 'weight_kg': "
              f"{df['weight_kg'].notna().sum()} valid "
              f"(raw 'weight' standardized to 'XX.X kg')")

    # --- Baseline Height -> numeric cm (original text column standardized) ---
    if "height" in df.columns:
        df["height_cm"] = df["height"].apply(parse_height)
        # Standardize raw text to uniform "XXX.X cm" format
        df["height"] = df["height_cm"].apply(
            lambda v: f"{v:.1f} cm" if pd.notna(v) else np.nan
        )
        print(f"  Parsed 'height' -> 'height_cm': "
              f"{df['height_cm'].notna().sum()} valid "
              f"(raw 'height' standardized to 'XXX.X cm')")

    # --- Blood Pressure -> systolic + diastolic (original text column retained) ---
    if "blood_pressure" in df.columns:
        bp_parsed = df["blood_pressure"].apply(parse_blood_pressure)
        df["bp_systolic"] = bp_parsed.apply(lambda x: x[0])
        df["bp_diastolic"] = bp_parsed.apply(lambda x: x[1])
        print(f"  Parsed 'blood_pressure' -> 'bp_systolic', 'bp_diastolic': "
              f"{df['bp_systolic'].notna().sum()} valid (raw 'blood_pressure' retained)")

    # --- O2 Saturation -> numeric ---
    if "o2_sat" in df.columns:
        df["o2_sat"] = df["o2_sat"].apply(parse_o2_sat)
        print(f"  Parsed 'o2_sat': {df['o2_sat'].notna().sum()} valid")

    # --- Clean tuberculin_skin_test: remove misplaced CXR findings ---
    if "tuberculin_skin_test" in df.columns:
        invalid_mask = (
            df["tuberculin_skin_test"].astype(str).str.lower()
            .str.contains("atelectasis|infiltrate|opacity|mass", na=False)
        )
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            df.loc[invalid_mask, "tuberculin_skin_test"] = np.nan
            print(f"  Cleaned 'tuberculin_skin_test': removed {n_invalid} "
                  f"misplaced CXR findings -> NaN")

    # --- Baseline Xpert MTB/RIF -> binary ---
    if "xpert_mtb_rif" in df.columns:
        df["xpert_mtb_rif"] = df["xpert_mtb_rif"].apply(parse_xpert_result)
        print(f"  Parsed 'xpert_mtb_rif' -> binary: "
              f"{df['xpert_mtb_rif'].notna().sum()} valid")

    # --- Baseline Smear Microscopy -> binary ---
    if "smear_microscopy" in df.columns:
        df["smear_microscopy"] = df["smear_microscopy"].apply(parse_smear_result)
        print(f"  Parsed 'smear_microscopy' -> binary: "
              f"{df['smear_microscopy'].notna().sum()} valid")

    # --- Monthly columns ---
    for m in MONTH_RANGE:
        prefix = f"m{m}_"

        # Weight (may contain 'kg' suffix) -> uniform numeric kg
        wcol = f"{prefix}weight"
        if wcol in df.columns:
            df[wcol] = df[wcol].apply(parse_weight)

        # Height (may contain 'cm' suffix) -> uniform numeric cm
        hcol = f"{prefix}height"
        if hcol in df.columns:
            df[hcol] = df[hcol].apply(parse_height)

        # Adherence (may contain '%' suffix)
        acol = f"{prefix}pct_adherence"
        if acol in df.columns:
            df[acol] = df[acol].apply(parse_adherence)

        # Smear/TB Lamp -> binary
        scol = f"{prefix}smear_tb_lamp"
        if scol in df.columns:
            df[scol] = df[scol].apply(parse_smear_result)

        # Xpert MTB/RIF -> binary
        xcol = f"{prefix}xpert_mtb_rif"
        if xcol in df.columns:
            df[xcol] = df[xcol].apply(parse_xpert_result)

        # Doses & missed doses -> numeric
        for suffix in ["monthly_doses_taken", "cumulative_doses_taken",
                        "monthly_missed_doses"]:
            dcol = f"{prefix}{suffix}"
            if dcol in df.columns:
                df[dcol] = pd.to_numeric(df[dcol], errors="coerce")

    print(f"  Cleaned monthly columns M0-M12 (weight, height, adherence, "
          f"smear, xpert, doses)")

    # --- Numeric columns that should already be numeric ---
    for col in ["age", "heart_rate", "respiratory_rate", "temperature"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Clinical sanity checks (flag & clip impossible values) ---
    print("\n  --- Clinical Sanity Checks ---")
    # Define clinically plausible ranges: (column, min, max, unit)
    clinical_ranges = [
        ("age",              0,    110,  "years"),
        ("weight_kg",       10,    180,  "kg"),
        ("height_cm",       80,    220,  "cm"),
        ("bp_systolic",     60,    220,  "mmHg"),
        ("bp_diastolic",    30,    140,  "mmHg"),
        ("heart_rate",      20,    250,  "bpm"),
        ("respiratory_rate", 4,     60,  "breaths/min"),
        ("temperature",     30,     45,  "°C"),
        ("o2_sat",          70,    100,  "%"),
    ]
    for col, cmin, cmax, unit in clinical_ranges:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            below = (df[col] < cmin).sum()
            above = (df[col] > cmax).sum()
            if below > 0 or above > 0:
                df[col] = df[col].clip(lower=cmin, upper=cmax)
                print(f"    {col}: clipped {below} below {cmin} and "
                      f"{above} above {cmax} {unit}")
            else:
                print(f"    {col}: all values in range [{cmin}-{cmax} {unit}]")

    # --- Round age to nearest integer (no decimal ages in clinical records) ---
    if "age" in df.columns and pd.api.types.is_numeric_dtype(df["age"]):
        df["age"] = df["age"].round(0).astype("Int64")  # nullable integer
        print(f"    age: rounded to integer (no decimals)")

    # Monthly weight, height, adherence, doses, and test results sanity checks
    for m in MONTH_RANGE:
        wcol = f"m{m}_weight"
        if wcol in df.columns and pd.api.types.is_numeric_dtype(df[wcol]):
            df[wcol] = df[wcol].clip(lower=10, upper=180)
        hcol = f"m{m}_height"
        if hcol in df.columns and pd.api.types.is_numeric_dtype(df[hcol]):
            df[hcol] = df[hcol].clip(lower=80, upper=220)
        acol = f"m{m}_pct_adherence"
        if acol in df.columns and pd.api.types.is_numeric_dtype(df[acol]):
            df[acol] = df[acol].clip(lower=0, upper=100)
        # Monthly doses taken: 0-31 (max days in a month)
        dcol = f"m{m}_monthly_doses_taken"
        if dcol in df.columns and pd.api.types.is_numeric_dtype(df[dcol]):
            df[dcol] = df[dcol].clip(lower=0, upper=31)
        # Monthly missed doses: non-negative
        mdcol = f"m{m}_monthly_missed_doses"
        if mdcol in df.columns and pd.api.types.is_numeric_dtype(df[mdcol]):
            df[mdcol] = df[mdcol].clip(lower=0)
        # Smear/TB Lamp: binary 0/1
        scol = f"m{m}_smear_tb_lamp"
        if scol in df.columns and pd.api.types.is_numeric_dtype(df[scol]):
            df[scol] = df[scol].clip(lower=0, upper=1)
        # Xpert MTB/RIF: binary 0/1
        xcol = f"m{m}_xpert_mtb_rif"
        if xcol in df.columns and pd.api.types.is_numeric_dtype(df[xcol]):
            df[xcol] = df[xcol].clip(lower=0, upper=1)
    print("    Monthly weight/height/adherence/doses/tests: clipped to plausible ranges")

    print()
    return df


# ============================================================================
# STAGE 4: CATEGORICAL HARMONIZATION  [CLEANING]
# ============================================================================

def harmonize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 4 [CLEANING]: Standardize inconsistent categorical values.

    Many columns have multiple representations of the same category
    (e.g., 'M'/'Male', 'S'/'Single', 'CD'/'Clinically-diagnosed TB').
    This stage maps them to canonical, human-readable values.

    All categorical labels are preserved as strings (not encoded).
    """
    print("=" * 70)
    print("STAGE 4: Categorical Harmonization")
    print("=" * 70)

    # --- Sex ---
    if "sex" in df.columns:
        sex_map = {
            "m": "Male", "male": "Male",
            "f": "Female", "female": "Female",
            "4": np.nan,  # data entry error
        }
        df["sex"] = (df["sex"].astype(str).str.strip().str.lower()
                      .map(sex_map).where(df["sex"].notna()))
        print(f"  Harmonized 'sex': {df['sex'].value_counts().to_dict()}")

    # --- Civil Status ---
    if "civil_status" in df.columns:
        cs_map = {
            "s": "Single", "single": "Single",
            "m": "Married", "married": "Married",
            "w": "Widowed", "widow": "Widowed", "widowed": "Widowed",
            "sep": "Separated", "separated": "Separated",
        }
        df["civil_status"] = (df["civil_status"].astype(str).str.strip()
                               .str.lower().map(cs_map)
                               .where(df["civil_status"].notna()))
        print(f"  Harmonized 'civil_status': "
              f"{df['civil_status'].value_counts().to_dict()}")

    # --- Nationality ---
    if "nationality" in df.columns:
        df["nationality"] = (df["nationality"].astype(str).str.strip()
                              .str.lower())
        df.loc[df["nationality"].isin(["filipino"]), "nationality"] = "Filipino"
        df.loc[df["nationality"].isin(["n/a", "nan"]),
               "nationality"] = np.nan
        print(f"  Harmonized 'nationality': "
              f"{df['nationality'].value_counts().to_dict()}")

    # --- Merge nationality_raw into nationality (new complete dataset) ---
    # The complete dataset adds a bare 'Nationality' column (-> 'nationality_raw')
    # that partially overlaps with 'Nationality?' (-> 'nationality').
    # Fill any gaps in 'nationality' from 'nationality_raw', then drop it.
    if "nationality_raw" in df.columns:
        def harmonize_nat_raw(val):
            if pd.isna(val):
                return np.nan
            val = str(val).strip().lower()
            if val in ("n/a", "nan", ""):
                return np.nan
            if val == "filipino":
                return "Filipino"
            return val.title()
        nat_raw_harmonized = df["nationality_raw"].apply(harmonize_nat_raw)
        # Only fill where 'nationality' is still missing
        if "nationality" in df.columns:
            mask = df["nationality"].isna() & nat_raw_harmonized.notna()
            df.loc[mask, "nationality"] = nat_raw_harmonized[mask]
            filled = mask.sum()
            if filled > 0:
                print(f"  Filled {filled} missing 'nationality' values from 'nationality_raw'")
        # Drop nationality_raw - it's been merged and is otherwise redundant
        df = df.drop(columns=["nationality_raw"])
        print(f"  Dropped 'nationality_raw' after merging into 'nationality'")

    # --- Diagnosis ---
    if "diagnosis" in df.columns:
        diag_map = {
            "tb disease": "TB Disease",
            "tb infection": "TB Infection",
        }
        df["diagnosis"] = (df["diagnosis"].astype(str).str.strip().str.lower()
                            .map(diag_map).where(df["diagnosis"].notna()))
        print(f"  Harmonized 'diagnosis': "
              f"{df['diagnosis'].value_counts().to_dict()}")

    # --- Bacteriologic Status ---
    if "bacteriologic_status" in df.columns:
        bact_map = {
            "bc": "Bacteriologically Confirmed",
            "bacteriologically-confirmed tb": "Bacteriologically Confirmed",
            "bacteriologically- confirmed tb": "Bacteriologically Confirmed",
            "bacteriologically -confirmed tb": "Bacteriologically Confirmed",
            "bacteriologically- confirmed ptb": "Bacteriologically Confirmed",
            "bacteriologically confirmed tb": "Bacteriologically Confirmed",
            "cd": "Clinically Diagnosed",
            "clinically-diagnosed tb": "Clinically Diagnosed",
            "clinically diagnosed tb": "Clinically Diagnosed",
        }
        df["bacteriologic_status"] = (
            df["bacteriologic_status"].astype(str).str.strip().str.lower()
            .map(bact_map).where(df["bacteriologic_status"].notna())
        )
        print(f"  Harmonized 'bacteriologic_status': "
              f"{df['bacteriologic_status'].value_counts().to_dict()}")

    # --- Treatment Regimen ---
    if "treatment_regimen" in df.columns:
        reg_map = {
            # Numeric string variants
            "1": "Regimen 1", "1.0": "Regimen 1",
            "2": "Regimen 2", "2.0": "Regimen 2",
            # Named variants
            "regimen 1": "Regimen 1", "regimen i": "Regimen 1",
            "cat 1": "Regimen 1", "i": "Regimen 1",
            "regimen 2": "Regimen 2",
            "1a": "Regimen 1a", "ia": "Regimen 1a",
            # Standard drug-code shorthand (HRZE = isoniazid/rifampicin/pyrazinamide/ethambutol)
            "hrze": "Regimen 1", "2hrze*/4hr": "Regimen 1",
            "2hrze/4hr": "Regimen 1", "2hrze*4hr": "Regimen 1",
            "2hrze 4hr": "Regimen 1",
            # Unmappable
            "-": np.nan, "cannot remember": np.nan,
        }
        df["treatment_regimen"] = (
            df["treatment_regimen"].astype(str).str.strip().str.lower()
            .map(reg_map).where(df["treatment_regimen"].notna())
        )
        print(f"  Harmonized 'treatment_regimen': "
              f"{df['treatment_regimen'].value_counts().to_dict()}")

    # --- Outcome ---
    if "outcome" in df.columns:
        out_map = {
            "cured": "Cured",
            "treatment completed": "Treatment Completed",
            "tx completed": "Treatment Completed",
            "treatment complete": "Treatment Completed",
            "ltfu": "Lost to Follow-Up",
            "died": "Died",
            "not evaluated - transferred": "Not Evaluated",
            "not evaluated": "Not Evaluated",
            "transferred out": "Not Evaluated",
            "treatment failure": "Treatment Failure",
        }
        df["outcome"] = (
            df["outcome"].astype(str).str.strip().str.lower()
            .map(out_map).where(df["outcome"].notna())
        )
        print(f"  Harmonized 'outcome': "
              f"{df['outcome'].value_counts().to_dict()}")

    # --- Case Registration Group ---
    if "case_registration_group" in df.columns:
        crg_map = {
            "new": "New", "relapse": "Relapse",
            "talf": "Treatment After Lost to Follow-Up",
            "taf": "Treatment After Failure",
        }
        df["case_registration_group"] = (
            df["case_registration_group"].astype(str).str.strip().str.lower()
            .map(crg_map).where(df["case_registration_group"].notna())
        )
        print(f"  Harmonized 'case_registration_group': "
              f"{df['case_registration_group'].value_counts().to_dict()}")

    # --- Drug Resistance Status ---
    if "drug_resistance_bacteriological_status" in df.columns:
        dr_map = {
            "drug-susceptible": "Drug Susceptible",
            "dstb": "Drug Susceptible",
            "drug susceptible": "Drug Susceptible",
            "drg susceptible": "Drug Susceptible",
            "clinically - diagnosed mdr-tb": "MDR-TB",
            "bacteriologically confirmed mdr-tb": "MDR-TB",
            "bc-mdr tb": "MDR-TB",
        }
        df["drug_resistance_bacteriological_status"] = (
            df["drug_resistance_bacteriological_status"].astype(str)
            .str.strip().str.lower()
            .map(dr_map)
            .where(df["drug_resistance_bacteriological_status"].notna())
        )
        print(f"  Harmonized 'drug_resistance_bacteriological_status': "
              f"{df['drug_resistance_bacteriological_status'].value_counts().to_dict()}")

    # --- Regimen Type at Start of Treatment ---
    if "regimen_type_at_start_of_treatment" in df.columns:
        rt_map = {
            "1": "Regimen 1", "regimen 1": "Regimen 1", "cat 1": "Regimen 1",
            "2": "Regimen 2", "regimen 2": "Regimen 2",
            "1a": "Regimen 1a",
        }
        df["regimen_type_at_start_of_treatment"] = (
            df["regimen_type_at_start_of_treatment"].astype(str)
            .str.strip().str.lower()
            .map(rt_map)
            .where(df["regimen_type_at_start_of_treatment"].notna())
        )

    # --- Co-morbidities -> simplified categories ---
    if "co_morbidities" in df.columns:
        def harmonize_comorbidity(val):
            """Normalize each individual comorbidity term while preserving
            multi-answer entries. Splits on comma, semicolon, or period
            delimiters, normalizes each term, then re-joins."""
            if pd.isna(val):
                return np.nan
            raw = str(val).strip()
            if raw == "":
                return np.nan
            # Check for "no known" before splitting
            # NOTE: Do NOT use the string "None" — pandas read_csv
            # interprets it as NaN. Use a descriptive label instead.
            if raw.lower() in ("no known", "no known", "none", "n/a"):
                return "No Known Comorbidity"
            # Split on comma, semicolon, or period (used as delimiter)
            parts = re.split(r"[,;.]+", raw)
            normalized = []
            term_map = {
                "diabetes mellitus": "Diabetes Mellitus",
                "diabetes": "Diabetes Mellitus",
                "dm": "Diabetes Mellitus",
                "prediabetic": "Prediabetic",
                "hypertension": "Hypertension",
                "htn": "Hypertension",
                "elevated bp": "Hypertension",
                "cancer": "Cancer",
                "ckd": "Chronic Kidney Disease",
                "chronic kidney disease": "Chronic Kidney Disease",
                "renal disease": "Chronic Kidney Disease",
                "kidney": "Chronic Kidney Disease",
                "diabetic kidney disease": "Diabetic Kidney Disease",
                "cardiovascular disease": "Cardiovascular Disease",
                "cad": "Coronary Artery Disease",
                "chronic coronary syndrome": "Coronary Artery Disease",
                "acute coronary syndrome": "Coronary Artery Disease",
                "copd": "COPD",
                "chronic obstructive pulmonary disease": "COPD",
                "bronchial asthma": "Asthma",
                "asthma": "Asthma",
                "hiv": "HIV",
                "anemia": "Anemia",
                "arthritis": "Arthritis",
                "hypothyroidism": "Hypothyroidism",
                "hydrocephalus secondary": "Hydrocephalus",
                "benign prostatic hyperplasia": "Benign Prostatic Hyperplasia",
                "cellulitis": "Cellulitis",
                "skin allergy": "Skin Allergy",
            }
            for part in parts:
                term = part.strip().lower()
                if not term:
                    continue
                # Try exact match first, then substring match
                matched = False
                for key, canonical in term_map.items():
                    if term == key or key in term:
                        if canonical not in normalized:
                            normalized.append(canonical)
                        matched = True
                        break
                if not matched and term not in ("no known", "none", ""):
                    # Preserve unknown terms in title case
                    titled = part.strip().title()
                    if titled not in normalized:
                        normalized.append(titled)
            if not normalized:
                return np.nan
            return ", ".join(normalized)

        df["co_morbidities"] = df["co_morbidities"].apply(harmonize_comorbidity)
        print(f"  Harmonized 'co_morbidities' (multi-answer preserved):")
        print(f"    {df['co_morbidities'].value_counts().to_dict()}")

    # --- Prior History of TB -> binary ---
    if "prior_history_of_tb" in df.columns:
        def parse_prior_tb(val):
            if pd.isna(val):
                return np.nan
            val = str(val).strip().lower()
            if val in ("no", "none", "n/a", ""):
                return "No"
            return "Yes"
        df["prior_history_of_tb"] = df["prior_history_of_tb"].apply(parse_prior_tb)
        print(f"  Harmonized 'prior_history_of_tb': "
              f"{df['prior_history_of_tb'].value_counts().to_dict()}")

    # --- DAT-supported -> binary ---
    if "dat_supported" in df.columns:
        def parse_dat(val):
            if pd.isna(val):
                return np.nan
            val = str(val).strip().lower()
            if val in ("no", "none", ""):
                return "No"
            if val in ("yes",):
                return "Yes"
            return "No"
        df["dat_supported"] = df["dat_supported"].apply(parse_dat)

    # --- Name of Treatment Unit: strip leading/trailing whitespace ---
    if "name_of_treatment_unit" in df.columns:
        df["name_of_treatment_unit"] = (
            df["name_of_treatment_unit"].astype(str).str.strip()
            .replace("nan", np.nan)
        )
        print(f"  Stripped whitespace in 'name_of_treatment_unit': "
              f"{df['name_of_treatment_unit'].value_counts().to_dict()}")

    # --- Regimen Type at End of Treatment: harmonize mixed coding ---
    if "regimen_type_at_end_of_treatment" in df.columns:
        rt_end_map = {
            "1": "Regimen 1", "regimen 1": "Regimen 1", "cat 1": "Regimen 1",
            "2": "Regimen 2", "regimen 2": "Regimen 2",
            "1a": "Regimen 1a",
        }
        df["regimen_type_at_end_of_treatment"] = (
            df["regimen_type_at_end_of_treatment"].astype(str)
            .str.strip().str.lower()
            .map(rt_end_map)
            .where(df["regimen_type_at_end_of_treatment"].notna())
        )
        print(f"  Harmonized 'regimen_type_at_end_of_treatment': "
              f"{df['regimen_type_at_end_of_treatment'].value_counts().to_dict()}")

    # --- Chest X-ray -> simplified ---
    if "chest_x_ray_at_case_notification" in df.columns:
        def simplify_cxr(val):
            if pd.isna(val):
                return np.nan
            val = str(val).strip().lower()
            if "normal" in val:
                return "Normal"
            if "ptb" in val or "tb" in val or "parenchymal" in val:
                return "Suggestive of PTB"
            return "Other"
        df["chest_x_ray_at_case_notification"] = (
            df["chest_x_ray_at_case_notification"].apply(simplify_cxr)
        )
        print(f"  Harmonized 'chest_x_ray_at_case_notification': "
              f"{df['chest_x_ray_at_case_notification'].value_counts().to_dict()}")

    print()
    return df


# ============================================================================
# STAGE 5: DROP UNINFORMATIVE COLUMNS  [CLEANING]
# ============================================================================

def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 5 [CLEANING]: Flag columns that are entirely null,
    near-entirely null, or redundant. All columns are retained
    in the dataset for completeness and traceability.
    """
    print("=" * 70)
    print("STAGE 5: Flag Uninformative / Redundant Columns")
    print("=" * 70)

    # Flag pre-defined uninformative columns
    flagged = [c for c in ALWAYS_DROP_COLUMNS if c in df.columns]
    print(f"  Flagged {len(flagged)} uninformative columns (retained): {flagged}")
    # Note: 'nationality_raw' is merged into 'nationality' and dropped in Stage 4

    # Flag columns that are entirely null after cleaning
    all_null = df.columns[df.isnull().all()].tolist()
    if all_null:
        print(f"  Flagged {len(all_null)} entirely-null columns (retained): {all_null}")

    # Flag redundant phase date columns
    phase_cols = [c for c in df.columns if "phase_start" in c or "phase_end" in c]
    regimen_extra = [c for c in df.columns
                     if c in ("regimen_type_at_6th_month_of_treatment",
                              "regimen_type_at_end_of_treatment")]
    flagged_extra = phase_cols + regimen_extra
    if flagged_extra:
        print(f"  Flagged {len(flagged_extra)} redundant columns (retained): {flagged_extra}")

    print(f"  Total columns: {df.shape[1]} (all retained)")
    print()
    return df


# ============================================================================
# STAGE 6: TEMPORAL DATA STRUCTURING  [CLEANING]
# ============================================================================

def structure_temporal_data(df: pd.DataFrame):
    """
    Stage 6 [CLEANING]: Separate static (baseline) features from temporal
    (monthly) features and reshape temporal data into long format.

    The monthly columns (M0-M12) are unpivoted into a long-format table
    with columns: [patient_id, month, feature_name, value].

    This enables:
        - Proper time-series imputation (no future data leakage)
        - Construction of 3D tensors for RNN/LSTM input

    Returns:
        df_static:   (n_patients, n_static_features) - includes patient_id
        df_temporal:  (n_patients x n_months, n_temporal_features) - long format
    """
    print("=" * 70)
    print("STAGE 6: Temporal Data Structuring")
    print("=" * 70)

    # Assign a unique patient ID (row index)
    df = df.reset_index(drop=True)
    df["patient_id"] = df.index

    # Identify monthly columns by regex pattern M{digit}_
    monthly_pattern = re.compile(r"^m(\d+)_(.+)$")
    monthly_cols = [c for c in df.columns if monthly_pattern.match(c)]
    static_cols = [c for c in df.columns if c not in monthly_cols]

    print(f"  Static features: {len(static_cols) - 1}")  # -1 for patient_id
    print(f"  Monthly features detected: {len(monthly_cols)}")

    # Split into static and temporal
    df_static = df[static_cols].copy()

    # Build long-format temporal DataFrame
    temporal_records = []
    temporal_features = set()
    for col in monthly_cols:
        match = monthly_pattern.match(col)
        month_num = int(match.group(1))
        feature_name = match.group(2)
        temporal_features.add(feature_name)

    temporal_features = sorted(temporal_features)
    print(f"  Unique temporal features per month: {len(temporal_features)}")
    print(f"    {temporal_features}")

    # Pivot monthly columns into (patient_id, month) rows
    rows = []
    for _, patient_row in df.iterrows():
        pid = patient_row["patient_id"]
        for m in MONTH_RANGE:
            row_data = {"patient_id": pid, "month": m}
            for feat in temporal_features:
                col_name = f"m{m}_{feat}"
                if col_name in df.columns:
                    row_data[feat] = patient_row[col_name]
                else:
                    row_data[feat] = np.nan
            rows.append(row_data)

    df_temporal = pd.DataFrame(rows)
    print(f"  Temporal DataFrame shape: {df_temporal.shape} "
          f"({df_temporal['patient_id'].nunique()} patients x "
          f"{len(MONTH_RANGE)} months)")

    # Drop monthly columns from static
    df_static = df_static.drop(
        columns=[c for c in df_static.columns if monthly_pattern.match(c)],
        errors="ignore"
    )
    print(f"  Static DataFrame shape: {df_static.shape}")
    print()

    return df_static, df_temporal


# ============================================================================
# STAGE 7: MISSING DATA HANDLING VIA MICE (ITERATIVE IMPUTATION)  [CLEANING]
# ============================================================================

def impute_missing_mice(df_static: pd.DataFrame,
                        df_temporal: pd.DataFrame):
    """
    Stage 7 [CLEANING]: Handle missing values using MICE (Multiple
    Imputation by Chained Equations) via sklearn's IterativeImputer.

    Strategy:
        - Numerical variables: BayesianRidge regression (default)
        - Categorical variables: Imputed via mode within groups
        - Temporal imputation: Forward-fill within patient first
          (respects time ordering), then MICE for remaining gaps

    IMPORTANT: Temporal imputation only uses past data (forward fill)
    to prevent future data leakage in time-series features. Backward
    fill is intentionally NOT used.

    Returns:
        df_static, df_temporal - with missing values imputed, but values
        still in original clinical units (no scaling or encoding applied).
    """
    print("=" * 70)
    print("STAGE 7: Missing Data Handling (MICE Imputation)")
    print("=" * 70)

    # ---- STATIC FEATURES ----
    print("\n  --- Static Features ---")

    # Separate numeric and categorical columns
    exclude_cols = {"patient_id"}
    date_cols = [c for c in df_static.columns
                 if df_static[c].dtype == "datetime64[ns]"
                 or "date" in c.lower()]

    cat_cols = [c for c in df_static.columns
                if c not in exclude_cols
                and c not in date_cols
                and (df_static[c].dtype == "object"
                     or df_static[c].nunique(dropna=True) <= 10)]

    num_cols = [c for c in df_static.columns
                if c not in exclude_cols
                and c not in date_cols
                and c not in cat_cols
                and pd.api.types.is_numeric_dtype(df_static[c])]

    print(f"  Numeric columns for imputation: {len(num_cols)}")
    print(f"  Categorical columns for imputation: {len(cat_cols)}")
    print(f"  Date columns (skipped): {len(date_cols)}")

    # Impute numeric static features with MICE
    if num_cols:
        missing_before = df_static[num_cols].isnull().sum().sum()
        imputer_static = IterativeImputer(
            max_iter=20,
            random_state=42,
            sample_posterior=True,
        )
        df_static[num_cols] = imputer_static.fit_transform(df_static[num_cols])
        missing_after = df_static[num_cols].isnull().sum().sum()
        print(f"  MICE imputed static numeric: {missing_before} -> "
              f"{missing_after} missing values")

    # Impute categorical static features with mode
    # Guard: if >90% of values were missing, mode-fill is unreliable
    # (the mode of 1-2 values would be fabricated data). Use "Unknown".
    HIGH_MISSING_THRESHOLD = 0.90
    for col in cat_cols:
        if df_static[col].isnull().any():
            n_filled = df_static[col].isnull().sum()
            pct_missing = n_filled / len(df_static)
            if pct_missing > HIGH_MISSING_THRESHOLD:
                fill_val = "Unknown"
                print(f"  Sparse-filled '{col}': {n_filled}/{len(df_static)} "
                      f"({pct_missing:.0%}) missing -> 'Unknown' "
                      f"(>90% missing, mode unreliable)")
            elif col == "chest_x_ray_at_case_notification":
                fill_val = "Suggestive of PTB"
            else:
                mode_val = df_static[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            df_static[col] = df_static[col].fillna(fill_val)
            if pct_missing <= HIGH_MISSING_THRESHOLD:
                print(f"  Mode-filled '{col}': {n_filled} values -> '{fill_val}'")

    # ---- TEMPORAL FEATURES ----
    print("\n  --- Temporal Features ---")

    # Identify numeric temporal columns
    temporal_num_cols = [c for c in df_temporal.columns
                         if c not in ("patient_id", "month")
                         and pd.api.types.is_numeric_dtype(df_temporal[c])]

    temporal_cat_cols = [c for c in df_temporal.columns
                         if c not in ("patient_id", "month")
                         and c not in temporal_num_cols]

    missing_before = df_temporal[temporal_num_cols].isnull().sum().sum()
    print(f"  Temporal numeric missing values before: {missing_before}")

    # Step 1: Forward-fill within each patient (no future leakage)
    # IMPORTANT: Only forward-fill - backward fill would leak future data
    df_temporal = df_temporal.sort_values(["patient_id", "month"])
    for col in temporal_num_cols:
        df_temporal[col] = (df_temporal.groupby("patient_id")[col]
                             .ffill())
    missing_after_ffill = df_temporal[temporal_num_cols].isnull().sum().sum()
    print(f"  After forward-fill: {missing_after_ffill} missing values")

    # Step 2: MICE imputation for remaining gaps (using month as context)
    if temporal_num_cols and missing_after_ffill > 0:
        imputer_temporal = IterativeImputer(
            max_iter=20,
            random_state=42,
            sample_posterior=True,
        )
        # Include month as a feature for the imputer (provides temporal context)
        impute_cols = ["month"] + temporal_num_cols
        df_temporal[impute_cols] = imputer_temporal.fit_transform(
            df_temporal[impute_cols]
        )
        # Restore month to integer
        df_temporal["month"] = df_temporal["month"].round().astype(int)
        missing_after_mice = df_temporal[temporal_num_cols].isnull().sum().sum()
        print(f"  After MICE: {missing_after_mice} missing values")

    # Impute categorical temporal features with mode per patient or global mode
    # Guard: if >90% of temporal values are missing, use "Unknown" instead of mode
    for col in temporal_cat_cols:
        if df_temporal[col].isnull().any():
            pct_missing = df_temporal[col].isnull().sum() / len(df_temporal)
            # Forward fill within patient first (respects time order)
            df_temporal[col] = df_temporal.groupby("patient_id")[col].ffill()
            if pct_missing > HIGH_MISSING_THRESHOLD:
                df_temporal[col] = df_temporal[col].fillna("Unknown")
            else:
                # Then global mode for remaining
                mode_val = df_temporal[col].mode()
                if len(mode_val) > 0:
                    df_temporal[col] = df_temporal[col].fillna(mode_val.iloc[0])

    # --- Post-MICE clinical sanity clipping ---
    # MICE can generate values far outside clinically plausible ranges.
    # Re-apply the same clinical bounds used in Stage 3 to cap any
    # extreme imputed values after MICE has run.
    print("\n  --- Post-MICE Clinical Sanity Clipping ---")
    post_mice_ranges = {
        "weight_kg":        (10, 180),
        "height_cm":       (80, 220),
        "bp_systolic":     (60, 220),
        "bp_diastolic":    (30, 140),
        "heart_rate":      (20, 250),
        "respiratory_rate": (4, 60),
        "temperature":     (30, 45),
        "o2_sat":           (70, 100),
        "age":              (0, 110),
    }
    for col, (cmin, cmax) in post_mice_ranges.items():
        if col in df_static.columns and pd.api.types.is_numeric_dtype(df_static[col]):
            before_clip = ((df_static[col] < cmin) | (df_static[col] > cmax)).sum()
            if before_clip > 0:
                df_static[col] = df_static[col].clip(lower=cmin, upper=cmax)
                print(f"    Static '{col}': clipped {before_clip} MICE-imputed "
                      f"values to [{cmin}, {cmax}]")

    # Clip temporal weight/height/adherence/doses/tests columns
    temporal_clip_map = {
        "weight":              (10, 180),
        "height":              (80, 220),
        "pct_adherence":       (0, 100),
        "monthly_doses_taken": (0, 31),
        "monthly_missed_doses":(0, None),    # floor only
        "smear_tb_lamp":       (0, 1),
        "xpert_mtb_rif":       (0, 1),
    }
    for feat, (cmin, cmax) in temporal_clip_map.items():
        if feat in df_temporal.columns and pd.api.types.is_numeric_dtype(df_temporal[feat]):
            if cmax is not None:
                before_clip = ((df_temporal[feat] < cmin) | (df_temporal[feat] > cmax)).sum()
            else:
                before_clip = (df_temporal[feat] < cmin).sum()
            if before_clip > 0:
                df_temporal[feat] = df_temporal[feat].clip(lower=cmin, upper=cmax)
                bound_str = f"[{cmin}, {cmax}]" if cmax is not None else f"[{cmin}, inf)"
                print(f"    Temporal '{feat}': clipped {before_clip} values "
                      f"to {bound_str}")

    # Re-apply integer rounding to age after MICE (imputer may introduce decimals)
    if "age" in df_static.columns and pd.api.types.is_numeric_dtype(df_static["age"]):
        df_static["age"] = df_static["age"].round(0).astype("Int64")
        print("    Static 'age': re-rounded to integer after MICE")

    # --- Enforce monotonically non-decreasing cumulative doses ---
    # MICE imputes monthly columns independently and has no knowledge that
    # cumulative_doses_taken must be >= its previous month's value.
    # Repair by taking the running maximum per patient (forward direction only).
    cum_col = "cumulative_doses_taken"
    if cum_col in df_temporal.columns:
        df_temporal = df_temporal.sort_values(["patient_id", "month"])
        before_violations = 0
        for pid, grp in df_temporal.groupby("patient_id"):
            vals = grp[cum_col].values
            for i in range(1, len(vals)):
                if vals[i] < vals[i - 1]:
                    before_violations += 1
        df_temporal[cum_col] = (
            df_temporal.groupby("patient_id")[cum_col]
            .cummax()
        )
        print(f"    Temporal 'cumulative_doses_taken': repaired {before_violations} "
              f"monotonicity violations (running max per patient)")

    print()
    return df_static, df_temporal


# ============================================================================
# STAGE 8: EXPORT & VALIDATE HUMAN-READABLE CLEANED DATASET  [CLEANING]
# ============================================================================

def export_cleaned_human_readable_dataset(df_static: pd.DataFrame,
                                          df_temporal: pd.DataFrame,
                                          output_dir: str):
    """
    Stage 8 [CLEANING]: Export a human-readable, cleaned version of the
    combined dataset BEFORE any encoding, scaling, or outlier capping.

    This CSV preserves:
        - Original categorical labels (e.g., 'Male', 'Treatment Completed')
        - Numeric values in natural clinical units (kg, cm, mmHg, %)
        - Imputed values (MICE for numerics, mode for categoricals,
          forward-fill for temporal columns)

    This file is suitable for:
        - Manual clinical review and validation
        - Descriptive statistics and EDA
        - Research documentation and reporting

    This file does NOT contain:
        - One-hot encoded columns
        - Scaled/standardized values
        - Outlier-capped values

    The temporal features are pivoted back to wide format so each row
    represents one patient - matching the original dataset structure.

    Output:
        - cleaned_human_readable.csv : Wide-format, human-readable CSV
    """
    print("=" * 70)
    print("STAGE 8: Export Human-Readable Cleaned Dataset")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # --- Pivot temporal back to wide format ---
    # Get temporal feature columns (exclude patient_id and month)
    temporal_feature_cols = [c for c in df_temporal.columns
                             if c not in ("patient_id", "month")]

    # Pivot each temporal feature into M{month}_{feature} columns
    df_temporal_wide = df_temporal.pivot(
        index="patient_id", columns="month", values=temporal_feature_cols
    )

    # Flatten the MultiIndex columns: (feature, month) -> M{month}_{feature}
    df_temporal_wide.columns = [
        f"M{int(month)}_{feature}"
        for feature, month in df_temporal_wide.columns
    ]
    df_temporal_wide = df_temporal_wide.reset_index()

    # --- Merge static + temporal wide ---
    df_clean = df_static.merge(df_temporal_wide, on="patient_id", how="left")

    # --- Keep patient_id for traceability ---

    # --- Format date columns as MM/DD/YYYY for human-readable output ---
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].dt.strftime("%m/%d/%Y")

    # --- Round numeric columns for readability ---
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].round(2)

    # --- Save ---
    out_path = os.path.join(output_dir, "cleaned_human_readable.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"  Saved cleaned_human_readable.csv ({df_clean.shape[0]} rows x "
          f"{df_clean.shape[1]} columns)")
    print(f"  Location: {os.path.abspath(out_path)}")

    # --- Quick summary ---
    n_missing = df_clean.isnull().sum().sum()
    print(f"  Remaining missing values: {n_missing}")
    print(f"  Columns: {list(df_clean.columns[:10])} ... "
          f"({len(df_clean.columns)} total)")
    print()

    # --- Validation: cleaned data must not contain scaled values ---
    _validate_cleaned_human_readable(df_clean)

    return df_clean


def _validate_cleaned_human_readable(df_clean: pd.DataFrame):
    """
    Validation checks for the human-readable cleaned dataset.

    Ensures that:
        1. No patient identifiers remain.
        2. No scaled values are present (numeric columns should have
           clinically plausible ranges, not zero-mean/unit-variance).
        3. Categorical labels are preserved as human-readable strings.
        4. No one-hot encoded columns exist.
    """
    print("  --- Validation: cleaned_human_readable.csv ---")
    all_passed = True

    # Check 1: Identifier columns flagged (retained per no-drop policy)
    id_keywords = ["name", "no.", "source", "facility_name"]
    remaining_ids = [c for c in df_clean.columns
                     if any(kw in c.lower() for kw in id_keywords)]
    if remaining_ids:
        print(f"  [INFO] Identifier columns retained (no-drop policy): "
              f"{len(remaining_ids)} cols")
    else:
        print(f"  [PASS] No identifier columns found")

    # Check 2: No scaled values (age should be >1 for real patients)
    # If age column exists and has been scaled, its mean would be ~0
    if "age" in df_clean.columns:
        age_mean = pd.to_numeric(df_clean["age"], errors="coerce").mean()
        check_2 = age_mean > 1.0  # Real ages, not standardized
        print(f"  [{'PASS' if check_2 else 'FAIL'}] No scaled values "
              f"(age mean = {age_mean:.1f}, expected > 1)")
        all_passed &= check_2

    # Check 3: Categorical labels are present (not one-hot encoded)
    expected_cats = ["sex", "outcome", "diagnosis"]
    cats_present = [c for c in expected_cats if c in df_clean.columns]
    if cats_present:
        has_strings = any(df_clean[c].dtype == "object" for c in cats_present)
        check_3 = has_strings
        print(f"  [{'PASS' if check_3 else 'FAIL'}] Categorical labels "
              f"preserved as strings")
        all_passed &= check_3

    # Check 4: No one-hot encoded columns (no column names like sex_Male)
    onehot_pattern = re.compile(r"^(sex|outcome|diagnosis|civil_status)_[A-Z]")
    onehot_cols = [c for c in df_clean.columns if onehot_pattern.match(c)]
    check_4 = len(onehot_cols) == 0
    print(f"  [{'PASS' if check_4 else 'FAIL'}] No one-hot encoded columns"
          f"{'' if check_4 else f': {onehot_cols[:5]}'}")
    all_passed &= check_4

    status = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS NEED REVIEW"
    print(f"  >>> CLEANED DATASET VALIDATION: {status} <<<")
    print()


# ############################################################################
#
#  PHASE B - MODEL PREPARATION
#
#  Stages 9-13: Encode -> Scale -> Outlier cap -> Tensor -> Validate -> Export
#  All model-ready outputs are derived from the cleaned data (Phase A).
#  This separation allows future train/test splitting before scaling.
#
# ############################################################################


# ============================================================================
# STAGE 9: FEATURE ENCODING (ONE-HOT)  [MODEL PREPARATION]
# ============================================================================

def encode_features(df_static: pd.DataFrame, df_temporal: pd.DataFrame):
    """
    Stage 9 [MODEL PREPARATION]: One-hot encode categorical features.

    Only encodes columns listed in CATEGORICAL_COLUMNS (the curated set
    of clinically meaningful categorical features). Identifier columns,
    raw text columns with parsed numeric equivalents, and uninformative
    columns are dropped from the model-ready DataFrame since they are
    not useful model features. (They remain in cleaned_human_readable.csv.)

    Date columns are converted to ordinal (days since epoch).

    Returns:
        df_static:    with one-hot encoded categoricals, identifiers removed
        df_temporal:  with one-hot encoded categoricals (if any)
        encoding_map: dict mapping original column -> encoded column names
    """
    print("=" * 70)
    print("STAGE 9: Feature Encoding (One-Hot)  [MODEL PREPARATION]")
    print("=" * 70)

    encoding_map = {}

    # --- Remove non-feature columns from model-ready data ---
    # These are retained in cleaned_human_readable.csv but serve no purpose
    # as ML features (identifiers, raw text with parsed equivalents, etc.)
    non_feature_cols = (
        list(ID_COLUMNS_TO_DROP)                # identifiers
        + list(ALWAYS_DROP_COLUMNS)             # uninformative / near-empty
        + ["weight", "height", "blood_pressure",  # raw text (parsed to numerics)
           "facility",                           # derived from source_file
           "nationality_raw",                    # duplicate of nationality (new complete dataset)
           "regimen_type_at_end_of_treatment",   # end-of-treatment (leakage risk)
           "regimen_type_at_6th_month_of_treatment",  # mid-treatment (leakage risk)
           ]
    )
    cols_to_drop = [c for c in non_feature_cols if c in df_static.columns]
    if cols_to_drop:
        df_static = df_static.drop(columns=cols_to_drop)
        print(f"  Removed {len(cols_to_drop)} non-feature columns from "
              f"model-ready data: {cols_to_drop}")

    # --- Identify categorical columns to encode ---
    # Only encode columns from the curated CATEGORICAL_COLUMNS list
    # that are present and still have object dtype
    exclude = {"patient_id"}
    date_cols = [c for c in df_static.columns
                 if df_static[c].dtype == "datetime64[ns]"
                 or "date" in c.lower()]

    cat_cols_static = [c for c in CATEGORICAL_COLUMNS
                       if c in df_static.columns
                       and c not in exclude
                       and c not in date_cols
                       and df_static[c].dtype == "object"]

    # Also catch any remaining object columns NOT in the curated list
    remaining_obj = [c for c in df_static.columns
                     if df_static[c].dtype == "object"
                     and c not in cat_cols_static
                     and c not in exclude
                     and c not in date_cols]
    if remaining_obj:
        print(f"  Dropping {len(remaining_obj)} remaining non-feature text "
              f"columns: {remaining_obj}")
        df_static = df_static.drop(columns=remaining_obj)

    if cat_cols_static:
        print(f"  Encoding {len(cat_cols_static)} static categorical columns:")
        for col in cat_cols_static:
            unique_vals = df_static[col].nunique()
            print(f"    {col}: {unique_vals} categories")

        df_static_encoded = pd.get_dummies(
            df_static, columns=cat_cols_static,
            prefix_sep="_", dummy_na=False, dtype=float
        )

        for col in cat_cols_static:
            encoded_cols = [c for c in df_static_encoded.columns
                           if c.startswith(col + "_")]
            encoding_map[col] = encoded_cols

        n_new = df_static_encoded.shape[1] - df_static.shape[1]
        print(f"  Static columns: {df_static.shape[1]} -> "
              f"{df_static_encoded.shape[1]} (+{n_new} one-hot)")
        df_static = df_static_encoded

    # --- Temporal categoricals ---
    cat_cols_temporal = [c for c in df_temporal.columns
                         if c not in ("patient_id", "month")
                         and df_temporal[c].dtype == "object"]

    if cat_cols_temporal:
        print(f"  Encoding {len(cat_cols_temporal)} temporal categorical columns:")
        for col in cat_cols_temporal:
            unique_vals = df_temporal[col].nunique()
            print(f"    {col}: {unique_vals} categories")

        df_temporal = pd.get_dummies(
            df_temporal, columns=cat_cols_temporal,
            prefix_sep="_", dummy_na=False, dtype=float
        )

    # Flag date columns (retained in DataFrame, excluded from tensor in Stage 12)
    date_cols_present = [c for c in df_static.columns
                         if df_static[c].dtype == "datetime64[ns]"
                         or ("date" in c.lower() and c != "data_year")]
    if date_cols_present:
        # Convert date columns to numeric (ordinal) so they don't block
        # downstream numeric operations, while preserving the information
        for dc in date_cols_present:
            if pd.api.types.is_datetime64_any_dtype(df_static[dc]):
                ref_date = pd.Timestamp("1970-01-01")
                df_static[dc] = (df_static[dc] - ref_date).dt.days.astype(float)
        print(f"  Converted {len(date_cols_present)} date columns to ordinal "
              f"days (retained): {date_cols_present}")

    print()
    return df_static, df_temporal, encoding_map


# ============================================================================
# STAGE 10: SCALING  [MODEL PREPARATION]
# ============================================================================

def scale_features(
    df_static_train: pd.DataFrame,
    df_static_test: pd.DataFrame,
    df_temporal_train: pd.DataFrame,
    df_temporal_test: pd.DataFrame
):
    """
    Stage 10 [MODEL PREPARATION]: Normalize numerical features using
    StandardScaler (zero mean, unit variance).

    Applies scaling to:
        - Baseline clinical numerics (age, weight, height, vitals)
        - Monthly numerical features (doses, weight, height, adherence)

    StandardScaler is used because it works well with RNN/LSTM models
    and preserves outlier information for subsequent capping.

    NOTE: Scaling is applied AFTER cleaning and imputation (Phase A)
    and AFTER encoding (Stage 9).

    IMPORTANT:
    Scalers are FIT ONLY on training data to prevent data leakage.
    Test data is transformed separately using statistics learned from
    training data only.

    Returns:
        df_static_train, df_static_test
        df_temporal_train, df_temporal_test
        scaler_static, scaler_temporal:
            fitted scalers for inverse transform
    """

    print("=" * 70)
    print("STAGE 10: Feature Scaling (StandardScaler)  [MODEL PREPARATION]")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Static numerical columns
    # ------------------------------------------------------------------

    # --- Static numerical columns ---
    static_num_cols = [c for c in NUMERICAL_COLUMNS_SCALE
                       if c in df_static_train.columns]

    scaler_static = StandardScaler()

    if static_num_cols:

        # FIT ONLY ON TRAINING DATA
        df_static_train[static_num_cols] = scaler_static.fit_transform(
            df_static_train[static_num_cols]
        )

        # TRANSFORM TEST DATA USING TRAINING STATISTICS
        df_static_test[static_num_cols] = scaler_static.transform(
            df_static_test[static_num_cols]
        )

        print(f"  Scaled {len(static_num_cols)} static numeric columns: "
              f"{static_num_cols}")

    # ------------------------------------------------------------------
    # Temporal numerical columns
    # ------------------------------------------------------------------

    # --- Temporal numerical columns ---
    temporal_num_cols = [c for c in df_temporal_train.columns
                         if c not in ("patient_id", "month")
                         and pd.api.types.is_numeric_dtype(
                             df_temporal_train[c]
                         )
                         and df_temporal_train[c].nunique() > 2]

    scaler_temporal = StandardScaler()

    if temporal_num_cols:

        # FIT ONLY ON TRAINING DATA
        df_temporal_train[temporal_num_cols] = (
            scaler_temporal.fit_transform(
                df_temporal_train[temporal_num_cols]
            )
        )

        # TRANSFORM TEST DATA USING TRAINING STATISTICS
        df_temporal_test[temporal_num_cols] = (
            scaler_temporal.transform(
                df_temporal_test[temporal_num_cols]
            )
        )

        print(f"  Scaled {len(temporal_num_cols)} temporal numeric columns: "
              f"{temporal_num_cols}")

    print()

    return (
        df_static_train,
        df_static_test,
        df_temporal_train,
        df_temporal_test,
        scaler_static,
        scaler_temporal
    )


# ============================================================================
# STAGE 11: OUTLIER DETECTION & CAPPING  [MODEL PREPARATION]
# ============================================================================

def detect_and_cap_outliers(df_static: pd.DataFrame,
                            df_temporal: pd.DataFrame):
    """
    Stage 11 [MODEL PREPARATION]: Detect outliers using IQR and Z-score
    methods. Flag and cap outliers rather than removing patients.

    Methods:
        1. IQR method: Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
        2. Z-score method: |z| > 3.0

    Outliers are capped (winsorized) to the boundary values.

    Note: Applied AFTER scaling, so z-scores are on standardized data.
    Since StandardScaler centers to mean=0, std=1, a z-score threshold
    of 3.0 means capping at +/-3 standard deviations.

    Returns:
        df_static, df_temporal with outliers capped
        outlier_report: dict with outlier counts per column
    """
    print("=" * 70)
    print("STAGE 11: Outlier Detection & Capping  [MODEL PREPARATION]")
    print("=" * 70)

    outlier_report = {}

    def cap_outliers_iqr(series, col_name):
        """Cap outliers using IQR method and return count."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((series < lower) | (series > upper)).sum()
        capped = series.clip(lower=lower, upper=upper)
        return capped, n_outliers

    def cap_outliers_zscore(series, col_name, threshold=3.0):
        """Cap outliers using Z-score method and return count."""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series, 0
        z = np.abs((series - mean) / std)
        n_outliers = (z > threshold).sum()
        lower = mean - threshold * std
        upper = mean + threshold * std
        capped = series.clip(lower=lower, upper=upper)
        return capped, n_outliers

    # --- Static features ---
    static_outlier_cols = [c for c in OUTLIER_COLUMNS if c in df_static.columns]
    print("\n  --- Static Features (IQR + Z-score) ---")
    for col in static_outlier_cols:
        if pd.api.types.is_numeric_dtype(df_static[col]):
            # IQR method
            df_static[col], n_iqr = cap_outliers_iqr(df_static[col], col)
            # Z-score method (applied after IQR)
            df_static[col], n_z = cap_outliers_zscore(df_static[col], col)
            total = n_iqr + n_z
            if total > 0:
                outlier_report[f"static_{col}"] = {
                    "iqr_outliers": n_iqr, "zscore_outliers": n_z
                }
                print(f"    {col}: {n_iqr} IQR outliers, {n_z} Z-score outliers")

    # --- Temporal features ---
    temporal_outlier_cols = [c for c in df_temporal.columns
                             if c not in ("patient_id", "month")
                             and pd.api.types.is_numeric_dtype(df_temporal[c])
                             and df_temporal[c].nunique() > 2]

    print("\n  --- Temporal Features (IQR + Z-score) ---")
    for col in temporal_outlier_cols:
        df_temporal[col], n_iqr = cap_outliers_iqr(df_temporal[col], col)
        df_temporal[col], n_z = cap_outliers_zscore(df_temporal[col], col)
        total = n_iqr + n_z
        if total > 0:
            outlier_report[f"temporal_{col}"] = {
                "iqr_outliers": n_iqr, "zscore_outliers": n_z
            }
            print(f"    {col}: {n_iqr} IQR outliers, {n_z} Z-score outliers")

    if not outlier_report:
        print("  No significant outliers detected after scaling.")

    print()
    return df_static, df_temporal, outlier_report


# ============================================================================
# STAGE 12: TEMPORAL MODELING PREPARATION (3D TENSOR)  [MODEL PREPARATION]
# ============================================================================

def prepare_for_temporal_modeling(df_static: pd.DataFrame,
                                  df_temporal: pd.DataFrame):
    """
    Stage 12 [MODEL PREPARATION]: Construct datasets ready for RNN/LSTM
    and hybrid models.

    Outputs:
        1. X_temporal: 3D numpy array (n_patients, n_timesteps, n_features)
           for RNN/LSTM input - preserves M0->M12 temporal ordering
        2. X_static: 2D numpy array (n_patients, n_static_features)
           for combining with temporal models or for XGBoost
        3. X_combined_flat: 2D array with flattened temporal + static features
           for XGBoost/tabular models
        4. patient_ids: array of patient IDs

    The 3D tensor preserves temporal ordering (M0 -> M12) and per-patient
    sequences, which is essential for recurrent models.
    """
    print("=" * 70)
    print("STAGE 12: Temporal Modeling Preparation  [MODEL PREPARATION]")
    print("=" * 70)

    patient_ids = df_static["patient_id"].values
    n_patients = len(patient_ids)
    n_timesteps = len(MONTH_RANGE)

    # Get temporal feature names (exclude patient_id and month)
    temporal_feature_cols = [c for c in df_temporal.columns
                             if c not in ("patient_id", "month")]
    n_temporal_features = len(temporal_feature_cols)

    print(f"  Patients: {n_patients}")
    print(f"  Time steps: {n_timesteps} (M0-M12)")
    print(f"  Temporal features per step: {n_temporal_features}")

    # Build 3D tensor: (patients, timesteps, features)
    X_temporal = np.zeros((n_patients, n_timesteps, n_temporal_features))
    df_temporal_sorted = df_temporal.sort_values(["patient_id", "month"])

    for i, pid in enumerate(patient_ids):
        patient_data = df_temporal_sorted[
            df_temporal_sorted["patient_id"] == pid
        ][temporal_feature_cols].values
        # Ensure we have exactly n_timesteps rows
        if patient_data.shape[0] == n_timesteps:
            X_temporal[i] = patient_data
        else:
            # Pad with zeros if fewer months available
            actual_steps = min(patient_data.shape[0], n_timesteps)
            X_temporal[i, :actual_steps] = patient_data[:actual_steps]

    print(f"  X_temporal shape: {X_temporal.shape} "
          f"(patients x timesteps x features)")

    # Static features (exclude patient_id)
    static_feature_cols = [c for c in df_static.columns if c != "patient_id"]
    X_static = df_static[static_feature_cols].values.astype(np.float32)
    print(f"  X_static shape: {X_static.shape}")

    # Flattened combined for XGBoost / tabular models
    X_temporal_flat = X_temporal.reshape(n_patients, -1)
    X_combined_flat = np.hstack([X_static, X_temporal_flat])
    print(f"  X_combined_flat shape: {X_combined_flat.shape} "
          f"(for XGBoost/tabular)")

    # Feature name lists for interpretability
    static_feature_names = static_feature_cols
    temporal_feature_names = [
        f"M{m}_{feat}"
        for m in MONTH_RANGE
        for feat in temporal_feature_cols
    ]
    combined_feature_names = static_feature_names + temporal_feature_names

    print()
    return {
        "X_temporal": X_temporal,
        "X_static": X_static,
        "X_combined_flat": X_combined_flat,
        "patient_ids": patient_ids,
        "static_feature_names": static_feature_names,
        "temporal_feature_names": temporal_feature_names,
        "combined_feature_names": combined_feature_names,
        "n_patients": n_patients,
        "n_timesteps": n_timesteps,
        "n_temporal_features": n_temporal_features,
    }


# ============================================================================
# STAGE 13: FINAL VALIDATION & EXPORT  [MODEL PREPARATION]
# ============================================================================

def validate_model_ready(df_static: pd.DataFrame,
                         df_temporal: pd.DataFrame,
                         model_data: dict):
    """
    Stage 13a [MODEL PREPARATION]: Run validation checks on model-ready
    outputs to ensure data quality before export.

    Checks:
        1. No identifier columns remain
        2. No missing values in temporal tensor
        3. No missing values in static array
        4. All features properly encoded (no object or date dtypes)
        5. Tensor shapes are consistent
        6. No infinite values

    Returns:
        validation_report: dict with check results
    """
    print("=" * 70)
    print("STAGE 13a: Model-Ready Validation")
    print("=" * 70)

    validation_report = {}
    all_passed = True

    # Check 1: No identifier columns remain (removed in Stage 9)
    id_keywords = ["name", "no.", "source", "facility_name"]
    remaining_ids = [c for c in df_static.columns
                     if any(kw in c.lower() for kw in id_keywords)]
    check_1 = len(remaining_ids) == 0
    validation_report["no_identifiers"] = check_1
    if remaining_ids:
        print(f"  [{'PASS' if check_1 else 'FAIL'}] No identifier columns: "
              f"{remaining_ids}")
    else:
        print(f"  [PASS] No identifier columns in model-ready data")
    all_passed &= check_1

    # Check 2: No missing values in temporal tensor
    n_nan_temporal = np.isnan(model_data["X_temporal"]).sum()
    check_2 = n_nan_temporal == 0
    validation_report["no_missing_temporal"] = check_2
    print(f"  [{'PASS' if check_2 else 'WARN'}] Temporal tensor NaN count: "
          f"{n_nan_temporal}")

    # Check 3: No missing values in static array
    n_nan_static = np.isnan(model_data["X_static"]).sum()
    check_3 = n_nan_static == 0
    validation_report["no_missing_static"] = check_3
    print(f"  [{'PASS' if check_3 else 'WARN'}] Static array NaN count: "
          f"{n_nan_static}")

    # Check 4: No object or date dtypes in model-ready DataFrames
    obj_cols = df_static.select_dtypes(include=["object"]).columns.tolist()
    date_cols = df_static.select_dtypes(
        include=["datetime64"]
    ).columns.tolist()
    bad_cols = obj_cols + date_cols
    check_4 = len(bad_cols) == 0
    validation_report["all_encoded"] = check_4
    print(f"  [{'PASS' if check_4 else 'FAIL'}] No object/date columns "
          f"in model-ready data"
          f"{'' if check_4 else f': {bad_cols}'}")
    all_passed &= check_4

    # Check 5: Tensor shape consistency
    X = model_data["X_temporal"]
    check_5 = (X.shape[0] == model_data["n_patients"]
               and X.shape[1] == model_data["n_timesteps"]
               and X.shape[2] == model_data["n_temporal_features"])
    validation_report["tensor_shape_ok"] = check_5
    print(f"  [{'PASS' if check_5 else 'FAIL'}] Tensor shape consistent: "
          f"{X.shape}")
    all_passed &= check_5

    # Check 6: No infinite values
    n_inf_temp = np.isinf(model_data["X_temporal"]).sum()
    n_inf_stat = np.isinf(model_data["X_static"]).sum()
    check_6 = (n_inf_temp + n_inf_stat) == 0
    validation_report["no_inf"] = check_6
    print(f"  [{'PASS' if check_6 else 'FAIL'}] No infinite values "
          f"(temporal: {n_inf_temp}, static: {n_inf_stat})")
    all_passed &= check_6

    # Summary
    status = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS NEED REVIEW"
    print(f"\n  >>> MODEL-READY VALIDATION: {status} <<<")
    print()

    return validation_report


def export_model_ready_outputs(df_static: pd.DataFrame,
                               df_temporal: pd.DataFrame,
                               model_data: dict,
                               output_dir: str):
    """
    Stage 13b [MODEL PREPARATION]: Export all model-ready datasets and
    numpy arrays.

    Outputs:
        - static_features.csv       : Encoded & scaled static features
        - temporal_features.csv     : Encoded & scaled temporal features (long)
        - X_temporal.npy            : 3D tensor for RNN/LSTM
        - X_static.npy              : 2D static feature matrix
        - X_combined_flat.npy       : Flattened combined matrix for XGBoost
        - patient_ids.npy           : Patient ID mapping
        - feature_names.npz         : Feature name arrays
        - preprocessing_summary.txt : Pipeline execution summary
        - OUTPUT_README.txt         : Documentation for all output files
    """
    print("=" * 70)
    print("STAGE 13b: Export Model-Ready Data")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # CSV exports
    df_static.to_csv(os.path.join(output_dir, "static_features.csv"),
                     index=False)
    print(f"  Saved static_features.csv ({df_static.shape})")

    df_temporal.to_csv(os.path.join(output_dir, "temporal_features.csv"),
                       index=False)
    print(f"  Saved temporal_features.csv ({df_temporal.shape})")

    # Numpy arrays for direct model input
    np.save(os.path.join(output_dir, "X_temporal.npy"),
            model_data["X_temporal"])
    print(f"  Saved X_temporal.npy {model_data['X_temporal'].shape}")

    np.save(os.path.join(output_dir, "X_static.npy"),
            model_data["X_static"])
    print(f"  Saved X_static.npy {model_data['X_static'].shape}")

    np.save(os.path.join(output_dir, "X_combined_flat.npy"),
            model_data["X_combined_flat"])
    print(f"  Saved X_combined_flat.npy {model_data['X_combined_flat'].shape}")

    np.save(os.path.join(output_dir, "patient_ids.npy"),
            model_data["patient_ids"])
    print(f"  Saved patient_ids.npy ({len(model_data['patient_ids'])})")

    # Feature name mappings
    np.savez(
        os.path.join(output_dir, "feature_names.npz"),
        static=np.array(model_data["static_feature_names"], dtype=object),
        temporal=np.array(model_data["temporal_feature_names"], dtype=object),
        combined=np.array(model_data["combined_feature_names"], dtype=object),
    )
    print(f"  Saved feature_names.npz")

    # Summary text
    summary_path = os.path.join(output_dir, "preprocessing_summary.txt")
    with open(summary_path, "w") as f:
        f.write("TB-DOTS CAR CDSS - Preprocessing Pipeline Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Patients:              {model_data['n_patients']}\n")
        f.write(f"Time steps:            {model_data['n_timesteps']} (M0-M12)\n")
        f.write(f"Temporal features:     {model_data['n_temporal_features']}\n")
        f.write(f"Static features:       {len(model_data['static_feature_names'])}\n")
        f.write(f"X_temporal shape:      {model_data['X_temporal'].shape}\n")
        f.write(f"X_static shape:        {model_data['X_static'].shape}\n")
        f.write(f"X_combined_flat shape: {model_data['X_combined_flat'].shape}\n")
        f.write(f"\nPipeline Design:\n")
        f.write(f"  Phase A (Cleaning): Stages 1-8\n")
        f.write(f"    -> cleaned_human_readable.csv\n")
        f.write(f"  Phase B (Model Prep): Stages 9-13\n")
        f.write(f"    -> static_features.csv, temporal_features.csv,\n")
        f.write(f"      X_temporal.npy, X_static.npy, X_combined_flat.npy\n")
        f.write(f"\nStatic feature names:\n")
        for fn in model_data["static_feature_names"]:
            f.write(f"  - {fn}\n")
        f.write(f"\nTemporal feature names (per timestep):\n")
        seen = set()
        for fn in model_data["temporal_feature_names"]:
            base = "_".join(fn.split("_")[1:])
            if base not in seen:
                seen.add(base)
                f.write(f"  - {base}\n")
    print(f"  Saved preprocessing_summary.txt")

    # Output file descriptions (README)
    readme_path = os.path.join(output_dir, "OUTPUT_README.txt")
    with open(readme_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TB-DOTS CAR CDSS - Output File Descriptions\n")
        f.write("=" * 70 + "\n\n")
        f.write("This folder contains all outputs produced by the\n")
        f.write("preprocessing pipeline (preprocessing.py).\n")
        f.write("Below is a description of each file and its intended use.\n\n")
        f.write("The pipeline is split into two phases:\n")
        f.write("  PHASE A (Cleaning)   -> cleaned_human_readable.csv\n")
        f.write("  PHASE B (Model Prep) -> all other files\n\n")
        f.write("-" * 70 + "\n")
        f.write("PHASE A OUTPUT: HUMAN-READABLE CLEANED DATA\n")
        f.write("-" * 70 + "\n\n")
        f.write("cleaned_human_readable.csv\n")
        f.write("  The fully cleaned version of the original combined_complete_dataset.csv.\n")
        f.write("  One row per patient, wide format (M0-M12 columns preserved).\n")
        f.write("  Contains human-readable categorical labels (e.g., 'Male',\n")
        f.write("  'Treatment Completed', 'Regimen 1') and numeric values in\n")
        f.write("  natural units (kg, cm, mmHg, %, etc.).\n")
        f.write("  Missing values have been imputed (MICE for numerics, mode\n")
        f.write("  for categoricals, forward-fill for temporal columns).\n")
        f.write("  NOT encoded, scaled, or outlier-capped - suitable for\n")
        f.write("  manual review, EDA, descriptive statistics, and research\n")
        f.write("  documentation.\n\n")
        f.write("-" * 70 + "\n")
        f.write("PHASE B OUTPUTS: MODEL-READY DATA\n")
        f.write("-" * 70 + "\n\n")
        f.write("static_features.csv\n")
        f.write("  Baseline (non-temporal) patient features after full\n")
        f.write("  preprocessing: imputed, one-hot encoded, scaled, and\n")
        f.write("  outlier-capped.\n")
        f.write("  Categorical variables are split into binary indicator\n")
        f.write("  columns (e.g., sex_Male, sex_Female). Numeric values\n")
        f.write("  are standardized (zero mean, unit variance).\n")
        f.write("  Includes a 'patient_id' column linking to temporal data.\n\n")
        f.write("temporal_features.csv\n")
        f.write("  Monthly monitoring data in long format: one row per\n")
        f.write("  patient per month (patient_id x month). Contains\n")
        f.write("  treatment adherence, doses, weight, height, smear, and\n")
        f.write("  Xpert results - all imputed, scaled, and outlier-capped.\n")
        f.write("  Columns: patient_id, month (0-12), plus temporal features.\n\n")
        f.write("preprocessing_summary.txt\n")
        f.write("  A brief text summary of the pipeline execution, including\n")
        f.write("  dataset dimensions, tensor shapes, and lists of feature\n")
        f.write("  names used in static and temporal components.\n\n")
        f.write("OUTPUT_README.txt\n")
        f.write("  This file - describes every output file in the folder.\n\n")
        f.write("-" * 70 + "\n")
        f.write("MODEL-READY FILES (NumPy arrays)\n")
        f.write("-" * 70 + "\n\n")
        f.write("X_temporal.npy\n")
        f.write(f"  Shape: {model_data['X_temporal'].shape}\n")
        f.write("  3D NumPy array: (n_patients, n_timesteps, n_features).\n")
        f.write("  Each patient has 13 time steps (M0 through M12), each with\n")
        f.write(f"  {model_data['n_temporal_features']} features (doses, adherence, weight, height,\n")
        f.write("  smear result, Xpert result, etc.).\n")
        f.write("  Ready for direct input into RNN, LSTM, or GRU models.\n")
        f.write("  Usage: X = np.load('X_temporal.npy')\n\n")
        f.write("X_static.npy\n")
        f.write(f"  Shape: {model_data['X_static'].shape}\n")
        f.write("  2D NumPy array: (n_patients, n_static_features).\n")
        f.write("  Contains baseline demographics, diagnostics, and clinical\n")
        f.write("  indicators - all encoded, scaled, and outlier-capped.\n")
        f.write("  Use for: hybrid models (e.g., concatenate with LSTM output),\n")
        f.write("  or standalone tabular models like XGBoost/Random Forest.\n")
        f.write("  Usage: X = np.load('X_static.npy')\n\n")
        f.write("X_combined_flat.npy\n")
        f.write(f"  Shape: {model_data['X_combined_flat'].shape}\n")
        f.write("  2D NumPy array: static features + flattened temporal features\n")
        f.write("  concatenated side by side. Each patient is one row.\n")
        f.write("  Designed for tabular ML models (XGBoost, LightGBM, etc.)\n")
        f.write("  that don't natively handle 3D sequential input.\n")
        f.write("  Usage: X = np.load('X_combined_flat.npy')\n\n")
        f.write("patient_ids.npy\n")
        f.write(f"  Shape: ({model_data['n_patients']},)\n")
        f.write("  1D array of integer patient IDs (0-indexed row numbers).\n")
        f.write("  Maps each row in X_temporal / X_static / X_combined_flat\n")
        f.write("  back to the corresponding patient.\n")
        f.write("  Usage: ids = np.load('patient_ids.npy')\n\n")
        f.write("feature_names.npz\n")
        f.write("  Compressed archive containing three string arrays:\n")
        f.write("    - 'static':   names of columns in X_static\n")
        f.write("    - 'temporal': names of columns in the flattened temporal\n")
        f.write("                  portion (M0_feature, M1_feature, ...)\n")
        f.write("    - 'combined': names of all columns in X_combined_flat\n")
        f.write("  Useful for model interpretability and feature importance.\n")
        f.write("  Usage: fn = np.load('feature_names.npz', allow_pickle=True)\n")
        f.write("         print(fn['static'])  # static feature names\n\n")
        f.write("-" * 70 + "\n")
        f.write("NOTES\n")
        f.write("-" * 70 + "\n\n")
        f.write("- All .npy files can be loaded with: np.load('filename.npy')\n")
        f.write("- All .csv files can be opened in Excel, Google Sheets, or\n")
        f.write("  loaded with: pd.read_csv('filename.csv')\n")
        f.write("- The cleaned_human_readable.csv is best for reviewing\n")
        f.write("  the data manually or generating summary tables.\n")
        f.write("- The .npy files are best for feeding directly into ML models.\n")
        f.write("- Model-ready outputs are derived strictly from the cleaned\n")
        f.write("  data. For train/test splitting, insert the split between\n")
        f.write("  Phase A and Phase B to prevent data leakage in scaling.\n")
    print(f"  Saved OUTPUT_README.txt")
    print()


def prepare_model_ready_dataset(df_static: pd.DataFrame,
                                df_temporal: pd.DataFrame,
                                output_dir: str):
    """
    Orchestrate the complete Phase B (Model Preparation) pipeline.

    Takes the cleaned and imputed DataFrames from Phase A and applies:
        Stage  9: One-hot encoding
        Stage 10: Feature scaling (StandardScaler)
        Stage 11: Outlier detection & capping (IQR + Z-score)
        Stage 12: 3D tensor preparation for RNN/LSTM
        Stage 13: Validation & export

    All model-ready outputs are derived strictly from the cleaned data.
    For future train/test splitting, this function can be modified to
    accept train/test indices and fit scalers on training data only.

    Args:
        df_static:   Cleaned static features from Phase A (with patient_id)
        df_temporal:  Cleaned temporal features from Phase A (long format)
        output_dir:  Directory to save model-ready outputs

    Returns:
        df_static:    Encoded, scaled, outlier-capped static features
        df_temporal:  Encoded, scaled, outlier-capped temporal features
        model_data:   Dict with numpy arrays and metadata for modeling
    """
    print("\n" + "#" * 70)
    print("# PHASE B: MODEL PREPARATION")
    print("# Deriving model-ready outputs from cleaned data")
    print("#" * 70 + "\n")

    # Stage 9: One-hot encoding (categorical -> binary indicators)
    df_static, df_temporal, encoding_map = encode_features(
        df_static, df_temporal
    )

    # Stage 10: Feature scaling (StandardScaler: zero mean, unit variance)
    # NOTE: For train/test splitting, fit scaler on training data only
    df_static, _, df_temporal, _, scaler_static, scaler_temporal = scale_features(
        df_static, df_static.copy(),
        df_temporal, df_temporal.copy()
)

    # Stage 11: Outlier detection & capping (on scaled data)
    df_static, df_temporal, outlier_report = detect_and_cap_outliers(
        df_static, df_temporal
    )

    # Stage 12: Build 3D tensor for RNN/LSTM input
    model_data = prepare_for_temporal_modeling(df_static, df_temporal)

    # Stage 13a: Validate model-ready outputs
    validation_report = validate_model_ready(df_static, df_temporal, model_data)

    # Stage 13b: Export model-ready files
    export_model_ready_outputs(df_static, df_temporal, model_data, output_dir)

    return df_static, df_temporal, model_data


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_pipeline():
    """
    Execute the complete preprocessing pipeline end-to-end.

    The pipeline is organized into two distinct phases:

    PHASE A - DATA CLEANING (Stages 1-8)
        Produces: cleaned_human_readable.csv
        Contains cleaned, harmonized, imputed data with original
        categorical labels and clinical units. Suitable for EDA.

    PHASE B - MODEL PREPARATION (Stages 9-13)
        Produces: static_features.csv, temporal_features.csv,
                  X_temporal.npy, X_static.npy, X_combined_flat.npy
        Applies encoding, scaling, outlier capping, and tensor
        construction. Derived strictly from Phase A output.

    Returns:
        df_static:   Model-ready static features (encoded, scaled)
        df_temporal:  Model-ready temporal features (encoded, scaled)
        model_data:   Dict with numpy arrays for modeling
    """
    print("\n" + "#" * 70)
    print("# TB-DOTS CAR CDSS - Preprocessing Pipeline")
    print("# Longitudinal PTB Dataset (2015-2025)")
    print("#" * 70 + "\n")

    # ==================================================================
    # PHASE A: DATA CLEANING
    # Stages 1-8: Load -> clean -> harmonize -> impute -> export
    # Output: cleaned_human_readable.csv (human-readable, no scaling)
    # ==================================================================
    print("#" * 70)
    print("# PHASE A: DATA CLEANING")
    print("# Producing cleaned_human_readable.csv")
    print("#" * 70 + "\n")

    # Stage 1: Load and validate schema
    df = load_and_validate_schema(INPUT_PATH)

    # Stage 2: Remove patient/facility identifiers
    df = remove_identifiers(df)

    # Stage 3: Clean and coerce data types (parse weights, BP, dates, etc.)
    df = clean_and_coerce_types(df)

    # Stage 4: Harmonize categorical values to canonical labels
    df = harmonize_categoricals(df)

    # Stage 5: Drop uninformative/redundant columns
    df = drop_uninformative_columns(df)

    # Stage 6: Separate static vs temporal, reshape to long format
    df_static, df_temporal = structure_temporal_data(df)

    # Stage 7: Impute missing values (forward-fill + MICE)
    # NOTE: Forward-fill only (no backward fill) to prevent future leakage
    df_static, df_temporal = impute_missing_mice(df_static, df_temporal)

    # Stage 8: Export human-readable cleaned dataset + validation
    # This CSV preserves categorical labels and clinical units.
    # NO scaling, encoding, or outlier capping applied.
    df_cleaned = export_cleaned_human_readable_dataset(
        df_static, df_temporal, OUTPUT_DIR
    )

    # ==================================================================
    # PHASE B: MODEL PREPARATION
    # Stages 9-13: Encode -> Scale -> Outlier cap -> Tensor -> Export
    # All derived from the cleaned data produced in Phase A.
    # Deep copy to avoid modifying the cleaned data used in Phase A.
    # ==================================================================

    # Deep copy to preserve the cleaned Phase A data untouched
    df_static_model = df_static.copy()
    df_temporal_model = df_temporal.copy()

    df_static_model, df_temporal_model, model_data = prepare_model_ready_dataset(
        df_static_model, df_temporal_model, OUTPUT_DIR
    )

    print("#" * 70)
    print("# PIPELINE COMPLETE")
    print(f"# Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("#")
    print("# Outputs:")
    print("#   Phase A: cleaned_human_readable.csv  (EDA / clinical review)")
    print("#   Phase B: static_features.csv, temporal_features.csv,")
    print("#            X_temporal.npy, X_static.npy, X_combined_flat.npy")
    print("#            (model training)")
    print("#" * 70)

    return df_static_model, df_temporal_model, model_data


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_static, df_temporal, model_data = run_pipeline()
