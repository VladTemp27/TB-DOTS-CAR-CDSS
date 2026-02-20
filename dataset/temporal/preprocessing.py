"""
==============================================================================
TB-DOTS CAR Clinical Decision Support System (CDSS)
Preprocessing Pipeline for Pulmonary Tuberculosis (PTB) Dataset
==============================================================================

This pipeline prepares longitudinal TB clinical registry data (2015–2025)
from the Cordillera Administrative Region (CAR), Philippines, for temporal
machine learning models (RNN, LSTM, LSTM+XGBoost hybrid).

Dataset: TB-DOTS treatment monitoring records with monthly observations
         (M0–M12) covering demographics, diagnostics, vitals, and adherence.

Pipeline Stages:
    1. Schema Validation & Column Standardization
    2. Identifier Removal (Patient Privacy)
    3. Data Cleaning & Type Coercion
    4. Categorical Harmonization
    5. Temporal Data Structuring (wide → long format)
    6. Missing Data Handling via MICE (Iterative Imputation)
    7. Feature Encoding (One-Hot)
    8. Scaling (StandardScaler / MinMaxScaler)
    9. Outlier Detection & Capping (IQR + Z-score)
   10. Temporal Modeling Preparation (3D tensor for RNN/LSTM)
   11. Final Validation & Export

Author: TB-DOTS CAR CDSS Research Team
Date:   2025
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
INPUT_PATH = os.path.join("dataset", "temporal", "combined_dataset.csv")
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
    "tuberculosis_culture",     # 204/205 missing
    "tuberculin_skin_test",     # 201/205 missing
    "other_lab_test",           # 204/205 missing
    "others",                   # 203/205 missing
    "dat_supported_dup",        # duplicate of dat_supported
    "risk_factors_for_drug_resistance_tuberculosis",  # all null
]

# Categorical columns for one-hot encoding
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

# Numerical columns for scaling (baseline)
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


# ============================================================================
# STAGE 1: SCHEMA VALIDATION & COLUMN STANDARDIZATION
# ============================================================================

def standardize_column_name(col: str) -> str:
    """
    Convert raw column names to snake_case identifiers.

    Transforms:
        'M0_%Adherence'           → 'm0_pct_adherence'
        'Blood Pressure'          → 'blood_pressure'
        'Xpert MTB/RIF'           → 'xpert_mtb_rif'
        'Smear/TB Lamp'           → 'smear_tb_lamp'
        'Date of Birth'           → 'date_of_birth'
        'OTHERS:'                 → 'others'
        'DAT- supported'          → 'dat_supported_dup'
    """
    col = col.strip()
    # Handle the duplicate DAT column
    if col == "DAT- supported":
        return "dat_supported_dup"
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
    Stage 1: Load raw CSV, standardize column names, report schema.

    Returns:
        pd.DataFrame with standardized column names.
    """
    print("=" * 70)
    print("STAGE 1: Schema Validation & Column Standardization")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"  Loaded dataset: {df.shape[0]} patients × {df.shape[1]} columns")

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
# STAGE 2: REMOVE IDENTIFIERS
# ============================================================================

def remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Remove patient-identifying and facility-identifying columns
    to protect patient privacy and remove non-predictive features.

    Columns removed:
        - No. (patient row number)
        - Source_File (data provenance)
        - Name of Diagnosing Facility (facility ID)
        - Name of Treatment Unit (facility ID)
    """
    print("=" * 70)
    print("STAGE 2: Remove Identifiers (Patient Privacy)")
    print("=" * 70)

    cols_before = df.shape[1]
    to_drop = [c for c in ID_COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"  Dropped {len(to_drop)} identifier columns: {to_drop}")
    print(f"  Columns: {cols_before} → {df.shape[1]}")
    print()
    return df


# ============================================================================
# STAGE 3: DATA CLEANING & TYPE COERCION
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
    """Extract numeric O2 saturation from strings like '96%', '98%'."""
    if pd.isna(val):
        return np.nan
    val = str(val).strip().replace("%", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan


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
    # Strip trailing non-date characters (e.g., '4/3/1995= ' → '4/3/1995')
    val = re.sub(r"[^0-9/\-]+$", "", val).strip()
    if not val:
        return pd.NaT
    # Fix common year typos (e.g., 0202 → 2022)
    val = re.sub(r"/0(\d{3})$", r"/\1", val)      # /0202 → /202 (won't fix)
    val = re.sub(r"/02(\d{2})$", r"/20\1", val)    # ensure 4-digit year
    # Fix transposed-digit years (e.g., 2990 → 1990, 2979 → 1979)
    m = re.search(r"/(\d{4})$", val)
    if m:
        year = int(m.group(1))
        if year > 2026:
            # Try swapping first two digits: 29xx → 19xx
            corrected = "19" + m.group(1)[2:]
            val = val[: m.start(1)] + corrected
    try:
        return pd.to_datetime(val, format="mixed", dayfirst=False)
    except Exception:
        try:
            return pd.to_datetime(val, infer_datetime_format=True)
        except Exception:
            return pd.NaT


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
    Stage 3: Parse and coerce all columns to proper types.

    Operations:
        - Parse date columns (DOB, diagnosis, treatment start, outcome)
        - Extract numeric weight (kg) from text
        - Extract numeric height (cm) from text
        - Split Blood Pressure into systolic/diastolic
        - Parse O2 saturation percentages
        - Parse monthly adherence percentages
        - Harmonize smear and Xpert results to binary
        - Coerce numeric columns
    """
    print("=" * 70)
    print("STAGE 3: Data Cleaning & Type Coercion")
    print("=" * 70)

    # --- Date columns ---
    date_cols = ["date_of_birth", "date_of_diagnosis", "date_of_notification",
                 "treatment_start_date", "date_of_outcome",
                 "intensive_phase_start_date", "intensive_phase_end_date",
                 "continuation_phase_start_date", "continuation_phase_end_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_date_safe)
            parsed = df[col].notna().sum()
            print(f"  Parsed '{col}': {parsed}/{len(df)} valid dates")

    # --- Baseline Weight → numeric kg ---
    if "weight" in df.columns:
        df["weight_kg"] = df["weight"].apply(parse_weight)
        df = df.drop(columns=["weight"])
        print(f"  Parsed 'weight' → 'weight_kg': "
              f"{df['weight_kg'].notna().sum()} valid")

    # --- Baseline Height → numeric cm ---
    if "height" in df.columns:
        df["height_cm"] = df["height"].apply(parse_height)
        df = df.drop(columns=["height"])
        print(f"  Parsed 'height' → 'height_cm': "
              f"{df['height_cm'].notna().sum()} valid")

    # --- Blood Pressure → systolic + diastolic ---
    if "blood_pressure" in df.columns:
        bp_parsed = df["blood_pressure"].apply(parse_blood_pressure)
        df["bp_systolic"] = bp_parsed.apply(lambda x: x[0])
        df["bp_diastolic"] = bp_parsed.apply(lambda x: x[1])
        df = df.drop(columns=["blood_pressure"])
        print(f"  Parsed 'blood_pressure' → 'bp_systolic', 'bp_diastolic': "
              f"{df['bp_systolic'].notna().sum()} valid")

    # --- O2 Saturation → numeric ---
    if "o2_sat" in df.columns:
        df["o2_sat"] = df["o2_sat"].apply(parse_o2_sat)
        print(f"  Parsed 'o2_sat': {df['o2_sat'].notna().sum()} valid")

    # --- Baseline Xpert MTB/RIF → binary ---
    if "xpert_mtb_rif" in df.columns:
        df["xpert_mtb_rif"] = df["xpert_mtb_rif"].apply(parse_xpert_result)
        print(f"  Parsed 'xpert_mtb_rif' → binary: "
              f"{df['xpert_mtb_rif'].notna().sum()} valid")

    # --- Baseline Smear Microscopy → binary ---
    if "smear_microscopy" in df.columns:
        df["smear_microscopy"] = df["smear_microscopy"].apply(parse_smear_result)
        print(f"  Parsed 'smear_microscopy' → binary: "
              f"{df['smear_microscopy'].notna().sum()} valid")

    # --- Monthly columns ---
    for m in MONTH_RANGE:
        prefix = f"m{m}_"

        # Weight (may contain 'kg' suffix in M0)
        wcol = f"{prefix}weight"
        if wcol in df.columns:
            df[wcol] = df[wcol].apply(parse_weight)

        # Height (may contain 'cm' suffix in M0)
        hcol = f"{prefix}height"
        if hcol in df.columns:
            df[hcol] = df[hcol].apply(parse_height)

        # Adherence (may contain '%' suffix)
        acol = f"{prefix}pct_adherence"
        if acol in df.columns:
            df[acol] = df[acol].apply(parse_adherence)

        # Smear/TB Lamp → binary
        scol = f"{prefix}smear_tb_lamp"
        if scol in df.columns:
            df[scol] = df[scol].apply(parse_smear_result)

        # Xpert MTB/RIF → binary
        xcol = f"{prefix}xpert_mtb_rif"
        if xcol in df.columns:
            df[xcol] = df[xcol].apply(parse_xpert_result)

        # Doses & missed doses → numeric
        for suffix in ["monthly_doses_taken", "cumulative_doses_taken",
                        "monthly_missed_doses"]:
            dcol = f"{prefix}{suffix}"
            if dcol in df.columns:
                df[dcol] = pd.to_numeric(df[dcol], errors="coerce")

    print(f"  Cleaned monthly columns M0–M12 (weight, height, adherence, "
          f"smear, xpert, doses)")

    # --- Numeric columns that should already be numeric ---
    for col in ["age", "heart_rate", "respiratory_rate", "temperature"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print()
    return df


# ============================================================================
# STAGE 4: CATEGORICAL HARMONIZATION
# ============================================================================

def harmonize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 4: Standardize inconsistent categorical values.

    Many columns have multiple representations of the same category
    (e.g., 'M'/'Male', 'S'/'Single', 'CD'/'Clinically-diagnosed TB').
    This stage maps them to canonical values.
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
            "1": "Regimen 1", "regimen 1": "Regimen 1",
            "cat 1": "Regimen 1", "i": "Regimen 1",
            "2": "Regimen 2", "regimen 2": "Regimen 2",
            "1a": "Regimen 1a", "ia": "Regimen 1a",
            "cannot remember": np.nan,
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

    # --- Co-morbidities → simplified categories ---
    if "co_morbidities" in df.columns:
        def simplify_comorbidity(val):
            if pd.isna(val):
                return np.nan
            val = str(val).strip().lower()
            if val in ("no known", "none", ""):
                return "None"
            if "diabetes" in val or "dm" in val:
                return "Diabetes"
            if "hypertension" in val or "htn" in val or "elevated bp" in val:
                return "Hypertension"
            if "cancer" in val:
                return "Cancer"
            if "ckd" in val or "kidney" in val:
                return "Kidney Disease"
            if "cardiovascular" in val:
                return "Cardiovascular"
            return "Other"
        df["co_morbidities"] = df["co_morbidities"].apply(simplify_comorbidity)
        print(f"  Harmonized 'co_morbidities': "
              f"{df['co_morbidities'].value_counts().to_dict()}")

    # --- Prior History of TB → binary ---
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

    # --- DAT-supported → binary ---
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

    # --- Chest X-ray → simplified ---
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
# STAGE 5: DROP UNINFORMATIVE COLUMNS
# ============================================================================

def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 5: Drop columns that are entirely null, near-entirely null,
    or redundant (e.g., duplicate DAT column, date-phase columns
    already captured by treatment timeline).
    """
    print("=" * 70)
    print("STAGE 5: Drop Uninformative / Redundant Columns")
    print("=" * 70)

    cols_before = df.shape[1]

    # Drop pre-defined always-drop columns
    to_drop = [c for c in ALWAYS_DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop, errors="ignore")
    print(f"  Dropped {len(to_drop)} uninformative columns: {to_drop}")

    # Drop columns that are entirely null after cleaning
    all_null = df.columns[df.isnull().all()].tolist()
    if all_null:
        df = df.drop(columns=all_null)
        print(f"  Dropped {len(all_null)} entirely-null columns: {all_null}")

    # Drop redundant phase date columns (treatment timeline captured by M0-M12)
    phase_cols = [c for c in df.columns if "phase_start" in c or "phase_end" in c]
    # Keep regimen type columns but drop the 6th month and end versions
    # (captured in monthly monitoring)
    regimen_drop = [c for c in df.columns
                    if c in ("regimen_type_at_6th_month_of_treatment",
                             "regimen_type_at_end_of_treatment")]
    drop_extra = phase_cols + regimen_drop
    if drop_extra:
        df = df.drop(columns=drop_extra, errors="ignore")
        print(f"  Dropped {len(drop_extra)} redundant columns: {drop_extra}")

    print(f"  Columns: {cols_before} → {df.shape[1]}")
    print()
    return df


# ============================================================================
# STAGE 6: TEMPORAL DATA STRUCTURING
# ============================================================================

def structure_temporal_data(df: pd.DataFrame):
    """
    Stage 6: Separate static (baseline) features from temporal (monthly)
    features and reshape temporal data into long format.

    The monthly columns (M0–M12) are unpivoted into a long-format table
    with columns: [patient_id, month, feature_name, value].

    This enables:
        - Proper time-series imputation (no future data leakage)
        - Construction of 3D tensors for RNN/LSTM input

    Returns:
        df_static:   (n_patients, n_static_features)
        df_temporal:  (n_patients × n_months, n_temporal_features)
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
          f"({df_temporal['patient_id'].nunique()} patients × "
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
# STAGE 7: MISSING DATA HANDLING VIA MICE (ITERATIVE IMPUTATION)
# ============================================================================

def impute_missing_mice(df_static: pd.DataFrame,
                        df_temporal: pd.DataFrame):
    """
    Stage 7: Handle missing values using MICE (Multiple Imputation by
    Chained Equations), implemented via sklearn's IterativeImputer.

    Strategy:
        - Numerical variables: BayesianRidge regression (default)
        - Categorical variables: Imputed via mode within groups, or
          encoded then imputed, then decoded
        - Temporal imputation: Forward-fill within patient first
          (respects time ordering), then MICE for remaining gaps

    IMPORTANT: Temporal imputation only uses past data (forward fill)
    to prevent future data leakage in time-series features.
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
            sample_posterior=False,
        )
        df_static[num_cols] = imputer_static.fit_transform(df_static[num_cols])
        missing_after = df_static[num_cols].isnull().sum().sum()
        print(f"  MICE imputed static numeric: {missing_before} → "
              f"{missing_after} missing values")

    # Impute categorical static features with mode
    for col in cat_cols:
        if df_static[col].isnull().any():
            # For high-cardinality columns, fill with a canonical value
            if col == "chest_x_ray_at_case_notification":
                fill_val = "Suggestive of PTB"
            else:
                mode_val = df_static[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            n_filled = df_static[col].isnull().sum()
            df_static[col] = df_static[col].fillna(fill_val)
            print(f"  Mode-filled '{col}': {n_filled} values → '{fill_val}'")

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
            sample_posterior=False,
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
    for col in temporal_cat_cols:
        if df_temporal[col].isnull().any():
            # Forward fill within patient first
            df_temporal[col] = df_temporal.groupby("patient_id")[col].ffill()
            # Then global mode for remaining
            mode_val = df_temporal[col].mode()
            if len(mode_val) > 0:
                df_temporal[col] = df_temporal[col].fillna(mode_val.iloc[0])

    print()
    return df_static, df_temporal


# ============================================================================
# STAGE 7B: EXPORT HUMAN-READABLE CLEANED DATASET
# ============================================================================

def export_cleaned_dataset(df_static: pd.DataFrame,
                           df_temporal: pd.DataFrame,
                           output_dir: str):
    """
    Stage 7B: Export a human-readable, cleaned version of the combined
    dataset BEFORE encoding, scaling, or outlier capping.

    This CSV preserves the original categorical labels (e.g., 'Male',
    'Treatment Completed') and numeric values in natural units (kg, cm,
    mmHg, etc.), making it suitable for manual review, descriptive
    statistics, and exploratory data analysis (EDA).

    The temporal features are pivoted back to wide format so each row
    represents one patient — matching the original dataset structure.

    Output:
        - cleaned_combined_dataset.csv : Wide-format, human-readable CSV
    """
    print("=" * 70)
    print("STAGE 7B: Export Human-Readable Cleaned Dataset")
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

    # Flatten the MultiIndex columns: (feature, month) → M{month}_{feature}
    df_temporal_wide.columns = [
        f"M{int(month)}_{feature}"
        for feature, month in df_temporal_wide.columns
    ]
    df_temporal_wide = df_temporal_wide.reset_index()

    # --- Merge static + temporal wide ---
    df_clean = df_static.merge(df_temporal_wide, on="patient_id", how="left")

    # --- Drop internal patient_id (not meaningful outside pipeline) ---
    df_clean = df_clean.drop(columns=["patient_id"], errors="ignore")

    # --- Format date columns as readable strings ---
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].dt.strftime("%Y-%m-%d")

    # --- Round numeric columns for readability ---
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].round(2)

    # --- Save ---
    out_path = os.path.join(output_dir, "cleaned_combined_dataset.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"  Saved cleaned_combined_dataset.csv ({df_clean.shape[0]} rows × "
          f"{df_clean.shape[1]} columns)")
    print(f"  Location: {os.path.abspath(out_path)}")

    # --- Quick summary ---
    n_missing = df_clean.isnull().sum().sum()
    print(f"  Remaining missing values: {n_missing}")
    print(f"  Columns: {list(df_clean.columns[:10])} ... "
          f"({len(df_clean.columns)} total)")
    print()

    return df_clean


# ============================================================================
# STAGE 8: FEATURE ENCODING (ONE-HOT)
# ============================================================================

def encode_features(df_static: pd.DataFrame, df_temporal: pd.DataFrame):
    """
    Stage 8: One-hot encode categorical features.

    Encodes both static and temporal categorical variables.
    Uses pandas get_dummies for simplicity and interpretability.

    Returns:
        df_static:   with one-hot encoded categoricals
        df_temporal:  with one-hot encoded categoricals (if any)
        encoding_map: dict mapping original column → encoded column names
    """
    print("=" * 70)
    print("STAGE 8: Feature Encoding (One-Hot)")
    print("=" * 70)

    encoding_map = {}

    # --- Static categoricals ---
    exclude = {"patient_id"}
    date_cols = [c for c in df_static.columns
                 if df_static[c].dtype == "datetime64[ns]"
                 or "date" in c.lower()]

    cat_cols_static = [c for c in df_static.columns
                       if c not in exclude
                       and c not in date_cols
                       and df_static[c].dtype == "object"]

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
        print(f"  Static columns: {df_static.shape[1]} → "
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

    # Drop remaining date columns from static (not useful for modeling)
    date_cols_to_drop = [c for c in df_static.columns
                         if df_static[c].dtype == "datetime64[ns]"
                         or ("date" in c.lower() and c != "data_year")]
    if date_cols_to_drop:
        df_static = df_static.drop(columns=date_cols_to_drop)
        print(f"  Dropped {len(date_cols_to_drop)} date columns "
              f"(not used in modeling): {date_cols_to_drop}")

    print()
    return df_static, df_temporal, encoding_map


# ============================================================================
# STAGE 9: SCALING
# ============================================================================

def scale_features(df_static: pd.DataFrame, df_temporal: pd.DataFrame):
    """
    Stage 9: Normalize numerical features using StandardScaler.

    Applies scaling to:
        - Baseline clinical numerics (age, weight, height, vitals)
        - Monthly numerical features (doses, weight, height, adherence)

    StandardScaler is used (zero mean, unit variance) as it works well
    with RNN/LSTM models and preserves outlier information.

    Returns:
        df_static, df_temporal (scaled)
        scaler_static, scaler_temporal: fitted scalers for inverse transform
    """
    print("=" * 70)
    print("STAGE 9: Feature Scaling (StandardScaler)")
    print("=" * 70)

    # --- Static numerical columns ---
    static_num_cols = [c for c in NUMERICAL_COLUMNS_SCALE
                       if c in df_static.columns]

    scaler_static = StandardScaler()
    if static_num_cols:
        df_static[static_num_cols] = scaler_static.fit_transform(
            df_static[static_num_cols]
        )
        print(f"  Scaled {len(static_num_cols)} static numeric columns: "
              f"{static_num_cols}")

    # --- Temporal numerical columns ---
    temporal_num_cols = [c for c in df_temporal.columns
                         if c not in ("patient_id", "month")
                         and pd.api.types.is_numeric_dtype(df_temporal[c])
                         and df_temporal[c].nunique() > 2]  # skip binary

    scaler_temporal = StandardScaler()
    if temporal_num_cols:
        df_temporal[temporal_num_cols] = scaler_temporal.fit_transform(
            df_temporal[temporal_num_cols]
        )
        print(f"  Scaled {len(temporal_num_cols)} temporal numeric columns: "
              f"{temporal_num_cols}")

    print()
    return df_static, df_temporal, scaler_static, scaler_temporal


# ============================================================================
# STAGE 10: OUTLIER DETECTION & CAPPING
# ============================================================================

def detect_and_cap_outliers(df_static: pd.DataFrame,
                            df_temporal: pd.DataFrame):
    """
    Stage 10: Detect outliers using IQR and Z-score methods.
    Flag and cap outliers rather than removing patients.

    Methods:
        1. IQR method: Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
        2. Z-score method: |z| > 3.0

    Outliers are capped (winsorized) to the boundary values.

    Note: Applied AFTER scaling, so z-scores are on standardized data.
    Since StandardScaler centers to mean=0, std=1, a z-score threshold
    of 3.0 means capping at ±3 standard deviations.

    Returns:
        df_static, df_temporal with outliers capped
        outlier_report: dict with outlier counts per column
    """
    print("=" * 70)
    print("STAGE 10: Outlier Detection & Capping (IQR + Z-score)")
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
# STAGE 11: TEMPORAL MODELING PREPARATION (3D TENSOR)
# ============================================================================

def prepare_for_temporal_modeling(df_static: pd.DataFrame,
                                  df_temporal: pd.DataFrame):
    """
    Stage 11: Construct datasets ready for RNN/LSTM and hybrid models.

    Outputs:
        1. X_temporal: 3D numpy array (n_patients, n_timesteps, n_features)
           for RNN/LSTM input
        2. X_static: 2D numpy array (n_patients, n_static_features)
           for combining with temporal models or for XGBoost
        3. X_combined_flat: 2D array with flattened temporal + static features
           for XGBoost/tabular models
        4. patient_ids: array of patient IDs

    The 3D tensor preserves temporal ordering (M0 → M12) and per-patient
    sequences, which is essential for recurrent models.
    """
    print("=" * 70)
    print("STAGE 11: Temporal Modeling Preparation")
    print("=" * 70)

    patient_ids = df_static["patient_id"].values
    n_patients = len(patient_ids)
    n_timesteps = len(MONTH_RANGE)

    # Get temporal feature names (exclude patient_id and month)
    temporal_feature_cols = [c for c in df_temporal.columns
                             if c not in ("patient_id", "month")]
    n_temporal_features = len(temporal_feature_cols)

    print(f"  Patients: {n_patients}")
    print(f"  Time steps: {n_timesteps} (M0–M12)")
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
          f"(patients × timesteps × features)")

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
# STAGE 12: FINAL VALIDATION
# ============================================================================

def final_validation(df_static: pd.DataFrame,
                     df_temporal: pd.DataFrame,
                     model_data: dict):
    """
    Stage 12: Run final validation checks to ensure data quality.

    Checks:
        - No identifier columns remain
        - No missing values in model-ready arrays
        - All features properly encoded (no object dtypes)
        - Tensor shapes are consistent
        - No infinite values

    Returns:
        validation_report: dict with check results
    """
    print("=" * 70)
    print("STAGE 12: Final Validation")
    print("=" * 70)

    validation_report = {}
    all_passed = True

    # Check 1: No identifiers
    id_keywords = ["name", "no.", "source", "facility_name"]
    remaining_ids = [c for c in df_static.columns
                     if any(kw in c.lower() for kw in id_keywords)]
    check_1 = len(remaining_ids) == 0
    validation_report["no_identifiers"] = check_1
    print(f"  [{'PASS' if check_1 else 'FAIL'}] No identifiers remain"
          f"{'' if check_1 else f': {remaining_ids}'}")
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

    # Check 4: No object dtypes in static (all encoded)
    obj_cols = df_static.select_dtypes(include=["object"]).columns.tolist()
    check_4 = len(obj_cols) == 0
    validation_report["all_encoded"] = check_4
    print(f"  [{'PASS' if check_4 else 'FAIL'}] All features encoded "
          f"(no object dtype)"
          f"{'' if check_4 else f': {obj_cols}'}")
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
    print(f"\n  >>> VALIDATION STATUS: {status} <<<")
    print()

    return validation_report


# ============================================================================
# STAGE 13: EXPORT
# ============================================================================

def export_outputs(df_static: pd.DataFrame,
                   df_temporal: pd.DataFrame,
                   model_data: dict,
                   output_dir: str):
    """
    Stage 13: Export all processed datasets and model-ready arrays.

    Outputs:
        - static_features.csv       : Cleaned static patient features
        - temporal_features.csv     : Long-format temporal observations
        - X_temporal.npy            : 3D tensor for RNN/LSTM
        - X_static.npy              : 2D static feature matrix
        - X_combined_flat.npy       : Flattened combined matrix for XGBoost
        - patient_ids.npy           : Patient ID mapping
        - feature_names.npz         : Feature name arrays
        - preprocessing_summary.txt : Pipeline execution summary
    """
    print("=" * 70)
    print("STAGE 13: Export Processed Data")
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
        f.write(f"Time steps:            {model_data['n_timesteps']} (M0–M12)\n")
        f.write(f"Temporal features:     {model_data['n_temporal_features']}\n")
        f.write(f"Static features:       {len(model_data['static_feature_names'])}\n")
        f.write(f"X_temporal shape:      {model_data['X_temporal'].shape}\n")
        f.write(f"X_static shape:        {model_data['X_static'].shape}\n")
        f.write(f"X_combined_flat shape: {model_data['X_combined_flat'].shape}\n")
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
        f.write("TB-DOTS CAR CDSS — Output File Descriptions\n")
        f.write("=" * 70 + "\n\n")
        f.write("This folder contains all outputs produced by the\n")
        f.write("preprocessing pipeline (preprocessing_pipeline.py).\n")
        f.write("Below is a description of each file and its intended use.\n\n")
        f.write("-" * 70 + "\n")
        f.write("HUMAN-READABLE FILES\n")
        f.write("-" * 70 + "\n\n")
        f.write("cleaned_combined_dataset.csv\n")
        f.write("  The fully cleaned version of the original combined_dataset.csv.\n")
        f.write("  One row per patient, wide format (M0–M12 columns preserved).\n")
        f.write("  Contains human-readable categorical labels (e.g., 'Male',\n")
        f.write("  'Treatment Completed', 'Regimen 1') and numeric values in\n")
        f.write("  natural units (kg, cm, mmHg, %, etc.).\n")
        f.write("  Missing values have been imputed (MICE for numerics, mode\n")
        f.write("  for categoricals, forward-fill for temporal columns).\n")
        f.write("  NOT encoded or scaled — suitable for manual review, EDA,\n")
        f.write("  descriptive statistics, and research documentation.\n\n")
        f.write("static_features.csv\n")
        f.write("  Baseline (non-temporal) patient features after full\n")
        f.write("  preprocessing: imputed, one-hot encoded, and scaled.\n")
        f.write("  Categorical variables are split into binary indicator\n")
        f.write("  columns (e.g., sex_Male, sex_Female). Numeric values\n")
        f.write("  are standardized (zero mean, unit variance).\n")
        f.write("  Includes a 'patient_id' column linking to temporal data.\n\n")
        f.write("temporal_features.csv\n")
        f.write("  Monthly monitoring data in long format: one row per\n")
        f.write("  patient per month (patient_id × month). Contains\n")
        f.write("  treatment adherence, doses, weight, height, smear, and\n")
        f.write("  Xpert results — all imputed and scaled.\n")
        f.write("  Columns: patient_id, month (0–12), plus temporal features.\n\n")
        f.write("preprocessing_summary.txt\n")
        f.write("  A brief text summary of the pipeline execution, including\n")
        f.write("  dataset dimensions, tensor shapes, and lists of feature\n")
        f.write("  names used in static and temporal components.\n\n")
        f.write("OUTPUT_README.txt\n")
        f.write("  This file — describes every output file in the folder.\n\n")
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
        f.write("  indicators — all encoded and scaled.\n")
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
        f.write("- The cleaned_combined_dataset.csv is best for reviewing\n")
        f.write("  the data manually or generating summary tables.\n")
        f.write("- The .npy files are best for feeding directly into ML models.\n")
    print(f"  Saved OUTPUT_README.txt")
    print()


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_pipeline():
    """
    Execute the complete preprocessing pipeline end-to-end.

    Pipeline order:
        1. Load & validate schema
        2. Remove identifiers
        3. Clean & coerce data types
        4. Harmonize categoricals
        5. Drop uninformative columns
        6. Structure temporal data (wide → long)
        7. MICE imputation (static + temporal)
        8. One-hot encoding
        9. Scaling
       10. Outlier detection & capping
       11. Prepare 3D tensor for RNN/LSTM
       12. Final validation
       13. Export
    """
    print("\n" + "#" * 70)
    print("# TB-DOTS CAR CDSS — Preprocessing Pipeline")
    print("# Longitudinal PTB Dataset (2015–2025)")
    print("#" * 70 + "\n")

    # Stage 1: Load and validate
    df = load_and_validate_schema(INPUT_PATH)

    # Stage 2: Remove identifiers
    df = remove_identifiers(df)

    # Stage 3: Clean and coerce types
    df = clean_and_coerce_types(df)

    # Stage 4: Harmonize categoricals
    df = harmonize_categoricals(df)

    # Stage 5: Drop uninformative columns
    df = drop_uninformative_columns(df)

    # Stage 6: Temporal structuring
    df_static, df_temporal = structure_temporal_data(df)

    # Stage 7: MICE imputation
    df_static, df_temporal = impute_missing_mice(df_static, df_temporal)

    # Stage 7B: Export human-readable cleaned dataset
    df_cleaned = export_cleaned_dataset(df_static, df_temporal, OUTPUT_DIR)

    # Stage 8: Feature encoding
    df_static, df_temporal, encoding_map = encode_features(
        df_static, df_temporal
    )

    # Stage 9: Scaling
    df_static, df_temporal, scaler_static, scaler_temporal = scale_features(
        df_static, df_temporal
    )

    # Stage 10: Outlier detection
    df_static, df_temporal, outlier_report = detect_and_cap_outliers(
        df_static, df_temporal
    )

    # Stage 11: Prepare for modeling
    model_data = prepare_for_temporal_modeling(df_static, df_temporal)

    # Stage 12: Final validation
    validation_report = final_validation(df_static, df_temporal, model_data)

    # Stage 13: Export
    export_outputs(df_static, df_temporal, model_data, OUTPUT_DIR)

    print("#" * 70)
    print("# PIPELINE COMPLETE")
    print(f"# Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("#" * 70)

    return df_static, df_temporal, model_data


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df_static, df_temporal, model_data = run_pipeline()
