#!/usr/bin/env python3
"""
Consolidate yearly TB case CSV files (2015–2025).
Apply preprocessing including MICE imputation.
Output human-readable clean CSV and ML-ready (encoded, un-scaled) CSV.
"""

import logging
import pandas as pd
import numpy as np
import os
import glob

from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-8s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = os.path.join("..", "..", "dataset", "non-temporal", "yearly_raw")
OUTPUT_CLEAN = os.path.join("..", "..", "dataset", "non-temporal", "2015-2025-consolidated-clean.csv")
OUTPUT_ML = os.path.join("..", "..", "dataset", "non-temporal", "2015-2025-ml-ready.csv")

AGE_MIN, AGE_MAX = 0, 110

# Philippine 3-season meteorological calendar (module-level so A5 can reference it)
SEASON_MAP = {
    12: "Tag-lamig", 1: "Tag-lamig", 2: "Tag-lamig",   # cool dry
     3: "Tag-init",  4: "Tag-init",  5: "Tag-init",     # hot dry
     6: "Tag-ulan",  7: "Tag-ulan",  8: "Tag-ulan",     # rainy
     9: "Tag-ulan", 10: "Tag-ulan", 11: "Tag-ulan",
}

# Columns to drop: PII identifiers + leaky/zero-variance columns
DROP_COLS = [
    "No.",
    "TB/TPT Case No.",
    "Date/Time Record was Created",
    "Date of Outcome/Status",   # causally downstream of label — direct leakage
    "Validation Status",         # 100% = 'Validated', zero variance
    "Brgy",                      # 403 unique values, 13.9% null — near-identifier
    "Region",                    # 96% = 'CAR' — near-constant in this CAR-specific study
]


# -----------------------------
# LOAD & CONSOLIDATE
# -----------------------------
def load_and_consolidate():
    pattern = os.path.join(INPUT_DIR, "2015-2025-study-without-names_*.csv")
    files = sorted(glob.glob(pattern))

    dfs = []
    for file in files:
        year = os.path.basename(file).split("_")[-1].replace(".csv", "")
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df = df.replace("No Data", np.nan)
        df["Year"] = int(year)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Remove rows where Outcome/Status equals the column name — header rows embedded as data
    if "Outcome/Status" in combined.columns:
        mask = combined["Outcome/Status"].astype(str).str.strip() == "Outcome/Status"
        n_header_rows = mask.sum()
        if n_header_rows:
            logger.warning("Removed %d embedded header rows (Outcome/Status == 'Outcome/Status')", n_header_rows)
            combined = combined[~mask].reset_index(drop=True)

    logger.info("Loaded %d rows from %d files", len(combined), len(files))
    return combined


# -----------------------------
# REMOVE IDENTIFIERS
# -----------------------------
def remove_identifiers(df):
    before = len(df.columns)
    df = df.drop(columns=DROP_COLS, errors="ignore")
    logger.info("remove_identifiers: dropped %d columns → %d remaining",
                before - len(df.columns), len(df.columns))
    return df


# -----------------------------
# DATE FEATURE ENGINEERING
# -----------------------------
def process_dates(df):
    date_cols = [
        "Date of Screening",
        "Date of Diagnosis",
        "Date of Notification",
        "Date Started Tx",
        "Microscopy Release Date",
        "RDT Release Date",
        "Birthdate"
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # === TEMPORAL FEATURES from diagnosis date ===
    if "Date of Diagnosis" in df.columns:
        df["Diagnosis_Month"] = df["Date of Diagnosis"].dt.month
        df["Diagnosis_Quarter"] = df["Date of Diagnosis"].dt.quarter
        df["Diagnosis_DayOfYear"] = df["Date of Diagnosis"].dt.dayofyear
        df["Diagnosis_Season"] = df["Diagnosis_Month"].map(SEASON_MAP)

    # === CLINICAL DELAYS ===
    if "Date of Diagnosis" in df.columns and "Date Started Tx" in df.columns:
        df["Days_To_Treatment"] = (df["Date Started Tx"] - df["Date of Diagnosis"]).dt.days
        n_invalid = ((df["Days_To_Treatment"] < 0) | (df["Days_To_Treatment"] > 365)).sum()
        logger.warning("Days_To_Treatment: %d rows out-of-range [0, 365] → set NaN", n_invalid)
        df.loc[(df["Days_To_Treatment"] < 0) | (df["Days_To_Treatment"] > 365), "Days_To_Treatment"] = np.nan

    if "Date of Screening" in df.columns and "Date of Diagnosis" in df.columns:
        df["Days_Screening_To_Diagnosis"] = (df["Date of Diagnosis"] - df["Date of Screening"]).dt.days
        n_invalid = ((df["Days_Screening_To_Diagnosis"] < 0) | (df["Days_Screening_To_Diagnosis"] > 90)).sum()
        logger.warning("Days_Screening_To_Diagnosis: %d rows out-of-range [0, 90] → set NaN", n_invalid)
        df.loc[(df["Days_Screening_To_Diagnosis"] < 0) | (df["Days_Screening_To_Diagnosis"] > 90), "Days_Screening_To_Diagnosis"] = np.nan

    if "Date of Diagnosis" in df.columns and "Microscopy Release Date" in df.columns:
        df["Days_To_Microscopy_Result"] = (df["Microscopy Release Date"] - df["Date of Diagnosis"]).dt.days
        n_invalid = ((df["Days_To_Microscopy_Result"] < -7) | (df["Days_To_Microscopy_Result"] > 30)).sum()
        logger.warning("Days_To_Microscopy_Result: %d rows out-of-range [-7, 30] → set NaN", n_invalid)
        df.loc[(df["Days_To_Microscopy_Result"] < -7) | (df["Days_To_Microscopy_Result"] > 30), "Days_To_Microscopy_Result"] = np.nan

    if "Date of Diagnosis" in df.columns and "RDT Release Date" in df.columns:
        df["Days_To_RDT_Result"] = (df["RDT Release Date"] - df["Date of Diagnosis"]).dt.days
        n_invalid = ((df["Days_To_RDT_Result"] < -7) | (df["Days_To_RDT_Result"] > 30)).sum()
        logger.warning("Days_To_RDT_Result: %d rows out-of-range [-7, 30] → set NaN", n_invalid)
        df.loc[(df["Days_To_RDT_Result"] < -7) | (df["Days_To_RDT_Result"] > 30), "Days_To_RDT_Result"] = np.nan

    # === AGE: merge Age + Computed_Age → Age_Final ===
    if "Birthdate" in df.columns:
        df["Computed_Age"] = df["Year"] - df["Birthdate"].dt.year
        df.loc[(df["Computed_Age"] < AGE_MIN) | (df["Computed_Age"] > AGE_MAX), "Computed_Age"] = np.nan

    age_col = "Age" if "Age" in df.columns else None
    computed_col = "Computed_Age" if "Computed_Age" in df.columns else None

    if age_col and computed_col:
        df["Age_Final"] = df[age_col].combine_first(df[computed_col])
        df.drop(columns=[age_col, computed_col], inplace=True)
    elif age_col:
        df.rename(columns={age_col: "Age_Final"}, inplace=True)
    elif computed_col:
        df.rename(columns={computed_col: "Age_Final"}, inplace=True)

    if "Age_Final" in df.columns:
        df["Age_Final"] = pd.to_numeric(df["Age_Final"], errors="coerce")
        df.loc[(df["Age_Final"] < AGE_MIN) | (df["Age_Final"] > AGE_MAX), "Age_Final"] = np.nan

    # Drop raw date columns
    dates_to_drop = [col for col in date_cols if col in df.columns]
    df = df.drop(columns=dates_to_drop)

    return df


# -----------------------------
# STANDARDIZE MICROSCOPY RESULTS
# -----------------------------
def standardize_microscopy(df):
    if "Microscopy Result" not in df.columns:
        return df

    microscopy_map = {
        '0': 'Negative',
        '(nothing)': 'Negative',
        '+': 'Scanty',
        '(+)': 'Scanty',
        '+n ()': 'Scanty',
        '1+': '1+',
        '2+': '2+',
        '3+': '3+',
        'Not Done': 'Not Done',
        'not done': 'Not Done',
        'NOT DONE': 'Not Done',
        'ODT': 'Not Done',
        'odt': 'Not Done',
    }

    df['Microscopy Result'] = df['Microscopy Result'].astype(str).str.strip()
    df['Microscopy Result'] = df['Microscopy Result'].replace(microscopy_map)

    valid_values = ['Negative', 'Scanty', '1+', '2+', '3+', 'Not Done']
    n_unmapped = (~df['Microscopy Result'].isin(valid_values)).sum()
    df.loc[~df['Microscopy Result'].isin(valid_values), 'Microscopy Result'] = np.nan

    # Preserve missingness as a signal — do NOT mode-impute; use explicit 'Unknown' category
    df['Microscopy Result'] = df['Microscopy Result'].fillna('Unknown')
    logger.info("standardize_microscopy: %d rows mapped to 'Unknown'", n_unmapped)

    return df
    

# -----------------------------
# BIN RDT RESULT (31 variants → 3 classes)
# -----------------------------
def bin_rdt_result(df):
    if "RDT Result" not in df.columns:
        return df

    detected_patterns = ['detected', 'positive', 'reactive', 'mtb detected']
    not_detected_patterns = ['not detected', 'negative', 'non-reactive', 'no mtb']

    def classify(val):
        if pd.isna(val):
            return 'Not_Done'
        v = str(val).lower().strip()
        if any(p in v for p in detected_patterns):
            return 'Detected'
        if any(p in v for p in not_detected_patterns):
            return 'Not_Detected'
        return 'Not_Done'

    original_nunique = df['RDT Result'].nunique()
    df['RDT Result'] = df['RDT Result'].apply(classify)
    logger.info("bin_rdt_result: %d unique values → 3 classes (Detected/Not_Detected/Not_Done)",
                original_nunique)
    return df


# -----------------------------
# OUTLIER REMOVAL
# -----------------------------
def remove_outliers(df):
    if "Age_Final" in df.columns:
        df["Age_Final"] = pd.to_numeric(df["Age_Final"], errors="coerce")
        n_invalid = ((df["Age_Final"] < AGE_MIN) | (df["Age_Final"] > AGE_MAX)).sum()
        df.loc[(df["Age_Final"] < AGE_MIN) | (df["Age_Final"] > AGE_MAX), "Age_Final"] = np.nan
        if n_invalid:
            logger.warning("remove_outliers: %d Age_Final out-of-range → set NaN (no rows dropped)", n_invalid)
    return df


# -----------------------------
# MICE IMPUTATION (on encoded data)
# -----------------------------
def apply_mice_encoded(df_encoded, encoders):
    impute_cols = [
        "Age_Final",
        "Days_To_Treatment",
        "Days_Screening_To_Diagnosis",
        "Days_To_Microscopy_Result",
        "Days_To_RDT_Result",
        "Diagnosis_Month",
        "Diagnosis_Quarter",
        "Diagnosis_DayOfYear",
        "Sex",
        "Registration Group",
        "Source of Patient",
        "Anatomical Site",
        "Year"
    ]

    impute_cols = [c for c in impute_cols if c in df_encoded.columns]
    logger.info("MICE imputation on %d columns: %s", len(impute_cols), impute_cols)

    imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=True)

    df_imputed = df_encoded.copy()
    df_imputed[impute_cols] = imputer.fit_transform(df_imputed[impute_cols])

    # Post-imputation clipping to valid ranges
    if "Age_Final" in df_imputed.columns:
        df_imputed["Age_Final"] = df_imputed["Age_Final"].clip(AGE_MIN, AGE_MAX)

    if "Days_To_Treatment" in df_imputed.columns:
        df_imputed["Days_To_Treatment"] = df_imputed["Days_To_Treatment"].clip(0, 365)

    if "Days_Screening_To_Diagnosis" in df_imputed.columns:
        df_imputed["Days_Screening_To_Diagnosis"] = df_imputed["Days_Screening_To_Diagnosis"].clip(0, 90)

    if "Days_To_Microscopy_Result" in df_imputed.columns:
        df_imputed["Days_To_Microscopy_Result"] = df_imputed["Days_To_Microscopy_Result"].clip(-7, 30)

    if "Days_To_RDT_Result" in df_imputed.columns:
        df_imputed["Days_To_RDT_Result"] = df_imputed["Days_To_RDT_Result"].clip(-7, 30)

    if "Diagnosis_Month" in df_imputed.columns:
        df_imputed["Diagnosis_Month"] = df_imputed["Diagnosis_Month"].clip(1, 12).round()

    if "Diagnosis_Quarter" in df_imputed.columns:
        df_imputed["Diagnosis_Quarter"] = df_imputed["Diagnosis_Quarter"].clip(1, 4).round()

    if "Diagnosis_DayOfYear" in df_imputed.columns:
        df_imputed["Diagnosis_DayOfYear"] = df_imputed["Diagnosis_DayOfYear"].clip(1, 365).round()

    # Re-derive Diagnosis_Season from MICE-corrected Diagnosis_Month, then re-encode
    if "Diagnosis_Month" in df_imputed.columns and "Diagnosis_Season" in df_imputed.columns:
        month_ints = df_imputed["Diagnosis_Month"].round().astype(int).clip(1, 12)
        season_strings = month_ints.map(SEASON_MAP).astype(str)
        if "Diagnosis_Season" in encoders:
            df_imputed["Diagnosis_Season"] = encoders["Diagnosis_Season"].fit_transform(season_strings)
            logger.info("Diagnosis_Season regenerated from imputed Diagnosis_Month and re-encoded")

    return df_imputed


# -----------------------------
# ENCODING (for ML pipeline)
# -----------------------------
def encode_categoricals(df):
    encoders = {}
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        # Fill NaN with mode before encoding — prevents 'nan' from becoming its own class
        fill_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df_encoded[col] = df[col].fillna(fill_val)
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders


# -----------------------------
# DECODING (back to human-readable)
# -----------------------------
def decode_categoricals(df_encoded, encoders):
    df_decoded = df_encoded.copy()

    for col, encoder in encoders.items():
        if col in df_decoded.columns:
            df_decoded[col] = df_decoded[col].round().astype(int)
            df_decoded[col] = df_decoded[col].clip(0, len(encoder.classes_) - 1)
            df_decoded[col] = encoder.inverse_transform(df_decoded[col])

    return df_decoded


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    logger.info("=== Consolidation pipeline start ===")

    logger.info("Loading data...")
    df = load_and_consolidate()

    logger.info("Removing identifiers and leaky/zero-variance columns...")
    df = remove_identifiers(df)

    logger.info("Processing dates and engineering temporal features...")
    df = process_dates(df)

    logger.info("Standardizing microscopy results...")
    df = standardize_microscopy(df)

    logger.info("Binning RDT Result variants...")
    df = bin_rdt_result(df)

    logger.info("Removing outliers (nullify, not drop)...")
    df = remove_outliers(df)

    logger.info("Encoding categorical variables for imputation...")
    df_encoded, encoders = encode_categoricals(df)

    logger.info("Applying MICE imputation on encoded data...")
    df_imputed_encoded = apply_mice_encoded(df_encoded, encoders)

    logger.info("Decoding back to human-readable format...")
    df_clean = decode_categoricals(df_imputed_encoded, encoders)

    logger.info("Saving human-readable cleaned dataset...")
    df_clean.to_csv(OUTPUT_CLEAN, index=False)
    logger.info("✓ Clean CSV → %s  (%d rows, %d cols)", OUTPUT_CLEAN, len(df_clean), len(df_clean.columns))

    logger.info("Saving ML-ready dataset (encoded, imputed, NOT scaled)...")
    df_imputed_encoded.to_csv(OUTPUT_ML, index=False)
    logger.info("✓ ML-ready CSV → %s  (%d rows, %d cols)", OUTPUT_ML, len(df_imputed_encoded), len(df_imputed_encoded.columns))

    logger.info("=== Consolidation pipeline complete ===")
    logger.info("  Clean CSV : %s", OUTPUT_CLEAN)
    logger.info("  ML-ready  : %s", OUTPUT_ML)


if __name__ == "__main__":
    main()
