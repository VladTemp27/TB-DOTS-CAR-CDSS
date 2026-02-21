#!/usr/bin/env python3
"""
Consolidate yearly TB case CSV files (2015–2025)
Apply preprocessing including MICE imputation.
Output human-readable clean CSV, then ML-ready scaled CSV.
"""

import pandas as pd
import numpy as np
import os
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = r"dataset\non-temporal\yearly_raw"
OUTPUT_CLEAN = r"dataset\non-temporal\2015-2025-consolidated-clean.csv"
OUTPUT_ML = r"dataset\non-temporal\2015-2025-ml-ready.csv"


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

    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# REMOVE IDENTIFIERS
# -----------------------------
def remove_identifiers(df):
    return df.drop(columns=[
        "No.",
        "TB/TPT Case No.",
        "Date/Time Record was Created"
    ], errors="ignore")


# -----------------------------
# DATE FEATURE ENGINEERING
# -----------------------------
def process_dates(df):
    """
    Extract temporal features from dates while preserving clinical timeline information.
    These are NOT personal identifiers - they're valuable temporal patterns.
    """

    date_cols = [
        "Date of Screening",
        "Date of Diagnosis", 
        "Date of Notification",
        "Date Started Tx",
        "Microscopy Release Date",
        "RDT Release Date",
        "Birthdate"
    ]

    # Convert all date columns to datetime
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # === TEMPORAL FEATURES ===
    # Extract useful temporal features from diagnosis date
    if "Date of Diagnosis" in df.columns:
        df["Diagnosis_Month"] = df["Date of Diagnosis"].dt.month
        df["Diagnosis_Quarter"] = df["Date of Diagnosis"].dt.quarter
        df["Diagnosis_DayOfYear"] = df["Date of Diagnosis"].dt.dayofyear
        # Season (Northern hemisphere - adjust if Philippines uses different seasons)
        df["Diagnosis_Season"] = df["Diagnosis_Month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        })

    # === CLINICAL DELAYS ===
    # Days from diagnosis to treatment start
    if "Date of Diagnosis" in df.columns and "Date Started Tx" in df.columns:
        df["Days_To_Treatment"] = (
            df["Date Started Tx"] - df["Date of Diagnosis"]
        ).dt.days
        
        # Validate: reasonable treatment delays are 0-365 days
        # Negative values = data entry error (treatment before diagnosis)
        # >365 days = likely data entry error or extreme outlier
        print(f"\n  Days_To_Treatment validation:")
        print(f"    Negative values: {(df['Days_To_Treatment'] < 0).sum()}")
        print(f"    >365 days: {(df['Days_To_Treatment'] > 365).sum()}")
        print(f"    >1000 days: {(df['Days_To_Treatment'] > 1000).sum()}")
        
        # Set impossible values to NaN for imputation
        df.loc[(df["Days_To_Treatment"] < 0) | (df["Days_To_Treatment"] > 365), "Days_To_Treatment"] = np.nan

    # Days from screening to diagnosis
    if "Date of Screening" in df.columns and "Date of Diagnosis" in df.columns:
        df["Days_Screening_To_Diagnosis"] = (
            df["Date of Diagnosis"] - df["Date of Screening"]
        ).dt.days
        # Validate (should be 0-90 days typically)
        df.loc[(df["Days_Screening_To_Diagnosis"] < 0) | (df["Days_Screening_To_Diagnosis"] > 90), "Days_Screening_To_Diagnosis"] = np.nan

    # Days from diagnosis to microscopy result
    if "Date of Diagnosis" in df.columns and "Microscopy Release Date" in df.columns:
        df["Days_To_Microscopy_Result"] = (
            df["Microscopy Release Date"] - df["Date of Diagnosis"]
        ).dt.days
        # Validate (should be -7 to 30 days - can come before formal diagnosis)
        df.loc[(df["Days_To_Microscopy_Result"] < -7) | (df["Days_To_Microscopy_Result"] > 30), "Days_To_Microscopy_Result"] = np.nan

    # Days from diagnosis to RDT result
    if "Date of Diagnosis" in df.columns and "RDT Release Date" in df.columns:
        df["Days_To_RDT_Result"] = (
            df["RDT Release Date"] - df["Date of Diagnosis"]
        ).dt.days
        # Validate (should be -7 to 30 days)
        df.loc[(df["Days_To_RDT_Result"] < -7) | (df["Days_To_RDT_Result"] > 30), "Days_To_RDT_Result"] = np.nan

    # === AGE CALCULATION ===
    if "Birthdate" in df.columns:
        df["Computed_Age"] = df["Year"] - df["Birthdate"].dt.year
        
        # Fix impossible computed ages
        df.loc[(df["Computed_Age"] < 0) | (df["Computed_Age"] > 110), "Computed_Age"] = np.nan

    # === DROP RAW DATES (keep only engineered features) ===
    # Raw dates are dropped to avoid data leakage and because absolute dates aren't useful for ML
    # The temporal features (month, season, delays) are what matter
    dates_to_drop = [col for col in date_cols if col in df.columns]
    df = df.drop(columns=dates_to_drop)

    return df


# -----------------------------
# STANDARDIZE MICROSCOPY RESULTS
# -----------------------------
def standardize_microscopy(df):
    """
    Standardize Microscopy Result entries to valid smear microscopy grades.
    
    Standard AFB smear microscopy grading:
    - Negative (0, no AFB seen)
    - Scanty/+ (1-9 AFB per 100 fields)
    - 1+ (10-99 AFB per 100 fields)
    - 2+ (1-10 AFB per field in 50 fields)
    - 3+ (>10 AFB per field in 20 fields)
    - Not Done (test not performed)
    """
    
    if "Microscopy Result" not in df.columns:
        return df
    
    # Store original for reporting
    original_values = df['Microscopy Result'].copy()
    
    # Create a mapping for standardization
    microscopy_map = {
        # Negative results
        '0': 'Negative',
        '(nothing)': 'Negative',
        
        # Positive results - standardize to clinical grading
        '+': 'Scanty',
        '(+)': 'Scanty',
        '+n ()': 'Scanty',  # Likely meant "scanty" or "+"
        
        '1+': '1+',
        '2+': '2+',
        '3+': '3+',
        
        # Not performed
        'Not Done': 'Not Done',
        'not done': 'Not Done',
        'NOT DONE': 'Not Done',
        
        # ODT might be "Other Diagnostic Test" or data entry error
        # Treating as Not Done unless you have domain knowledge otherwise
        'ODT': 'Not Done',
        'odt': 'Not Done',
    }
    
    # Apply mapping
    df['Microscopy Result'] = df['Microscopy Result'].astype(str).str.strip()
    df['Microscopy Result'] = df['Microscopy Result'].replace(microscopy_map)
    
    # Anything not mapped becomes NaN for imputation
    valid_values = ['Negative', 'Scanty', '1+', '2+', '3+', 'Not Done']
    df.loc[~df['Microscopy Result'].isin(valid_values), 'Microscopy Result'] = np.nan
    
    # Report changes
    print("\n  Microscopy Result Standardization Report:")
    unique_original = original_values.value_counts()
    unique_cleaned = df['Microscopy Result'].value_counts()
    
    print(f"  Original unique values: {len(unique_original)}")
    print(f"  Standardized values: {len(unique_cleaned)}")
    print(f"  Set to NaN (unmapped): {df['Microscopy Result'].isna().sum() - original_values.isna().sum()}")
    
    return df


# -----------------------------
# OUTLIER REMOVAL & AGE VALIDATION
# -----------------------------
def remove_outliers(df):
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df[(df["Age"] >= 0) & (df["Age"] <= 110)]
    
    # If both Age and Computed_Age exist, use Age as ground truth when available
    if "Age" in df.columns and "Computed_Age" in df.columns:
        # Where Age is valid, copy it to Computed_Age if Computed_Age is invalid
        valid_age_mask = df["Age"].notna()
        invalid_computed_age_mask = (df["Computed_Age"] < 0) | (df["Computed_Age"] > 110) | df["Computed_Age"].isna()
        
        df.loc[valid_age_mask & invalid_computed_age_mask, "Computed_Age"] = df.loc[valid_age_mask & invalid_computed_age_mask, "Age"]
    
    return df


# -----------------------------
# MICE IMPUTATION (on encoded data)
# -----------------------------
def apply_mice_encoded(df_encoded):
    """
    Apply MICE only to safe predictor columns.
    Avoid imputing outcome variables.
    """

    # Columns safe to impute
    impute_cols = [
        "Age",
        "Computed_Age",
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
        "Region",
        "Year"
    ]

    impute_cols = [c for c in impute_cols if c in df_encoded.columns]

    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        sample_posterior=True
    )

    df_imputed = df_encoded.copy()
    df_imputed[impute_cols] = imputer.fit_transform(df_imputed[impute_cols])
    
    # Post-imputation validation: clip to reasonable ranges
    if "Age" in df_imputed.columns:
        df_imputed["Age"] = df_imputed["Age"].clip(0, 110)
    
    if "Computed_Age" in df_imputed.columns:
        df_imputed["Computed_Age"] = df_imputed["Computed_Age"].clip(0, 110)
    
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

    return df_imputed


# -----------------------------
# ENCODING (for ML pipeline)
# -----------------------------
def encode_categoricals(df):
    """
    Encode categorical columns for ML.
    Returns encoded df and encoders dict.
    """
    encoders = {}
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders


# -----------------------------
# DECODING (back to human-readable)
# -----------------------------
def decode_categoricals(df_encoded, encoders):
    """
    Decode categorical columns back to original values.
    """
    df_decoded = df_encoded.copy()
    
    for col, encoder in encoders.items():
        if col in df_decoded.columns:
            # Round to nearest integer for categorical columns
            df_decoded[col] = df_decoded[col].round().astype(int)
            # Clip to valid range
            df_decoded[col] = df_decoded[col].clip(0, len(encoder.classes_) - 1)
            # Decode
            df_decoded[col] = encoder.inverse_transform(df_decoded[col])
    
    return df_decoded


# -----------------------------
# SCALING
# -----------------------------
def scale_features(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():

    print("Loading data...")
    df = load_and_consolidate()

    print("Removing identifiers...")
    df = remove_identifiers(df)

    print("Processing dates...")
    df = process_dates(df)

    print("Standardizing microscopy results...")
    df = standardize_microscopy(df)

    print("Removing outliers...")
    df = remove_outliers(df)

    # Save original data types for later
    print("Encoding categorical variables for imputation...")
    df_encoded, encoders = encode_categoricals(df)

    print("Applying MICE imputation on encoded data...")
    df_imputed_encoded = apply_mice_encoded(df_encoded)

    print("Decoding back to human-readable format...")
    df_clean = decode_categoricals(df_imputed_encoded, encoders)

    print("Saving human-readable cleaned dataset...")
    df_clean.to_csv(OUTPUT_CLEAN, index=False)
    print(f"✓ Saved to: {OUTPUT_CLEAN}")

    print("\nScaling for ML...")
    df_ml = scale_features(df_imputed_encoded.copy())
    df_ml.to_csv(OUTPUT_ML, index=False)
    print(f"✓ Saved to: {OUTPUT_ML}")

    print("\n✓ Consolidation + MICE preprocessing complete!")
    print(f"\nSummary:")
    print(f"  - Human-readable: {OUTPUT_CLEAN}")
    print(f"  - ML-ready: {OUTPUT_ML}")


if __name__ == "__main__":
    main()