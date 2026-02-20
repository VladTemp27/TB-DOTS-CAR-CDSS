#!/usr/bin/env python3
"""
Consolidate yearly TB case CSV files (2015–2025)
Apply preprocessing including MICE imputation.
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

    date_cols = [
        "Date of Diagnosis",
        "Date Started Tx",
        "Birthdate"
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Duration: diagnosis to treatment start
    if "Date of Diagnosis" in df.columns and "Date Started Tx" in df.columns:
        df["Days_To_Treatment"] = (
            df["Date Started Tx"] - df["Date of Diagnosis"]
        ).dt.days

    # Compute age from birthdate
    if "Birthdate" in df.columns:
        df["Computed_Age"] = df["Year"] - df["Birthdate"].dt.year

    return df


# -----------------------------
# OUTLIER REMOVAL
# -----------------------------
def remove_outliers(df):
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df = df[(df["Age"] >= 0) & (df["Age"] <= 110)]
    return df


# -----------------------------
# ENCODING (for MICE compatibility)
# -----------------------------
def encode_categoricals(df):
    encoders = {}
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


# -----------------------------
# MICE IMPUTATION
# -----------------------------
def apply_mice(df):
    """
    Apply MICE only to safe predictor columns.
    Avoid imputing outcome variables.
    """

    # Columns safe to impute
    impute_cols = [
        "Age",
        "Computed_Age",
        "Days_To_Treatment",
        "Sex",
        "Registration Group",
        "Source of Patient",
        "Anatomical Site",
        "Region",
        "Year"
    ]

    impute_cols = [c for c in impute_cols if c in df.columns]

    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        sample_posterior=True
    )

    df_impute = df.copy()

    df_impute[impute_cols] = imputer.fit_transform(df_impute[impute_cols])

    return df_impute


# -----------------------------
# SCALING
# -----------------------------
def scale_features(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


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

    print("Removing outliers...")
    df = remove_outliers(df)

    print("Encoding categorical variables...")
    df, encoders = encode_categoricals(df)

    print("Applying MICE imputation...")
    df = apply_mice(df)

    print("Saving cleaned dataset...")
    df.to_csv(OUTPUT_CLEAN, index=False)

    print("Scaling for ML...")
    df_ml = scale_features(df.copy())
    df_ml.to_csv(OUTPUT_ML, index=False)

    print("\n✓ Consolidation + MICE preprocessing complete!")


if __name__ == "__main__":
    main()