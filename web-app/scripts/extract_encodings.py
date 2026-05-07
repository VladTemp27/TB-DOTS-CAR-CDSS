#!/usr/bin/env python3
"""
Extract LabelEncoder mappings from the non-temporal dataset CSVs.

Reads both the human-readable clean CSV and the ML-ready (encoded) CSV,
then reconstructs the label → integer mappings for each categorical feature.

Run from web-app/ directory:
    python scripts/extract_encodings.py
"""

import json
import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(BASE, "dataset", "non-temporal", "2015-2025-consolidated-clean.csv")
ENCODED_CSV = os.path.join(BASE, "dataset", "non-temporal", "2015-2025-ml-ready.csv")
OUT_JSON = os.path.join(BASE, "web-app", "src", "data", "feature_encodings.json")

CATEGORICAL_FEATURES = [
    "Sex",
    "Anatomical Site",
    "Registration Group",
    "Bacteriologic Status",
    "Microscopy Result",
    "Source of Patient",
    "Type",
    "Province",
    "City/Municipality",
    "Treatment Health Facility",
    "Screening/Diagnosing Health Facility",
]

# Map column name → ONNX feature name (spaces/slashes → underscores)
def onnx_name(col: str) -> str:
    return col.replace("/", "_").replace(" ", "_")

def main():
    if not os.path.exists(CLEAN_CSV):
        print(f"ERROR: {CLEAN_CSV} not found. Run consolidate_non_temporal_data.py first.")
        sys.exit(1)
    if not os.path.exists(ENCODED_CSV):
        print(f"ERROR: {ENCODED_CSV} not found. Run consolidate_non_temporal_data.py first.")
        sys.exit(1)

    clean = pd.read_csv(CLEAN_CSV)
    encoded = pd.read_csv(ENCODED_CSV)

    result = {}
    for col in CATEGORICAL_FEATURES:
        if col not in clean.columns or col not in encoded.columns:
            print(f"  SKIP {col}: column not found in both CSVs")
            continue

        pairs = pd.DataFrame({
            "label": clean[col].astype(str),
            "code": encoded[col],
        }).dropna().drop_duplicates()

        mapping = {}
        for _, row in pairs.iterrows():
            label = str(row["label"]).strip()
            code = int(row["code"])
            if label and label.lower() not in ("nan", "none", ""):
                mapping[label] = code

        # Sort by code for readability
        mapping = dict(sorted(mapping.items(), key=lambda x: x[1]))
        key = onnx_name(col)
        result[key] = mapping
        print(f"  {key}: {len(mapping)} values → {list(mapping.keys())[:5]}{'...' if len(mapping) > 5 else ''}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

if __name__ == "__main__":
    main()
