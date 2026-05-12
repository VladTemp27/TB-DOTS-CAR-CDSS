#!/usr/bin/env python3
"""
Build a province → [city/municipality] mapping from the consolidated dataset CSV.

The output is UI-only: it drives the City/Municipality <select> in Step 1 of
the patient intake form, filtering options to match the chosen province.

Run from web-app/ directory:
    python scripts/extract_province_cities.py
"""

import json
import os
import sys
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))   # web-app/scripts/
WEBAPP_DIR  = os.path.dirname(SCRIPTS_DIR)               # web-app/
REPO_ROOT   = os.path.dirname(WEBAPP_DIR)                # repo root
CLEAN_CSV = os.path.join(REPO_ROOT, "dataset", "non-temporal", "2015-2025-consolidated-clean.csv")
ENCODINGS_JSON = os.path.join(WEBAPP_DIR, "src", "data", "feature_encodings.json")
OUT_JSON = os.path.join(WEBAPP_DIR, "src", "data", "province_cities.json")


def main():
    if not os.path.exists(CLEAN_CSV):
        print(f"ERROR: {CLEAN_CSV} not found. Run consolidate_non_temporal_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEAN_CSV, usecols=["Province", "City/Municipality"])
    df = df.dropna(subset=["Province"])
    df["Province"] = df["Province"].astype(str).str.strip()
    df["City/Municipality"] = df["City/Municipality"].astype(str).str.strip()

    # Load known city encodings for sanity-check (warn if UI would surface an un-encodable city)
    known_cities: set[str] = set()
    if os.path.exists(ENCODINGS_JSON):
        with open(ENCODINGS_JSON) as f:
            enc = json.load(f)
        known_cities = set(enc.get("__categorical", {}).get("City_Municipality", {}).keys())

    result: dict[str, list[str]] = {}
    for province, group in df.groupby("Province"):
        province = str(province)
        if province.lower() in ("nan", "none", ""):
            continue
        cities = sorted({
            c for c in group["City/Municipality"]
            if c and c.lower() not in ("nan", "none", "")
        })
        if cities:
            result[province] = cities

    # Sort provinces alphabetically
    result = dict(sorted(result.items()))

    # Sanity-check: warn if any city is missing from feature_encodings.json
    if known_cities:
        missing = []
        for province, cities in result.items():
            for city in cities:
                if city not in known_cities:
                    missing.append(f"  [{province}] {city}")
        if missing:
            print("WARNING: these cities are not in feature_encodings.json "
                  "(encodeCategorical will fall back to 0 for them):")
            for m in missing:
                print(m)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    total_cities = sum(len(v) for v in result.values())
    print(f"Wrote {len(result)} provinces, {total_cities} city entries -> {OUT_JSON}")


if __name__ == "__main__":
    main()
