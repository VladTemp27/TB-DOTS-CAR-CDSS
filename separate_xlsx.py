#!/usr/bin/env python3
"""
Split an XLSX file into multiple CSV files (one per sheet).

- Output directory: dataset/temporal/by-barangay_raw_complete
- Each CSV is named after its sheet
- Handles invalid filename characters
"""

import pandas as pd
import os
import re

# =============================================================================
# CONFIG
# =============================================================================

XLSX_FILE = r"dataset\RAW DATA.xlsx"
OUTPUT_DIR = r"dataset\temporal\by-barangay_raw_complete"


# =============================================================================
# UTILITIES
# =============================================================================

def sanitize_filename(name: str) -> str:
    """
    Remove characters that are invalid for filenames.
    """
    name = name.strip()
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name


# =============================================================================
# MAIN
# =============================================================================

def split_xlsx_to_csvs(file_path: str, output_dir: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    xls = pd.ExcelFile(file_path)
    
    print(f"Found {len(xls.sheet_names)} sheets. Exporting...\n")
    
    for i, sheet_name in enumerate(xls.sheet_names, 1):
        print(f"[{i}/{len(xls.sheet_names)}] Processing: {sheet_name}")
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            safe_name = sanitize_filename(sheet_name)
            output_path = os.path.join(output_dir, f"{safe_name}.csv")
            
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        except Exception as e:
            print(f"  ✗ Failed on sheet '{sheet_name}': {e}")
    
    print("\n✓ Done. All sheets exported.")


if __name__ == "__main__":
    split_xlsx_to_csvs(XLSX_FILE, OUTPUT_DIR)