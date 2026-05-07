#!/usr/bin/env python3
"""
Convert each sheet in an Excel file to a separate CSV file.
"""

import pandas as pd
import os

def convert_excel_to_csv(excel_file):
    """
    Convert each sheet in an Excel file to a separate CSV file.
    
    Args:
        excel_file: Path to the Excel file
    """
    # Read the Excel file
    xls = pd.ExcelFile(excel_file)
    
    # Get the base name without extension
    base_name = os.path.basename(os.path.splitext(excel_file)[0])
    
    # Define output directory
    output_dir = os.path.join("..", "..", "dataset", "non-temporal", "yearly_raw")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting '{excel_file}'...")
    print(f"Found {len(xls.sheet_names)} sheet(s): {xls.sheet_names}")
    print(f"Saving to: {output_dir}\n")
    
    # Convert each sheet to CSV
    for sheet_name in xls.sheet_names:
        # Read first 3 rows for inspection
        preview = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=3)

        use_second_row_as_header = False

        if preview.shape[0] >= 2:
            first_row = preview.iloc[0]
        
            # Count non-null cells in first row
            non_null_count = first_row.notna().sum()
        
            # Check if it looks like a title row (single value, contains 'case notification')
            if (
                non_null_count == 1 and
                str(first_row.dropna().iloc[0]).strip().lower().find("case notification") != -1
            ):
                use_second_row_as_header = True

        if use_second_row_as_header:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            new_header = df.iloc[1].fillna('').astype(str).str.strip()
            df = df.iloc[2:].copy()
            df.columns = new_header
        else:
            df = pd.read_excel(xls, sheet_name=sheet_name)

        # Clean completely empty rows
        df = df.dropna(how="all")

        # Clean completely empty columns
        df = df.dropna(axis=1, how="all")

        # Sanitize filename
        safe_sheet_name = sheet_name.replace('/', '_').replace('\\', '_')
        csv_file = os.path.join(output_dir, f"{base_name}_{safe_sheet_name}.csv")

        df.to_csv(csv_file, index=False)

        print(f"✓ Converted sheet '{sheet_name}' to '{csv_file}' ({len(df)} rows, {len(df.columns)} columns)")
    
    print(f"\nConversion complete! Created {len(xls.sheet_names)} CSV file(s).")

if __name__ == "__main__":
    excel_file = os.path.join("..", "dataset", "non-temporal", "raw", "2015-2025-study-without-names.xlsx")
    
    if os.path.exists(excel_file):
        convert_excel_to_csv(excel_file)
    else:
        print(f"Error: File '{excel_file}' not found!")
