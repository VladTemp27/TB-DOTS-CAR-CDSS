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
    output_dir = "dataset/non-temporal/yearly_raw"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting '{excel_file}'...")
    print(f"Found {len(xls.sheet_names)} sheet(s): {xls.sheet_names}")
    print(f"Saving to: {output_dir}\n")
    
    # Convert each sheet to CSV
    for sheet_name in xls.sheet_names:
        # Read the sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Create CSV filename (sanitize sheet name for filename)
        safe_sheet_name = sheet_name.replace('/', '_').replace('\\', '_')
        csv_file = os.path.join(output_dir, f"{base_name}_{safe_sheet_name}.csv")
        
        # Save to CSV
        df.to_csv(csv_file, index=False)
        
        print(f"✓ Converted sheet '{sheet_name}' to '{csv_file}' ({len(df)} rows, {len(df.columns)} columns)")
    
    print(f"\nConversion complete! Created {len(xls.sheet_names)} CSV file(s).")

if __name__ == "__main__":
    excel_file = "dataset/non-temporal/raw/2015-2025-study-without-names.xlsx"
    
    if os.path.exists(excel_file):
        convert_excel_to_csv(excel_file)
    else:
        print(f"Error: File '{excel_file}' not found!")
