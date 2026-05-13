"""
==============================================================================
TB-DOTS CAR CDSS - Temporal EDA Runner
==============================================================================
Run this script from inside the temporal/ folder:

    python run_analysis.py

Or from the project root:

    python temporal/run_analysis.py

Outputs are written to:
    paper/apa/tables/   ← LaTeX .tex files
    paper/apa/figures/  ← PDF figures
==============================================================================
"""

from pathlib import Path
import pandas as pd
from analysis_pipeline import TemporalEDAPipeline, _load_raw

# ── Input ────────────────────────────────────────────────────────────────────
# Path to the raw dataset, relative to this file
RAW_CSV = Path(__file__).resolve().parent / "combined_complete_dataset.csv"

# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at:\n  {RAW_CSV}\n"
            "Make sure combined_complete_dataset.csv is in the temporal/ folder."
        )

    print(f"Loading raw dataset from:\n  {RAW_CSV}")
    df_raw = _load_raw(RAW_CSV)
    print(f"Loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns\n")

    pipeline = TemporalEDAPipeline()
    pipeline.run(df_raw)