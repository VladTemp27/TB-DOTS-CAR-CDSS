import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / 'by-barangay_raw_complete'
OUTPUT_PATH = SCRIPT_DIR / 'combined_complete_dataset.csv'

# Normalize inconsistent facility name variants found across filenames.
# Key = lowercase raw name extracted from the filename stem.
FACILITY_ALIASES = {
    'atoktrail':  'Atok Trail',
    'atok trail': 'Atok Trail',
    'pacdal':     'Pacdal',
    'pinsao':     'Pinsao',
    'mines view': 'Mines View',
    'eng hill':   'Eng Hill',
}


def parse_filename(stem: str) -> tuple[str, int]:
    """Return (facility_name, data_year) extracted from a CSV file stem.

    The year is the first 4-digit number found; everything before it
    (stripped of trailing separators) is normalized to a canonical facility name.
    Raises ValueError if no 4-digit year is present.
    """
    m = re.search(r'(\d{4})', stem)
    if not m:
        raise ValueError(f"No 4-digit year found in filename: {stem!r}")
    year = int(m.group(1))
    raw_facility = stem[:m.start()].strip().rstrip(' -_').strip()
    facility = FACILITY_ALIASES.get(raw_facility.lower(), raw_facility)
    return facility, year


# ── Discover files ────────────────────────────────────────────────────────────

csv_files = sorted(DATA_DIR.glob('*.csv'))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
print(f"Discovered {len(csv_files)} CSV files in: {DATA_DIR}")

# ── Derive MONTHLY_SUBS from row 1 of the files (validate consistency) ────────
# Row 0 = main headers, Row 1 = monthly sub-headers, Row 2+ = patient data.
# Reading the sub-headers dynamically means adding or renaming metrics in the
# source sheets is picked up automatically, and any schema drift is surfaced.

MONTHLY_SUBS = None
for _path in csv_files:
    _raw2 = pd.read_csv(_path, header=None, nrows=2, dtype=str,
                        encoding='utf-8-sig', keep_default_na=False, na_values=[])
    _row0 = _raw2.iloc[0].tolist()
    _row1 = _raw2.iloc[1].tolist()
    _m0 = next((i for i, h in enumerate(_row0) if str(h).strip() == 'Month 0'), None)
    if _m0 is None:
        continue
    _subs = [s.strip() for s in _row1[_m0:_m0 + 8] if str(s).strip()]
    if not _subs:
        continue
    if MONTHLY_SUBS is None:
        MONTHLY_SUBS = _subs
    elif _subs != MONTHLY_SUBS:
        warnings.warn(
            f"Monthly sub-headers differ in {_path.name}:\n"
            f"  expected: {MONTHLY_SUBS}\n"
            f"  found:    {_subs}"
        )

if MONTHLY_SUBS is None:
    raise RuntimeError("Could not detect monthly sub-headers from any file in DATA_DIR.")

# ── Main loop ─────────────────────────────────────────────────────────────────

all_dfs = []
skipped = []

for csv_path in csv_files:
    fname = csv_path.name

    try:
        facility, year = parse_filename(csv_path.stem)
    except ValueError as e:
        warnings.warn(f"SKIP {fname}: {e}")
        skipped.append(fname)
        continue

    try:
        # keep_default_na=False prevents pandas from silently converting
        # "N/A", "NA", "None", etc. to NaN — those are valid clinical values.
        raw = pd.read_csv(csv_path, header=None, dtype=str,
                          encoding='utf-8-sig', keep_default_na=False, na_values=[])
    except Exception as e:
        warnings.warn(f"SKIP {fname}: failed to read CSV — {e}")
        skipped.append(fname)
        continue

    headers = raw.iloc[0].tolist()
    data    = raw.iloc[2:].reset_index(drop=True)
    data    = data[~data.apply(lambda r: r.str.strip().eq('').all(), axis=1)].reset_index(drop=True)

    # Find where monthly data starts
    m0 = next((i for i, h in enumerate(headers) if str(h).strip() == 'Month 0'), None)

    # Core columns
    if m0:
        core = data.iloc[:, :m0].copy()
        core.columns = [str(h).strip() if pd.notna(h) and str(h).strip() != '' else f'col_{i}'
                        for i, h in enumerate(headers[:m0])]
    else:
        core = data.copy()
        core.columns = [str(h).strip() if pd.notna(h) and str(h).strip() != '' else f'col_{i}'
                        for i, h in enumerate(headers)]

    # Monthly columns
    if m0:
        mdata = data.iloc[:, m0:]
        ns = len(MONTHLY_SUBS)
        for mi in range(mdata.shape[1] // ns):
            sl = mdata.iloc[:, mi*ns:(mi+1)*ns]
            sl.columns = [f'M{mi}_{s}' for s in MONTHLY_SUBS]
            for c in sl.columns:
                core[c] = sl[c].values

    core['Source_File'] = fname
    core['Facility']    = facility
    core['Data_Year']   = year
    all_dfs.append(core)
    print(f"  {fname}: {len(core)} records  [{facility}, {year}]")

if skipped:
    print(f"\nWARNING: {len(skipped)} file(s) skipped: {skipped}")

# ── Combine ───────────────────────────────────────────────────────────────────

if not all_dfs:
    raise RuntimeError("No files were successfully processed. Check DATA_DIR and filenames.")

df = pd.concat(all_dfs, ignore_index=True, sort=False)

# NOTE: "N/A", "None", etc. are kept as-is — they are valid recorded values
# (e.g. "N/A" in Smear Microscopy means test not applicable/not done).
# Only truly empty strings are converted to NaN.
for c in df.columns:
    df[c] = df[c].apply(lambda x: np.nan if pd.notna(x) and str(x).strip() == '' else x)

# Convert Age to numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

print(f"\n✅ Combined: {df.shape[0]} patients × {df.shape[1]} columns")

df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved → {OUTPUT_PATH}")
