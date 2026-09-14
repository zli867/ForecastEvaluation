"""Inspect the `monitor_type` column in a combined AirNow CSV.

Usage:
    python Check/check_airnow_monitor_type.py /path/to/airnow_2021.csv
"""
import sys
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/zongrunli/Desktop/ForecastEvaluation/data/airnow/airnow_2021.csv"

df = pd.read_csv(CSV)

print(f"file    : {CSV}")
print(f"rows    : {len(df):,}")
print(f"columns : {df.columns.tolist()}\n")

col = "monitor_type"
if col not in df.columns:
    print(f"[WARN] no '{col}' column; showing last column instead")
    col = df.columns[-1]

s = df[col]
print(f"--- {col} ---")
print(f"dtype        : {s.dtype}")
print(f"n_unique     : {s.nunique(dropna=False)}")
print(f"n_missing    : {s.isna().sum():,}\n")

vc = s.value_counts(dropna=False)
out = pd.DataFrame({"count": vc, "pct": (100 * vc / len(df)).round(3)})
print(out.head(30).to_string())

# Is it actually informative?
if s.nunique(dropna=False) <= 1:
    print(f"\n[VERDICT] constant -> carries no signal, drop it")
elif s.dtype == object and not pd.to_numeric(s, errors="coerce").notna().any():
    print(f"\n[VERDICT] all-string (e.g. 'UG/M3') -> this is the Unit column, "
          f"mislabeled as monitor_type in collectAirNow.py")

# Does monitor_type relate to concentration? (only meaningful if it varies)
if s.nunique(dropna=False) > 1 and "conc" in df.columns:
    print(f"\n--- conc by {col} ---")
    print(df.groupby(col)["conc"].agg(["count", "mean", "median", "std", "max"]).round(2).to_string())