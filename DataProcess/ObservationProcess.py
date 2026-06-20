import pandas as pd
import os
import glob

# TODO: How we use the POC? I removed negative; remove state code as 80
data_dir = "/Users/zongrunli/Desktop/SmkFcstEval/data/obs"
output_filename = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"

filenames = glob.glob(os.path.join(data_dir, '*'))
filenames = [f for f in filenames if f.endswith(".csv")]
dfs = []
for filename in filenames:
    df = pd.read_csv(
        filename,
        dtype={"State Code": str, "County Code": str},
        parse_dates=["Datetime GMT"]  # let pandas handle it
    )
    # ── keep only valid US states (FIPS 01–56); drop 80 and territories/non-US ──
    sc = pd.to_numeric(df["State Code"], errors="coerce")
    df = df[sc.between(1, 56)]
    # Keep only the necessary columns
    df = df[[
        "State Code", "County Code", "Site Num",
        "Latitude", "Longitude",
        "Sample Measurement", "Units of Measure", "Datetime GMT"
    ]]

    # Combine Date GMT and Time GMT
    df["Time"] = df["Datetime GMT"]
    df["PM25"] = df["Sample Measurement"]
    df["siteCode"] = (
            df["State Code"].str.zfill(2) +
            df["County Code"].str.zfill(3) +
            df["Site Num"].astype(str).str.zfill(4)
    )
    df = df[["Time", "PM25", "siteCode", "Latitude", "Longitude"]]
    dfs.append(df)

# Concatenate
dfs = pd.concat(dfs, ignore_index=True)
before = len(dfs)
dfs = (dfs.groupby(["siteCode", "Time"], as_index=False)
       .agg(PM25=("PM25", "mean"),
            Latitude=("Latitude", "mean"),
            Longitude=("Longitude", "mean")))
dfs = dfs[dfs["PM25"] >= 0]
print(f"collapsed {before} -> {len(dfs)} rows "
      f"({before - len(dfs)} duplicate rows averaged)")
dfs = dfs.sort_values("Time")
dfs.to_csv(output_filename, index=False)
print(f"saved -> {output_filename}")