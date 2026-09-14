import pandas as pd
import glob, os

DATA_DIR = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/06"
START, END = "20210101", "20241231"

# Collect daily files within the date range (extract the 8-digit date only)
files = []
for f in glob.glob(os.path.join(DATA_DIR, "site_level_aqm_*.csv")):
    date_str = os.path.basename(f).replace("site_level_aqm_", "").split("_")[0]  # -> "20210101"
    if START <= date_str <= END:
        files.append(f)
files.sort()

# Read and concatenate — keep siteCode as a string so leading zeros survive
df = pd.concat(
    (pd.read_csv(f, usecols=["Time", "PM25_UFSAQM", "lead_time", "siteCode"],
                 dtype={"siteCode": str})
     for f in files),
    ignore_index=True,
)
df["siteCode"] = df["siteCode"].str.zfill(9)        # ensure 9-digit zero-padded

# Average PM2.5 per site
site_avg = (
    df.groupby("siteCode")["PM25_UFSAQM"]
      .mean()
      .reset_index()
      .rename(columns={"PM25_UFSAQM": "avg_PM25"})
)

site_avg.to_csv("site_avg_pm25_2021_2024.csv", index=False)