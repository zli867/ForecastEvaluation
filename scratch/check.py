import os
import pickle
import pandas as pd
from datetime import datetime, timedelta

from utils.geoutil import FIPS_list

# ─── config (match your grouping script) ─────────────────────────────────────
start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
MODEL_VAR, OBS_VAR = "PM25_UFSAQM", "PM25"
initial_time = 12
stn_info_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time

STATES = [
    "Washington", "Oregon", "California", "Idaho", "Nevada", "Montana",
    "Wyoming", "Utah", "Arizona", "Colorado", "New Mexico", "North Dakota",
    "South Dakota", "Nebraska", "Kansas", "Oklahoma", "Texas", "Minnesota",
    "Iowa", "Missouri", "Arkansas", "Louisiana", "Wisconsin", "Illinois",
    "Michigan", "Indiana", "Ohio", "Kentucky", "Tennessee", "Mississippi",
    "Alabama", "Georgia", "Florida", "South Carolina", "North Carolina",
    "Virginia", "West Virginia", "Pennsylvania", "New York", "Vermont",
    "New Hampshire", "Maine", "Massachusetts", "Rhode Island", "Connecticut",
    "New Jersey", "Delaware", "Maryland", "District of Columbia",
]

def z9(s):
    return s.astype(str).str.zfill(9)

# ─── 1. stn_info sites ───────────────────────────────────────────────────────
with open(stn_info_file, "rb") as f:
    site_info = pickle.load(f)
stn_sites = {str(k).zfill(9) for k in site_info.keys()}
print(f"stn_info sites:            {len(stn_sites):,}")

# ─── 2. obs sites ────────────────────────────────────────────────────────────
obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]
obs_df["siteCode"] = z9(obs_df["siteCode"])
obs_sites = set(obs_df["siteCode"].unique())
print(f"obs sites (in window):     {len(obs_sites):,}")

# ─── 3. model sites ──────────────────────────────────────────────────────────
model_list, cur = [], start_time
while cur <= end_time:
    fp = os.path.join(stats_folder, f"site_level_aqm_{cur:%Y%m%d}_%02d.csv" % initial_time)
    if os.path.exists(fp):
        model_list.append(pd.read_csv(fp, parse_dates=["Time"], dtype={"siteCode": str}))
    cur += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)
model_df["siteCode"] = z9(model_df["siteCode"])
model_sites = set(model_df["siteCode"].unique())
print(f"model sites:               {len(model_sites):,}")

# ─── 4. merged (inner join) sites ────────────────────────────────────────────
merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]],
                        on=["siteCode", "Time"], how="inner")
merged_sites = set(merged["siteCode"].unique())
print(f"merged (paired) sites:     {len(merged_sites):,}")

# ─── 5. where do sites drop out? ─────────────────────────────────────────────
in_obs_not_model = sorted(obs_sites - model_sites)
in_model_not_obs = sorted(model_sites - obs_sites)
obs_lost_in_merge = sorted(obs_sites - merged_sites)   # obs sites with no paired record

print("\n--- drop-out diagnosis ---")
print(f"obs sites missing from model files: {len(in_obs_not_model)}")
print(in_obs_not_model)
print(f"\nobs sites lost in the merge (no paired hour): {len(obs_lost_in_merge)}")
print(obs_lost_in_merge)

# ─── 6. sites outside the 49 plotted regions ─────────────────────────────────
plotted_fips = set(FIPS_list(STATES))
outside = sorted(s for s in merged_sites if s[:2] not in plotted_fips)
print(f"\nmerged sites with FIPS outside plotted states: {len(outside)}")
print(outside)
print("their state FIPS:", sorted({s[:2] for s in outside}))

# ─── 7. final count that would appear on the state map ───────────────────────
plotted_sites = {s for s in merged_sites if s[:2] in plotted_fips}
print(f"\nsites plotted on state map: {len(plotted_sites):,}")