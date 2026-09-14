# ── PART 1: generate temporary obs/model pair data for the scatter plot ─────
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"
initial_time = 12

obs_file     = "/data/obs/PM25_obs_2021_2024.csv"
stats_folder = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time
save_path    = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/scatter_data_%02d.pkl" % initial_time
stn_locs_file = "/data/static/stn_info.pkl"


with open(stn_locs_file, 'rb') as file:
    site_dict = pickle.load(file)
obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]

model_list = []
cur_time = start_time
while cur_time <= end_time:
    date_str = cur_time.strftime("%Y%m%d")
    cur_model_file = os.path.join(stats_folder, f"site_level_aqm_{date_str}_%02d.csv" % initial_time)
    if os.path.exists(cur_model_file):
        model_list.append(pd.read_csv(cur_model_file, parse_dates=["Time"], dtype={"siteCode": str}))
    cur_time += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)

# ── merge & clean ────────────────────────────────────────────────────────────
merged = model_df.merge(
    obs_df[["siteCode", "Time", OBS_VAR]],
    on=["siteCode", "Time"], how="inner"
)
merged = merged.dropna(subset=[MODEL_VAR, OBS_VAR])
merged = merged[(merged[OBS_VAR] >= 0) & (merged[MODEL_VAR] >= 0)]
# ── Smoke flag from latest stn_info (site_dict["smoke_dates"]) ──────────────
smoke_day_sets = {
    c: {d.date() for d in site_dict[c].get("smoke_dates", [])}
    for c in site_dict
}
merged["isSmokeDay"] = [
    code in smoke_day_sets and t.date() in smoke_day_sets[code]
    for code, t in zip(merged["siteCode"], merged["Time"])
]
subsets_dict = {
    "All":       merged,
    "Smoke":     merged[merged["isSmokeDay"]],
    "Non-smoke": merged[~merged["isSmokeDay"]],
}

data = {
    "All": {
        "obs":   subsets_dict["All"][OBS_VAR].to_numpy(float),
        "model": subsets_dict["All"][MODEL_VAR].to_numpy(float)
    },
    "Smoke": {
        "obs":   subsets_dict["Smoke"][OBS_VAR].to_numpy(float),
        "model": subsets_dict["Smoke"][MODEL_VAR].to_numpy(float)
    },
    "Non-smoke": {
        "obs": subsets_dict["Non-smoke"][OBS_VAR].to_numpy(float),
        "model": subsets_dict["Non-smoke"][MODEL_VAR].to_numpy(float)
    }
}

with open(save_path, "wb") as f:
    pickle.dump(data, f)