import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.metrics import (
    pearson_r, RMSE, NMSE, grouped_metric, MB, FE, FB, COD
)

# ─── config ──────────────────────────────────────────────────────────────────
start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
MODEL_VAR, OBS_VAR = "PM25_UFSAQM", "PM25"
initial_time = 6
stn_file     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time
save_path    = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/state_county_performance.pkl"

METRICS = {"r": pearson_r, "RMSE": RMSE, "NMSE": NMSE, "MB": MB, "FE": FE, "FB": FB, "COD": COD}
LEVELS  = ["state", "county"]
SUBSETS = ["All", "Smoke", "Non-smoke"]


def grouped_pair_count(merged, MODEL_VAR, OBS_VAR, level):
    """Number of finite (paired) model/obs records in each state/county."""
    df = merged.copy()
    df["siteCode"]    = df["siteCode"].astype(str).str.zfill(9)
    df["state_fips"]  = df["siteCode"].str[:2]
    df["county_fips"] = df["siteCode"].str[:5]
    key = "state_fips" if level == "state" else "county_fips"
    finite = df[MODEL_VAR].notna() & df[OBS_VAR].notna()
    return df[finite].groupby(key).size().astype(int).to_dict()


# ─── load obs + model, merge ─────────────────────────────────────────────────
obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]

model_list, cur = [], start_time
while cur <= end_time:
    f = os.path.join(stats_folder, f"site_level_aqm_{cur:%Y%m%d}_%02d.csv" % initial_time)
    if os.path.exists(f):
        model_list.append(pd.read_csv(f, parse_dates=["Time"], dtype={"siteCode": str}))
    cur += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)

merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]],
                        on=["siteCode", "Time"], how="inner")

# ─── smoke flag from stn_info ────────────────────────────────────────────────
with open(stn_file, "rb") as f:
    site_dict = pickle.load(f)
smoke_day_sets = {c: {d.date() for d in site_dict[c].get("smoke_dates", [])} for c in site_dict}
merged["isSmokeDay"] = [code in smoke_day_sets and t.date() in smoke_day_sets[code]
                        for code, t in zip(merged["siteCode"], merged["Time"])]

# ─── subsets ─────────────────────────────────────────────────────────────────
subsets_dict = {
    "All":       merged,
    "Smoke":     merged[merged["isSmokeDay"]],
    "Non-smoke": merged[~merged["isSmokeDay"]],
}

# ─── compute metrics per subset per level ────────────────────────────────────
results = {m: {} for m in METRICS}
results["n_pairs"] = {}
for sub_name, sdf in subsets_dict.items():
    for metric_name, fn in METRICS.items():
        results[metric_name][sub_name] = {
            lvl: grouped_metric(sdf, MODEL_VAR, OBS_VAR, lvl, fn) for lvl in LEVELS
        }
    results["n_pairs"][sub_name] = {
        lvl: grouped_pair_count(sdf, MODEL_VAR, OBS_VAR, lvl) for lvl in LEVELS
    }
    print(f"[{sub_name}] rows={len(sdf):,}")

with open(save_path, "wb") as f:
    pickle.dump(results, f)
print(f"Saved to {save_path}")