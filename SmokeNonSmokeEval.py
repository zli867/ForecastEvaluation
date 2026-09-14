import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
from utils.geoutil import StatePolygon, CountyPolygon, FIPS_list
import matplotlib.pyplot as plt
from utils.metrics import MB, RMSE, NMSE, FB, FE, pearson_r, COD
from matplotlib.ticker import MultipleLocator
from utils.metrics import metrics_by_lead
from matplotlib.ticker import MultipleLocator, FuncFormatter

start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
metrics = {"r": pearson_r, "RMSE": RMSE, "NMSE": NMSE, "MB": MB, "FE": FE, "FB": FB, "COD": COD}
MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"
initial_time = 12
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time
save_path     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/metrics_bylead_init_%d.pkl" % initial_time

with open(stn_locs_file, 'rb') as file:
    site_dict = pickle.load(file)

obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]
obs_df["stateCode"]  = obs_df["siteCode"].str[:2]
obs_df["countyCode"] = obs_df["siteCode"].str[:5]

# ── Load model files ────────────────────────────────────────────────────────
model_list = []
cur_time = start_time
while cur_time <= end_time:
    date_str = cur_time.strftime("%Y%m%d")
    cur_model_file = os.path.join(stats_folder, f"site_level_aqm_{date_str}_%02d.csv" % initial_time)
    if os.path.exists(cur_model_file):
        model_list.append(pd.read_csv(cur_model_file, parse_dates=["Time"], dtype={"siteCode": str}))
    cur_time += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)

merged = model_df.merge(
    obs_df[["siteCode", "Time", OBS_VAR]],
    on=["siteCode", "Time"], how="inner"
)

# ── Smoke flag from latest stn_info (site_dict["smoke_dates"]) ──────────────
smoke_day_sets = {
    c: {d.date() for d in site_dict[c].get("smoke_dates", [])}
    for c in site_dict
}
merged["isSmokeDay"] = [
    code in smoke_day_sets and t.date() in smoke_day_sets[code]
    for code, t in zip(merged["siteCode"], merged["Time"])
]

# ── Build subsets and SAVE to pickle ────────────────────────────────────────
subsets_dict = {
    "All":       merged,
    "Smoke":     merged[merged["isSmokeDay"]],
    "Non-smoke": merged[~merged["isSmokeDay"]],
}

results = {}
for metric_name in metrics:
    results[metric_name] = {}
    for name, sdf in subsets_dict.items():
        s = metrics_by_lead(sdf, MODEL_VAR, OBS_VAR, metrics[metric_name])
        results[metric_name][name] = {
            "index": s.index,
            "values": s.values,
            "length": len(sdf)
        }

with open(save_path, "wb") as f:
    pickle.dump(results, f)
print(f"Saved subsets to {save_path}")