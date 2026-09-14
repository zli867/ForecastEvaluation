import os
import pandas as pd
from datetime import datetime, timedelta
from utils.metrics import pearson_r, RMSE, NMSE, MB, FE, FB, COD, spatial_group_metric
import pickle

# ─── config ──────────────────────────────────────────────────────────────────
start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
MODEL_VAR, OBS_VAR = "PM25_UFSAQM", "PM25"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
data_folder   = "/Volumes/T9/Evaluations"
save_path     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/metric_vs_resolution_subsets.pkl"
init_time = 6
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % init_time
metrics      = {"r": pearson_r, "RMSE": RMSE, "NMSE": NMSE, "MB": MB, "FE": FE, "FB": FB, "COD": COD}
LEAD_TIMES   = [None]                                   # pool all lead times
SUBSETS      = ["All", "Smoke", "Non-smoke"]
RES          = [2, 3, 4, 5, 6, 8, 10, 20, 30, 50]  # gridded resolutions present in data_folder
SUFFIX       = {"All": "all", "Smoke": "smoke", "Non-smoke": "nonsmoke"}  # file-name tag per subset
resolutions  = {"site": 5.079, "county": 50, "state": 400}                # effective linear res (km)

# ─── 1. native site-level pairs (+ smoke flag, + FIPS) ───────────────────────
with open(stn_locs_file, "rb") as f:
    site_dict = pickle.load(f)

obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]

model_list, cur = [], start_time
while cur <= end_time:
    f = os.path.join(stats_folder, f"site_level_aqm_{cur.strftime('%Y%m%d')}_{init_time:02d}.csv")
    if os.path.exists(f):
        model_list.append(pd.read_csv(f, parse_dates=["Time"], dtype={"siteCode": str}))
    cur += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)

merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]],
                        on=["siteCode", "Time"], how="inner")
merged["siteCode"]    = merged["siteCode"].astype(str).str.zfill(9)
merged["state_fips"]  = merged["siteCode"].str[:2]
merged["county_fips"] = merged["siteCode"].str[:5]

# smoke flag, keyed by the same zero-filled siteCode
smoke_day_sets = {str(c).zfill(9): {d.date() for d in site_dict[c].get("smoke_dates", [])}
                  for c in site_dict}
merged["isSmokeDay"] = [code in smoke_day_sets and t.date() in smoke_day_sets[code]
                        for code, t in zip(merged["siteCode"], merged["Time"])]

site_subsets = {
    "All":       merged,
    "Smoke":     merged[merged["isSmokeDay"]],
    "Non-smoke": merged[~merged["isSmokeDay"]],
}

# ─── 2. compute metrics: per subset, native (site/county/state) + gridded ────
performance = {mn: {s: {} for s in SUBSETS} for mn in metrics}

for mn, fn in metrics.items():
    for s in SUBSETS:
        sdf = site_subsets[s]
        performance[mn][s]["site"]   = spatial_group_metric(sdf, "siteCode",    MODEL_VAR, OBS_VAR, LEAD_TIMES, fn)[0]
        performance[mn][s]["county"] = spatial_group_metric(sdf, "county_fips", MODEL_VAR, OBS_VAR, LEAD_TIMES, fn)[0]
        performance[mn][s]["state"]  = spatial_group_metric(sdf, "state_fips",  MODEL_VAR, OBS_VAR, LEAD_TIMES, fn)[0]

for res_key in RES:
    for s in SUBSETS:
        fpath = os.path.join(data_folder, f"merged_cell_level_nearest_{SUFFIX[s]}_{res_key}.csv")
        if not os.path.exists(fpath):
            print(f"  missing: {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=["Time"])
        res_km = float(df["resolution"].iloc[0])
        for mn, fn in metrics.items():
            performance[mn][s][res_km] = spatial_group_metric(df, "cell_id", MODEL_VAR, OBS_VAR, LEAD_TIMES, fn)[0]

with open(save_path, "wb") as f:
    pickle.dump(performance, f)
print(f"saved -> {save_path}")