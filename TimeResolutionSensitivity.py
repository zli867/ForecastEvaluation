import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
from utils.metrics import MB, RMSE, NMSE, FB, FE, pearson_r, COD
from utils.metrics import overall_metrics_at_scale

start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"
MODEL_NAME = "UFSAQM"
init_time = 6
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % init_time
save_path     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/temporal_avg_subsets.pkl"
metrics = {"r": pearson_r, "RMSE": RMSE, "NMSE": NMSE, "MB": MB, "FE": FE, "FB": FB, "COD": COD}
scales  = ["Hourly", "Daily", "Weekly", "Monthly", "Seasonal"]
SUBSETS = ["All", "Smoke", "Non-smoke"]

with open(stn_locs_file, 'rb') as file:
    site_dict = pickle.load(file)

obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]

# ── Load model files ────────────────────────────────────────────────────────
model_list = []
cur_time = start_time
while cur_time <= end_time:
    date_str = cur_time.strftime("%Y%m%d")
    cur_model_file = os.path.join(stats_folder, f"site_level_aqm_{date_str}_{init_time:02d}.csv")
    if os.path.exists(cur_model_file):
        model_list.append(pd.read_csv(cur_model_file, parse_dates=["Time"], dtype={"siteCode": str}))
    cur_time += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)
merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]], on=["siteCode", "Time"], how="inner")

# ── smoke flag, then split into subsets ──────────────────────────────────────
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

# ── metrics at each temporal scale, per subset (all lead times pooled) ───────
results = {}
for metric_name, metric_fn in metrics.items():
    results[metric_name] = {}
    for subset_name in SUBSETS:
        sub = subsets_dict[subset_name]
        results[metric_name][subset_name] = [
            overall_metrics_at_scale(sub, s, metric_fn, MODEL_VAR, OBS_VAR) for s in scales
        ]

with open(save_path, "wb") as f:
    pickle.dump({"scales": scales, "results": results, "subsets": SUBSETS}, f)
print(f"saved -> {save_path}")