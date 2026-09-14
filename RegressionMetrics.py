import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import numpy as np
from utils.metrics import MB, RMSE, NMSE, FB, FE, pearson_r, COD

start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
metrics = {
    "MB":        MB,
    "RMSE":      RMSE,
    "NMSE":      NMSE,
    "FB":        FB,
    "FE":        FE,
    "pearson_r": pearson_r,
    "COD":       COD,
}
MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"
initial_time = 12
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time
csv_path      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/metrics/regression_metrics_%02d.csv" % initial_time
csv_lead_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/metrics/regression_metrics_byLead_%02d.csv" % initial_time

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

# ── Build subsets ────────────────────────────────────────────────────────────
subsets_dict = {
    "All":       merged,
    "Smoke":     merged[merged["isSmokeDay"]],
    "Non-smoke": merged[~merged["isSmokeDay"]],
}

# ============================================================================
# CSV 1 — metrics pooled over ALL lead times
# ============================================================================
rows = []
for subset_name, sdf in subsets_dict.items():
    mod = sdf[MODEL_VAR].values
    obs = sdf[OBS_VAR].values
    mask = np.isfinite(mod) & np.isfinite(obs)
    mod, obs = mod[mask], obs[mask]
    for metric_name, metric_fn in metrics.items():
        rows.append({
            "subset": subset_name,
            "metric": metric_name,
            "value":  metric_fn(mod, obs),
            "n":      int(mask.sum()),   # paired finite obs across all leads
        })

results_df = pd.DataFrame(rows)
results_df["subset"] = pd.Categorical(results_df["subset"],
                                      ["All", "Smoke", "Non-smoke"], ordered=True)
results_df["metric"] = pd.Categorical(results_df["metric"],
                                      list(metrics.keys()), ordered=True)
results_df = results_df.sort_values(["subset", "metric"]).reset_index(drop=True)

pd.set_option("display.max_rows", None)
print("===== CSV 1 — metrics over ALL lead times =====")
print(results_df)
results_df.to_csv(csv_path, index=False)
print(f"Saved overall dataframe to {csv_path}\n")

# ============================================================================
# CSV 2 — metrics broken out BY LEAD TIME
# ============================================================================
lead_rows = []
for subset_name, sdf in subsets_dict.items():
    for lead, g in sdf.groupby("lead_time"):
        mod = g[MODEL_VAR].values
        obs = g[OBS_VAR].values
        mask = np.isfinite(mod) & np.isfinite(obs)
        mod, obs = mod[mask], obs[mask]
        for metric_name, metric_fn in metrics.items():
            lead_rows.append({
                "lead":   int(lead),
                "subset": subset_name,
                "metric": metric_name,
                "value":  metric_fn(mod, obs) if mask.sum() > 1 else np.nan,
                "n":      int(mask.sum()),   # paired finite obs at this lead
            })

lead_df = pd.DataFrame(lead_rows)
lead_df["subset"] = pd.Categorical(lead_df["subset"],
                                   ["All", "Smoke", "Non-smoke"], ordered=True)
lead_df["metric"] = pd.Categorical(lead_df["metric"],
                                   list(metrics.keys()), ordered=True)
lead_df = lead_df.sort_values(["subset", "metric", "lead"]).reset_index(drop=True)

print("===== CSV 2 — metrics BY LEAD TIME =====")
print(lead_df)
lead_df.to_csv(csv_lead_path, index=False)
print(f"Saved by-lead dataframe to {csv_lead_path}")