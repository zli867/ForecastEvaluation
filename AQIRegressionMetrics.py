import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import numpy as np
from utils.metrics import MB, RMSE, NMSE, FB, FE, pearson_r, COD
from utils.utils import assign_aqi_category, AQI_ORDER

start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
metrics = {"r": pearson_r, "RMSE": RMSE, "NMSE": NMSE, "MB": MB, "FE": FE, "FB": FB, "COD": COD}
MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"
initial_time = 6
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time
csv_path      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/metrics/regression_metrics_byAQI_%02d.csv" % initial_time
csv_lead_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/metrics/regression_metrics_byAQI_byLead_%02d.csv" % initial_time

CAT_ORDER = ["All"] + AQI_ORDER

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

# ── regression metrics within each observed-AQI category ────────────────────
def regression_rows(mod_all, obs_all, base):
    """Long-format rows: every metric computed within each observed AQI category (+ 'All')."""
    finite  = np.isfinite(mod_all) & np.isfinite(obs_all)
    obs_cat = assign_aqi_category(obs_all)          # category from OBSERVED PM2.5
    rows = []
    for cat in CAT_ORDER:
        sel = finite if cat == "All" else (finite & (obs_cat == cat))
        mod, obs = mod_all[sel], obs_all[sel]
        n = int(sel.sum())
        for metric_name, metric_fn in metrics.items():
            try:
                val = metric_fn(mod, obs) if n > 1 else np.nan
            except Exception:                        # e.g. constant array -> Pearson undefined
                val = np.nan
            rows.append({**base, "category": cat, "metric": metric_name, "value": val, "n": n})
    return rows

def finalize(df):
    df["subset"]   = pd.Categorical(df["subset"], ["All", "Smoke", "Non-smoke"], ordered=True)
    df["category"] = pd.Categorical(df["category"], CAT_ORDER, ordered=True)
    df["metric"]   = pd.Categorical(df["metric"], list(metrics.keys()), ordered=True)
    return df

# ============================================================================
# CSV 1 — metrics per AQI category, pooled over ALL lead times
# ============================================================================
rows = []
for subset_name, sdf in subsets_dict.items():
    rows += regression_rows(sdf[MODEL_VAR].values, sdf[OBS_VAR].values, {"subset": subset_name})

results_df = finalize(pd.DataFrame(rows))
results_df = results_df.sort_values(["subset", "category", "metric"]).reset_index(drop=True)

pd.set_option("display.max_rows", None)
print("===== CSV 1 — regression metrics per AQI category (all leads) =====")
print(results_df)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
results_df.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}\n")

# ============================================================================
# CSV 2 — metrics per AQI category, broken out BY LEAD TIME
# ============================================================================
lead_rows = []
for subset_name, sdf in subsets_dict.items():
    for lead, g in sdf.groupby("lead_time"):
        lead_rows += regression_rows(g[MODEL_VAR].values, g[OBS_VAR].values,
                                     {"lead": int(lead), "subset": subset_name})

lead_df = finalize(pd.DataFrame(lead_rows))
lead_df = lead_df.sort_values(["subset", "category", "metric", "lead"]).reset_index(drop=True)

print("===== CSV 2 — regression metrics per AQI category, by lead =====")
print(lead_df)
lead_df.to_csv(csv_lead_path, index=False)
print(f"Saved to {csv_lead_path}")