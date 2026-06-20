import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from utils.metrics import R2, RMSE, NMSE, overall_metrics_at_scale
from matplotlib.ticker import MultipleLocator


# start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
# MODEL_VAR = "PM25_UFSAQM"
# OBS_VAR   = "PM25"
# MODEL_NAME = "UFSAQM"
# LEAD_TIMES = [None, 1, 6, 12, 24]
# stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
# obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
# stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW"
# save_path     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_r2_temporal_avg.pkl"
# metrics = {"R2": R2, "RMSE": RMSE, "NMSE": NMSE}
# scales = ["Hourly", "Daily", "Weekly", "Monthly", "Seasonal"]
#
# with open(stn_locs_file, 'rb') as file:
#     site_dict = pickle.load(file)
#
# obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
# obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]
#
# # ── Load model files ────────────────────────────────────────────────────────
# model_list = []
# cur_time = start_time
# while cur_time <= end_time:
#     date_str = cur_time.strftime("%Y%m%d")
#     cur_model_file = os.path.join(stats_folder, f"site_level_aqm_{date_str}.csv")
#     if os.path.exists(cur_model_file):
#         model_list.append(pd.read_csv(cur_model_file, parse_dates=["Time"], dtype={"siteCode": str}))
#     cur_time += timedelta(days=1)
# model_df = pd.concat(model_list, ignore_index=True)
# merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]], on=["siteCode", "Time"], how="inner")
#
# results = {}
# for metric_name in metrics:
#     results[metric_name] = {}
#     for lt in LEAD_TIMES:
#         if lt is not None:
#             sub = merged[merged["lead_time"] == lt]
#             results[metric_name][lt] = [overall_metrics_at_scale(sub, s, metrics[metric_name], MODEL_VAR, OBS_VAR) for s in scales]
#         else:
#             sub = merged.copy()
#             results[metric_name]["all"] = [overall_metrics_at_scale(sub, s, metrics[metric_name], MODEL_VAR, OBS_VAR) for s in scales]
#
# with open(save_path, "wb") as f:
#     pickle.dump({"scales": scales, "results": results, "lead_times": LEAD_TIMES}, f)
# print(f"saved -> {save_path}")

save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_r2_temporal_avg.pkl"
with open(save_path, "rb") as f:
    d = pickle.load(f)
scales, all_results = d["scales"], d["results"]


for metric in all_results:
    fig, ax = plt.subplots(figsize=(9, 6))
    results = all_results[metric]
    for k, vals in results.items():
        label = "All leads" if k == "all" else f"Lead {k} h"
        kw = dict(color="k", linewidth=3) if k == "all" else dict(linewidth=2.5)
        ax.plot(scales, vals, marker="o", markersize=8,
                markeredgecolor="k", markeredgewidth=0.6, label=label, **kw)

    ax.set_ylabel(f"{metric} Performance", fontsize=16)
    ax.set_title(rf'UFS-AQM {metric} Performance (Temporal Averaging)',
                 fontsize=16, fontweight="bold")

    ax.tick_params(axis='both', labelsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=2)
    ax.legend(frameon=False, fontsize=12)

    fig.tight_layout()
    plt.show()