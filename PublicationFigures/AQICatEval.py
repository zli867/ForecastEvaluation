import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from utils.utils import AQI_CATEGORIES, AQI_ORDER, assign_aqi_category
from utils.metrics import R2, RMSE, NMSE, metrics_by_lead_per_cat
from matplotlib.ticker import MultipleLocator

# start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
# MODEL_VAR  = "PM25_UFSAQM"
# OBS_VAR    = "PM25"
# MODEL_NAME = "UFS-AQM"
# metrics = {"R2": R2, "RMSE": RMSE, "NMSE": NMSE}
#
# stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
# obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
# stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW"
# save_path     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_r2_all_byAQI.pkl"
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
#
# merged = model_df.merge(
#     obs_df[["siteCode", "Time", OBS_VAR]],
#     on=["siteCode", "Time"], how="inner"
# )
#
#
# merged["obs_aqi_cat"]  = assign_aqi_category(merged[OBS_VAR].values)
# merged["pred_aqi_cat"] = assign_aqi_category(merged[MODEL_VAR].values)
#
# results = {}
# for metric_name in metrics:
#     cur_metric = metrics[metric_name]
#     cur_df = metrics_by_lead_per_cat(merged, MODEL_VAR, OBS_VAR, cur_metric)
#     results[metric_name] = cur_df
#
# with open(save_path, "wb") as f:
#     pickle.dump(results, f)


# ── Plot ────────────────────────────────────────────────────────────────────
save_path     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_r2_all_byAQI.pkl"
with open(save_path, "rb") as f:
    results = pickle.load(f)

for metric_name in results:
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in AQI_ORDER:
        cur_df = results[metric_name]
        sub = cur_df[cur_df["Category"] == cat]
        if sub.empty:
            continue
        ax.plot(sub["lead_time"], sub["value"],
                marker="o", markersize=8, markeredgewidth=0.6,
                color=AQI_CATEGORIES[cat]["color"], linewidth=2.5,
                label=AQI_CATEGORIES[cat]["label"])

    if metric_name == "R2":
        metric_name = "$R^2$"
    ax.set_xlabel("Lead Time (hr)", fontsize=16)
    ax.set_ylabel(f"{metric_name} Performance", fontsize=16)
    ax.set_title(f'{metric_name} vs Lead Time by AQI Category', fontsize=16, fontweight='bold')

    ax.tick_params(axis='both', labelsize=16)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=2)
    ax.legend(frameon=False, fontsize=12, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="AQI Category", title_fontsize=13)
    plt.tight_layout()
    plt.show()