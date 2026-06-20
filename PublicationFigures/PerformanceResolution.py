import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from utils.metrics import R2, RMSE, NMSE, spatial_group_metric
import pickle

# # For resolution, should we regrid data firstly and then compare with avg obs? Or should we do site matching on original resolution and
# # then aggregate the pair under different resolution?
# # ─── config ──────────────────────────────────────────────────────────────────
# start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
# MODEL_VAR, OBS_VAR = "PM25_UFSAQM", "PM25"
# obs_file     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
# stats_folder = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW"
# data_folder  = "/Volumes/T9/Evaluations"
# save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/r2_vs_resolution.pkl"
# metrics = {"R2": R2, "RMSE": RMSE, "NMSE": NMSE}
# res_files = {
#     2: "merged_cell_level_nearest_2.csv",
#     3: "merged_cell_level_nearest_3.csv",
#     4: "merged_cell_level_nearest_4.csv",
#     5: "merged_cell_level_nearest_5.csv",
#     6: "merged_cell_level_nearest_6.csv",
#     8: "merged_cell_level_nearest_8.csv",
#     10: "merged_cell_level_nearest_10.csv",
#     20: "merged_cell_level_nearest_20.csv",
#     30: "merged_cell_level_nearest_30.csv",
#     50: "merged_cell_level_nearest_50.csv",
#     80: "merged_cell_level_nearest_80.csv",
# }
# # res_files = {
# #     2: "merged_cell_level_multires_2.csv",
# #     3: "merged_cell_level_multires_3.csv",
# #     4: "merged_cell_level_multires_4.csv",
# #     5: "merged_cell_level_multires_5.csv",
# #     6: "merged_cell_level_multires_6.csv",
# #     8: "merged_cell_level_multires_8.csv",
# #     10: "merged_cell_level_multires_10.csv",
# #     20: "merged_cell_level_multires_20.csv",
# #     30: "merged_cell_level_multires_30.csv",
# #     50: "merged_cell_level_multires_50.csv",
# #     80: "merged_cell_level_multires_80.csv",
# # }
# LEAD_TIMES = [None, 1, 6, 12, 24]            # None = all lead times pooled
# resolutions = {"site": 5.079, "county": 50, "state": 400}   # effective linear resolution (km)
# obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
# obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]
#
# model_list, cur = [], start_time
# while cur <= end_time:
#     f = os.path.join(stats_folder, f"site_level_aqm_{cur.strftime('%Y%m%d')}.csv")
#     if os.path.exists(f):
#         model_list.append(pd.read_csv(f, parse_dates=["Time"], dtype={"siteCode": str}))
#     cur += timedelta(days=1)
# model_df = pd.concat(model_list, ignore_index=True)
#
# merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]], on=["siteCode", "Time"], how="inner")
# merged["siteCode"] = merged["siteCode"].astype(str).str.zfill(9)
# merged["state_fips"]  = merged["siteCode"].str[:2]
# merged["county_fips"] = merged["siteCode"].str[:5]
# performane_res = {}
#
# for metric_name in metrics:
#     metric_fn = metrics[metric_name]
#     cur_performance_res = {
#         "site":   spatial_group_metric(merged, "siteCode", MODEL_VAR, OBS_VAR, LEAD_TIMES, metric_fn),      # native model resolution
#         "county": spatial_group_metric(merged, "county_fips", MODEL_VAR, OBS_VAR, LEAD_TIMES, metric_fn),
#         "state":  spatial_group_metric(merged, "state_fips", MODEL_VAR, OBS_VAR, LEAD_TIMES, metric_fn),
#     }
#     performane_res[metric_name] = cur_performance_res
#
# # ─── 2. multires gridded files -> R2 vs resolution lines ─────────────────────
# for res_key in sorted(res_files):
#     df = pd.read_csv(os.path.join(data_folder, res_files[res_key]), parse_dates=["Time"])
#     for metric_name in metrics:
#         cur_performance_res = {lt: [] for lt in LEAD_TIMES}
#         metric_fn = metrics[metric_name]
#         vals = spatial_group_metric(df, "cell_id", MODEL_VAR, OBS_VAR, LEAD_TIMES, metric_fn)
#         for lt, v in zip(LEAD_TIMES, vals):
#             cur_performance_res[lt].append(v)
#         performane_res[metric_name][float(df["resolution"].iloc[0])] = cur_performance_res
#
# with open(save_path, "wb") as f:
#     pickle.dump(performane_res, f)
# print(f"saved -> {save_path}")


# ─── plot ────────────────────────────────────────────────────────────────────
# ─── load ────────────────────────────────────────────────────────────────────
save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/r2_vs_resolution.pkl"
METRICs = ["R2", "RMSE", "NMSE"]
LEAD_TIMES = [None, 1, 6, 12, 24]
resolutions = {"site": 5.079, "county": 50, "state": 400}  # km
lead_colors = {None: "black", 1: "tab:blue", 6: "tab:orange",
               12: "tab:green", 24: "tab:red"}

with open(save_path, "rb") as f:
    performane_res = pickle.load(f)

for METRIC in METRICs:
    m = performane_res[METRIC]
    grid_res = sorted(k for k in m if isinstance(k, (int, float)))
    print("gridded resolutions (km):", grid_res)

    # merge the site point into the gridded line as the leftmost (native) point
    line_res = [resolutions["site"]] + grid_res
    line_curves = {lt: [m["site"][i]] + [m[res][lt][0] for res in grid_res]
                   for i, lt in enumerate(LEAD_TIMES)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, lt in enumerate(LEAD_TIMES):
        c = lead_colors[lt]
        label = "All leads" if lt is None else f"Lead {lt} h"

        # single line: site (5 km) + all gridded resolutions
        ax.plot(line_res, line_curves[lt], marker="o", markersize=8,
                markeredgecolor="k", markeredgewidth=0.6, linewidth=2.5,
                color=c, label=label)

        # county & state: separate aggregation, keep as scatter
        ax.scatter(resolutions["county"], m["county"][i], color=c, marker="s",
                   s=110, edgecolor="k", linewidth=0.8, zorder=5)
        ax.scatter(resolutions["state"], m["state"][i], color=c, marker="^",
                   s=140, edgecolor="k", linewidth=0.8, zorder=5)

    # marker-shape legend
    ax.scatter([], [], marker="o", s=70, facecolor="grey", edgecolor="k",
               label="gridded incl. site (≈5 km)")
    ax.scatter([], [], marker="s", s=110, facecolor="grey", edgecolor="k",
               label="county (≈50 km)")
    ax.scatter([], [], marker="^", s=140, facecolor="grey", edgecolor="k",
               label="state (≈400 km)")

    ax.set_xscale("log")
    ax.set_xlabel("Resolution (km)", fontsize=16)
    ylabel = {"R2": r"$R^{2}$ Performance", "RMSE": "RMSE", "NMSE": "NMSE"}[METRIC]
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(rf'UFS-AQM PM$_{{2.5}}$ — {METRIC} vs Resolution',
                 fontsize=16, fontweight="bold")

    ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
    ax.tick_params(axis='both', which='minor', length=4, width=2)
    ax.legend(frameon=False, fontsize=11, ncol=2)
    fig.tight_layout()
    plt.show()