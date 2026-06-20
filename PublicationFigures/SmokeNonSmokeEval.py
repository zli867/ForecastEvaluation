import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
from utils.geoutil import StatePolygon, CountyPolygon, FIPS_list
import matplotlib.pyplot as plt
from utils.metrics import MB, ME, RMSE, CRMSE, NMB, NME, MNB, MNE, FB, FE, IOA, pearson_r, R2, NMSE
from matplotlib.ticker import MultipleLocator
from utils.metrics import metrics_by_lead
from matplotlib.ticker import MultipleLocator, FuncFormatter

start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
metrics = {"R2": R2, "RMSE": RMSE, "NMSE": NMSE}
MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"
initial_time = 6
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time
save_path     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_r2_smokenonsmoke_subsets_%02d_v7.pkl" % initial_time

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


# # ── Plot R2 vs lead time ────────────────────────────────────────────────────
# lead_time = 6
# save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_r2_smokenonsmoke_subsets_%02d.pkl" % lead_time
# with open(save_path, "rb") as f:
#     results = pickle.load(f)
#
# for metric in results:
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for name in results[metric]:
#         ax.plot(results[metric][name]["index"], results[metric][name]["values"],
#                 marker="o", markersize=8, markeredgecolor="k", markeredgewidth=0.6,
#                 linewidth=2.5, label=f"{name} (n={results[metric][name]['length']})")
#
#     ax.set_xlabel("Lead Time (hr)", fontsize=16)
#     if metric == "R2":
#         metric = "$R^2$"
#
#     ax.set_ylabel(f"{metric} Performance", fontsize=16)
#     ax.set_title(f'{metric} by Lead Time: Smoke vs Non-Smoke Start ',
#                  fontsize=16, fontweight='bold')
#
#     ax.tick_params(axis='both', labelsize=16)
#     ax.xaxis.set_minor_locator(MultipleLocator(1))
#     ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
#     ax.tick_params(axis='both', which='minor', length=4, width=2)
#     ax.legend(frameon=False, fontsize=12)
#
#     plt.tight_layout()
#     plt.show()

# # ── Plot R2 vs lead time ────────────────────────────────────────────────────
# lead_time = 12
# utc_offset = lead_time          # lead-time hr + offset = UTC hr  (change if init differs)
#
# save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_r2_smokenonsmoke_subsets_%02d.pkl" % lead_time
# with open(save_path, "rb") as f:
#     results = pickle.load(f)
#
# for metric in results:
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for name in results[metric]:
#         ax.plot(results[metric][name]["index"], results[metric][name]["values"],
#                 marker="o", markersize=8, markeredgecolor="k", markeredgewidth=0.6,
#                 linewidth=2.5, label=f"{name} (n={results[metric][name]['length']})")
#
#     ax.set_xlabel("Lead Time (hr)", fontsize=16)
#     if metric == "R2":
#         metric = "$R^2$"
#
#     ax.set_ylabel(f"{metric} Performance", fontsize=16)
#     ax.set_title(f'{metric} by Lead Time: Smoke vs Non-Smoke Start ',
#                  fontsize=16, fontweight='bold')
#
#     ax.tick_params(axis='both', labelsize=16)
#     ax.xaxis.set_minor_locator(MultipleLocator(1))
#     ax.tick_params(axis='both', which='major', labelsize=16, length=8, width=2)
#     ax.tick_params(axis='both', which='minor', length=4, width=2)
#     ax.legend(frameon=False, fontsize=12)
#
#     # ── secondary x-axis: UTC time (lead time + offset, wrapped to 0–23) ──
#     secax = ax.secondary_xaxis('top',
#                                functions=(lambda x: x + utc_offset,
#                                           lambda x: x - utc_offset))
#     secax.set_xlabel("UTC Time (hr)", fontsize=16)
#     secax.xaxis.set_major_formatter(
#         FuncFormatter(lambda val, pos: f"{int(round(val)) % 24:02d}"))
#     secax.xaxis.set_minor_locator(MultipleLocator(1))
#     secax.tick_params(axis='x', which='major', labelsize=16, length=8, width=2)
#     secax.tick_params(axis='x', which='minor', length=4, width=2)
#
#     plt.tight_layout()
#     plt.show()

# # ── Plot R2 vs lead time: 06z and 12z side by side ──────────────────────────
# import pickle
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator, FuncFormatter
#
# lead_times = [6, 12]
# save_tmpl = ("/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/"
#              "site_r2_smokenonsmoke_subsets_%02d.pkl")
#
# results_all = {}
# for lt in lead_times:
#     with open(save_tmpl % lt, "rb") as f:
#         results_all[lt] = pickle.load(f)
#
# metrics = list(results_all[lead_times[0]].keys())
#
# for metric in metrics:
#     metric_label = "$R^2$" if metric == "R2" else metric
#
#     fig, axes = plt.subplots(1, len(lead_times), figsize=(14, 5), sharey=True)
#
#     for i, (ax, lt) in enumerate(zip(axes, lead_times)):
#         utc_offset = lt  # lead-time hr + offset = UTC hr (change if init differs)
#         res = results_all[lt][metric]
#
#         for name in res:
#             if name != "All":
#                 continue
#             ax.plot(res[name]["index"], res[name]["values"],
#                     marker="o", markersize=8, markeredgecolor="k", markeredgewidth=0.6,
#                     linewidth=2.5, label=f"{name} (n={res[name]['length']:,})")
#
#         ax.set_xlabel("Lead Time (hr)", fontsize=16)
#         if i == 0:
#             ax.set_ylabel(f"{metric_label} Performance", fontsize=16)
#         ax.set_title(f"{lt:02d}z Initialization", fontsize=16, fontweight="bold")
#
#         ax.tick_params(axis="both", labelsize=16)
#         ax.xaxis.set_minor_locator(MultipleLocator(1))
#         ax.tick_params(axis="both", which="major", labelsize=16, length=8, width=2)
#         ax.tick_params(axis="both", which="minor", length=4, width=2)
#         ax.legend(frameon=False, fontsize=12)
#
#         # ── secondary x-axis: UTC time (lead time + offset, wrapped to 0–23) ──
#         # NOTE: bind utc_offset via default arg, otherwise both panels share
#         # the last loop value (late-binding closure)
#         secax = ax.secondary_xaxis(
#             "top",
#             functions=(lambda x, off=utc_offset: x + off,
#                        lambda x, off=utc_offset: x - off))
#         secax.set_xlabel("UTC Time (hr)", fontsize=16)
#         secax.xaxis.set_major_formatter(
#             FuncFormatter(lambda val, pos: f"{int(round(val)) % 24:02d}"))
#         secax.xaxis.set_minor_locator(MultipleLocator(1))
#         secax.tick_params(axis="x", which="major", labelsize=16, length=8, width=2)
#         secax.tick_params(axis="x", which="minor", length=4, width=2)
#
#     fig.suptitle(f"{metric_label} by Lead Time",
#                  fontsize=18, fontweight="bold")
#     fig.tight_layout()
#     plt.show()