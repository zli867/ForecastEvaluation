# import os
# from datetime import datetime, timedelta
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ─── config ──────────────────────────────────────────────────────────────────
# start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
# MODEL_VAR, OBS_VAR = "PM25_UFSAQM", "PM25"
# obs_file     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
# stats_folder = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW"
#
# # ─── load obs ────────────────────────────────────────────────────────────────
# obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
# obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]
#
# # ─── load model (site-level daily files) ─────────────────────────────────────
# model_list, cur = [], start_time
# while cur <= end_time:
#     f = os.path.join(stats_folder, f"site_level_aqm_{cur.strftime('%Y%m%d')}.csv")
#     if os.path.exists(f):
#         model_list.append(pd.read_csv(f, parse_dates=["Time"], dtype={"siteCode": str}))
#     cur += timedelta(days=1)
# model_df = pd.concat(model_list, ignore_index=True)
#
# # ─── merge + spatial keys ────────────────────────────────────────────────────
# merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]],
#                         on=["siteCode", "Time"], how="inner")
# merged["siteCode"]   = merged["siteCode"].astype(str).str.zfill(9)
# merged["state_fips"] = merged["siteCode"].str[:2]
#
# # ─── state-mean diagnostic ───────────────────────────────────────────────────
# ts = merged.groupby(["state_fips", "lead_time"])[[MODEL_VAR, OBS_VAR]].mean().reset_index()
# allt = ts  # lt=None equivalent (pools all lead times)
#
# print("n state groups:", merged["state_fips"].nunique())
# print("rows in ts (states x lead_times):", len(allt))
# print("any NaN means:", allt[[MODEL_VAR, OBS_VAR]].isna().any().to_dict())
#
# # correlation / R2 on the state means (matches spatial_group_metric, lt=None)
# mod = allt[MODEL_VAR].to_numpy(float)
# obs = allt[OBS_VAR].to_numpy(float)
# mask = np.isfinite(mod) & np.isfinite(obs)
# ss_res = np.sum((obs[mask] - mod[mask]) ** 2)
# ss_tot = np.sum((obs[mask] - obs[mask].mean()) ** 2)
# r2 = 1 - ss_res / ss_tot
# print(f"state spatial R2 (all leads): {r2:.4f}   (n={mask.sum()})")
#
# # show the most off-1:1 states (largest |model-obs| on the per-state mean)
# per_state = (allt.groupby("state_fips")[[MODEL_VAR, OBS_VAR]].mean()
#                  .assign(diff=lambda d: d[MODEL_VAR] - d[OBS_VAR])
#                  .sort_values("diff"))
# print("\nmost over/under-predicted states (model - obs):")
# print(per_state.head(5))
# print(per_state.tail(5))
#
# # ─── scatter (highlight state_fips == '80' in red) ──────────────────────────
# is80 = allt["state_fips"].to_numpy() == "80"
# plt.figure(figsize=(6, 6))
# plt.scatter(obs[~is80], mod[~is80], s=30, edgecolor="k", linewidth=0.5,
#             color="tab:blue", label="CONUS states")
# plt.scatter(obs[is80], mod[is80], s=60, edgecolor="k", linewidth=0.6,
#             color="red", zorder=5, label="state_fips 80 (non-CONUS)")
# hi = np.nanmax([np.nanmax(obs), np.nanmax(mod)])
# plt.plot([0, hi], [0, hi], "k--", label="1:1")
# plt.xlabel("state-mean OBS")
# plt.ylabel("state-mean MODEL")
# plt.title(f"State-mean PM2.5  (R2={r2:.3f}, n states={merged['state_fips'].nunique()})")
# plt.legend(frameon=False)
# plt.tight_layout()
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer

from utils.geoutil import FIPS_list, StatePolygon

# ─── config ──────────────────────────────────────────────────────────────────
obs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
STATES = [
    "Washington", "Oregon", "California", "Idaho", "Nevada", "Montana",
    "Wyoming", "Utah", "Arizona", "Colorado", "New Mexico", "North Dakota",
    "South Dakota", "Nebraska", "Kansas", "Oklahoma", "Texas", "Minnesota",
    "Iowa", "Missouri", "Arkansas", "Louisiana", "Wisconsin", "Illinois",
    "Michigan", "Indiana", "Ohio", "Kentucky", "Tennessee", "Mississippi",
    "Alabama", "Georgia", "Florida", "South Carolina", "North Carolina",
    "Virginia", "West Virginia", "Pennsylvania", "New York", "Vermont",
    "New Hampshire", "Maine", "Massachusetts", "Rhode Island", "Connecticut",
    "New Jersey", "Delaware", "Maryland", "District of Columbia", "Alaska", "Hawaii"
]

to_lcc = Transformer.from_crs("EPSG:4326", "ESRI:102004", always_xy=True).transform

# ─── obs site locations ──────────────────────────────────────────────────────
obs_df = pd.read_csv(obs_file, dtype={"siteCode": str})
obs_df["siteCode"]   = obs_df["siteCode"].astype(str).str.zfill(9)
obs_df["state_fips"] = obs_df["siteCode"].str[:2]

site_loc = obs_df.groupby(["siteCode", "state_fips"])[["Latitude", "Longitude"]] \
                 .mean().reset_index()
loc80  = site_loc[site_loc["state_fips"] == "80"]
locoth = site_loc[site_loc["state_fips"] != "80"]
print(f"state_fips '80' sites: {len(loc80)}")
print(loc80[["siteCode", "Latitude", "Longitude"]].to_string(index=False))

# ─── draw CONUS states + overlay sites ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

for fid, poly in zip(FIPS_list(STATES), StatePolygon(STATES)):
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        lon, lat = g.exterior.xy
        xs, ys = to_lcc(np.asarray(lon), np.asarray(lat))
        ax.fill(xs, ys, facecolor="whitesmoke", edgecolor="k", linewidth=0.4)

# all non-80 sites (context), then 80 sites in red
xo, yo = to_lcc(locoth["Longitude"].to_numpy(), locoth["Latitude"].to_numpy())
ax.scatter(xo, yo, s=5, color="lightsteelblue", zorder=4, label="other sites")

if len(loc80):
    x80, y80 = to_lcc(loc80["Longitude"].to_numpy(), loc80["Latitude"].to_numpy())
    ax.scatter(x80, y80, s=70, color="red", edgecolor="k", linewidth=0.6,
               zorder=6, label="state_fips 80")

ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Location of state_fips '80' sites", fontsize=16, fontweight="bold")
ax.legend(frameon=False, fontsize=12, loc="lower left")
plt.tight_layout()
plt.show()