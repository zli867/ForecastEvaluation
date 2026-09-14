import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS, Transformer

from utils.geoutil import StatePolygon
from utils.Visualizations import plotPolygons

# ─── config ──────────────────────────────────────────────────────────────────
start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
MODEL_VAR, OBS_VAR = "PM25_UFSAQM", "PM25"
initial_time = 6
stn_info_file     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
model_static_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/aqm_static.npz"
ufsaqm_mean_file  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/mean_pm25_first24h_2021_2024.npz"
obs_file          = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time

CMAP = "cividis"

STATES = [
    "Washington", "Oregon", "California", "Idaho", "Nevada", "Montana",
    "Wyoming", "Utah", "Arizona", "Colorado", "New Mexico", "North Dakota",
    "South Dakota", "Nebraska", "Kansas", "Oklahoma", "Texas", "Minnesota",
    "Iowa", "Missouri", "Arkansas", "Louisiana", "Wisconsin", "Illinois",
    "Michigan", "Indiana", "Ohio", "Kentucky", "Tennessee", "Mississippi",
    "Alabama", "Georgia", "Florida", "South Carolina", "North Carolina",
    "Virginia", "West Virginia", "Pennsylvania", "New York", "Vermont",
    "New Hampshire", "Maine", "Massachusetts", "Rhode Island", "Connecticut",
    "New Jersey", "Delaware", "Maryland", "District of Columbia",
]

# ─── model static: x, y, crs ─────────────────────────────────────────────────
model_static = np.load(model_static_file)
x, y = np.asarray(model_static["x"]), np.asarray(model_static["y"])
proj4 = str(np.asarray(model_static["crs_proj4"]).item())
crs_model = CRS.from_proj4(proj4)
to_model = Transformer.from_crs("EPSG:4326", crs_model, always_xy=True).transform
x_min, x_max = float(x.min()), float(x.max())
y_min, y_max = float(y.min()), float(y.max())

# ─── UFSAQM mean PM2.5 field ─────────────────────────────────────────────────
pm_field = np.load(ufsaqm_mean_file)["PM25_AQM"]
vmin = float(np.nanpercentile(pm_field, 2))
vmax = float(np.nanpercentile(pm_field, 98))
norm = Normalize(vmin=vmin, vmax=vmax)

# ─── obs (time-filtered) + model, inner-merge to get paired sites ────────────
obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]
obs_df["siteCode"] = obs_df["siteCode"].str.zfill(9)

model_list, cur = [], start_time
while cur <= end_time:
    fp = os.path.join(stats_folder, f"site_level_aqm_{cur:%Y%m%d}_%02d.csv" % initial_time)
    if os.path.exists(fp):
        model_list.append(pd.read_csv(fp, parse_dates=["Time"], dtype={"siteCode": str}))
    cur += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)
model_df["siteCode"] = model_df["siteCode"].str.zfill(9)

merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]],
                        on=["siteCode", "Time"], how="inner")

# per-site mean observed PM2.5 over the paired data (defines the 1160 sites)
site_mean = merged.groupby("siteCode")[OBS_VAR].mean()
print(f"paired sites: {site_mean.size}")     # should be 1160

# ─── attach coordinates from stn_info ────────────────────────────────────────
with open(stn_info_file, "rb") as f:
    site_info = pickle.load(f)

lons, lats, pm_obs = [], [], []
for code, val in site_mean.items():
    if not np.isfinite(val):
        continue
    info = site_info.get(code) or site_info.get(int(code)) or site_info.get(code.lstrip("0"))
    if info is None:
        print(f"  no coords for {code}")
        continue
    lons.append(info["coords"][0]); lats.append(info["coords"][1]); pm_obs.append(val)
lons, lats, pm_obs = np.array(lons), np.array(lats), np.array(pm_obs)
mx, my = to_model(lons, lats)
print(f"plotted monitors: {len(pm_obs)}")

# ─── plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
mesh = ax.pcolormesh(x, y, pm_field, cmap=CMAP, norm=norm, shading="auto", zorder=1)
plotPolygons(StatePolygon(STATES), ax, "k", crs=to_model, linewidth=0.4)

order = np.argsort(pm_obs)
sc = ax.scatter(mx[order], my[order], c=pm_obs[order], cmap=CMAP, norm=norm,
                s=20, edgecolors="r", linewidths=0.3, zorder=3)

pad_x, pad_y = 0.03 * (x_max - x_min), 0.03 * (y_max - y_min)
ax.set_xlim(x_min - pad_x, x_max + pad_x)
ax.set_ylim(y_min - pad_y, y_max + pad_y)
ax.set_aspect("equal")
ax.axis("off")

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(mesh, cax=cax)
cb.set_label(r"$PM_{2.5}$ ($\mu g/m^3$)", fontsize=16)
cb.ax.tick_params(labelsize=14, length=6, width=2)

fig.suptitle("UFS-AQM and Observed $PM_{2.5}$\n(# Monitors = %d)" % len(pm_obs),
             fontsize=16, fontweight="bold", y=0.98)
fig.tight_layout()
plt.show()