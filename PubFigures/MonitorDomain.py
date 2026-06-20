import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pyproj import CRS, Transformer

from utils.geoutil import StatePolygon
from utils.Visualizations import plotPolygons

# ─── config ──────────────────────────────────────────────────────────────────
stn_info_file    = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
model_static_file= "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/aqm_static.npz"
ufsaqm_mean_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/mean_pm25_first24h_2021_2024.npz"  # PM25_AQM, (607, 998)
monitor_csv      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/site_avg_pm25_2021_2024.csv"  # siteCode, avg_PM25

CMAP = "viridis"

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
pm_field = np.load(ufsaqm_mean_file)["PM25_AQM"]          # (607, 998)

# build a shared color scale from the field (robust percentiles)
vmin = float(np.nanpercentile(pm_field, 2))
vmax = float(np.nanpercentile(pm_field, 98))
norm = Normalize(vmin=vmin, vmax=vmax)

# ─── monitors: coords from pkl, PM2.5 from csv ───────────────────────────────
with open(stn_info_file, "rb") as f:
    site_info = pickle.load(f)
print(len(site_info.keys()))
site_avg = pd.read_csv(monitor_csv)
pm_by_site = dict(zip(site_avg["siteCode"].astype(str), site_avg["avg_PM25"]))

lons, lats, pm_obs = [], [], []
for k, info in site_info.items():
    code = str(k)
    if code in pm_by_site and np.isfinite(pm_by_site[code]):
        lons.append(info["coords"][0])
        lats.append(info["coords"][1])
        pm_obs.append(pm_by_site[code])
lons, lats, pm_obs = np.array(lons), np.array(lats), np.array(pm_obs)
mx, my = to_model(lons, lats)
all_lons, all_lats = [], []
for k, info in site_info.items():
    code = str(k)
    all_lons.append(info["coords"][0])
    all_lats.append(info["coords"][1])

# ─── plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

# UFSAQM PM2.5 field as background
mesh = ax.pcolormesh(x, y, pm_field, cmap=CMAP, norm=norm, shading="auto", zorder=1)

# CONUS states
plotPolygons(StatePolygon(STATES), ax, "k", crs=to_model, linewidth=0.4)

# monitor scatter, SAME cmap + norm as the field
order = np.argsort(pm_obs)  # high values on top
sc = ax.scatter(mx[order], my[order], c=pm_obs[order], cmap=CMAP, norm=norm,
                s=20, edgecolors="r", linewidths=0.3, zorder=3)
# single shared colorbar (driven by the field; scatter uses the identical scale)
cb = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
cb.set_label("$PM_{2.5}$ ($\mu g/m^3$)", fontsize=16)
cb.ax.tick_params(labelsize=14, length=6, width=2)

pad_x, pad_y = 0.03 * (x_max - x_min), 0.03 * (y_max - y_min)
ax.set_xlim(x_min - pad_x, x_max + pad_x)
ax.set_ylim(y_min - pad_y, y_max + pad_y)
ax.set_aspect("equal")
ax.axis("off")

fig.suptitle("UFS-AQM and Observed $PM_{2.5}$\n(# Monitors = %d)" % len(pm_obs),
             fontsize=16, fontweight="bold", y=0.98)
plt.show()