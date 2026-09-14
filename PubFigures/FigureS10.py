import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer

from utils.geoutil import CountyPolygon

to_lcc = Transformer.from_crs("EPSG:4326", "ESRI:102004", always_xy=True).transform

save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/state_county_performance.pkl"
with open(save_path, "rb") as f:
    results = pickle.load(f)

LEVEL = "county"
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

SUBSETS = ["Smoke", "Non-smoke"]                       # top 2 rows, bottom 2 rows
panel_order = [p for p in ["n_pairs", "RMSE", "NMSE", "FE", "r", "COD", "MB", "FB"] if p in results]
panel_labels = {"n_pairs": "Number of Samples", "MB": "MB", "RMSE": "RMSE", "NMSE": "NMSE",
                "FB": "FB", "FE": "FE", "r": "r", "COD": "COD"}
DIVERGING = {"MB", "FB"}

# ── reproject county polygons once ───────────────────────────────────────────
ids, polys = CountyPolygon(STATES)
proj, xb, yb = [], [], []
for fid, poly in zip(ids, polys):
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    gg = []
    for g in geoms:
        lon, lat = g.exterior.xy
        xs, ys = to_lcc(np.asarray(lon), np.asarray(lat))
        gg.append((xs, ys)); xb += [xs.min(), xs.max()]; yb += [ys.min(), ys.max()]
    proj.append((fid, gg))
x0, x1, y0, y1 = min(xb), max(xb), min(yb), max(yb)

# ── shared color scale per metric (across both subsets) ──────────────────────
norms, cmaps = {}, {}
for name in panel_order:
    allv = []
    for sub in SUBSETS:
        allv += list(results[name][sub][LEVEL].values())
    arr = np.array(allv, float); arr = arr[np.isfinite(arr)]

    if name == "n_pairs":
        pos = arr[arr > 0]
        norms[name] = colors.LogNorm(vmin=max(1, pos.min()), vmax=pos.max())
        cmaps[name] = plt.get_cmap("viridis")
    elif name in DIVERGING:
        vmin, vmax = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        m = max(abs(vmin), abs(vmax))
        norms[name] = colors.Normalize(vmin=-m, vmax=m); cmaps[name] = plt.get_cmap("RdBu_r")
    else:
        vmin, vmax = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        norms[name] = colors.Normalize(vmin=vmin, vmax=vmax); cmaps[name] = plt.get_cmap("viridis")

# ── figure: 4 x 4 ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(22, 16))

for si, sub in enumerate(SUBSETS):
    for p, name in enumerate(panel_order):
        rr = 2 * si + p // 4
        cc = p % 4
        ax = axes[rr, cc]
        vd = results[name][sub][LEVEL]
        norm, cmap = norms[name], cmaps[name]

        for fid, geoms in proj:
            val = vd.get(fid, np.nan)
            face = cmap(norm(val)) if np.isfinite(val) else "none"   # NaN -> no color
            for xs, ys in geoms:
                ax.fill(xs, ys, facecolor=face, edgecolor="k", linewidth=0.15)

        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(panel_labels[name], fontsize=13, fontweight="bold")

        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
        cb = fig.colorbar(sm, cax=cax, extend="both" if name != "n_pairs" else "neither")
        cb.ax.tick_params(labelsize=9)

# subset row labels on the left
fig.text(0.005, 0.74, "Smoke",     rotation=90, va="center", fontsize=18, fontweight="bold")
fig.text(0.005, 0.30, "Non-smoke", rotation=90, va="center", fontsize=18, fontweight="bold")

fig.suptitle(r"UFS-AQM $PM_{2.5}$ Performance by County",
             fontsize=18, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0.02, 0, 1, 0.98])
fig.savefig("FigS10.png", bbox_inches="tight")
# plt.show()