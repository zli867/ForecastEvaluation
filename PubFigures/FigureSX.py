import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from utils.utils import assign_aqi_category, AQI_ORDER, AQI_CATEGORIES

obs_filename = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
df = pd.read_csv(obs_filename)

pm = df["PM25"].to_numpy(dtype=float)
pm = pm[np.isfinite(pm)]
labels = assign_aqi_category(pm)
total = pm.size

fig = plt.figure(figsize=(15, 11))
gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.25)

# ---- TOTAL: linear step-10 bins, but LOG x-axis, colored by category ------
ax_total = fig.add_subplot(gs[0, :])
tmax = np.ceil(pm.max() / 10.0) * 10
total_bins = np.arange(0, tmax + 10, 10)

counts, edges = np.histogram(pm, bins=total_bins)
pct = counts / total * 100
centers = 0.5 * (edges[:-1] + edges[1:])
bar_colors = [AQI_CATEGORIES[c]["color"] for c in assign_aqi_category(centers)]

ax_total.bar(edges[:-1], pct, width=np.diff(edges), align="edge",
             color=bar_colors, edgecolor="white", linewidth=0.3)
ax_total.set_yscale("log")                 # <-- just the axis, bins unchanged
ax_total.set_xlim(total_bins[1], tmax)     # start at 10, not 0 (log can't show 0)
ax_total.set_title(f"All observations (n={total:,})", fontsize=12, fontweight="bold")
ax_total.set_xlabel("PM2.5 (µg m$^{-3}$)")
ax_total.set_ylabel("% of all observations")
ax_total.legend(handles=[Patch(facecolor=AQI_CATEGORIES[c]["color"],
                               label=AQI_CATEGORIES[c]["label"]) for c in AQI_ORDER],
                fontsize=8, ncol=3, loc="upper right")

# ---- per-category panels: step 1, % within category -----------------------
cat_positions = [gs[1, 0], gs[1, 1], gs[1, 2], gs[2, 0], gs[2, 1], gs[2, 2]]
for cat, pos in zip(AQI_ORDER, cat_positions):
    ax = fig.add_subplot(pos)
    lo, hi = AQI_CATEGORIES[cat]["pm25"]
    color = AQI_CATEGORIES[cat]["color"]
    data_c = pm[labels == cat]

    if data_c.size:
        top = min(hi, data_c.max()) if np.isfinite(hi) else data_c.max()
        bins = np.arange(lo, max(top, lo + 1) + 1, 1)
        weights = np.full(data_c.size, 100.0 / data_c.size)
        ax.hist(data_c, bins=bins, weights=weights,
                color=color, edgecolor="white", linewidth=0.3)
        ax.set_xlim(lo, bins[-1])
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(lo, (hi if np.isfinite(hi) else lo + 1))

    frac = 100.0 * data_c.size / total
    ax.set_title(f"{AQI_CATEGORIES[cat]['label']}  "
                 f"(n={data_c.size:,}, {frac:.2f}% of all)", fontsize=10)
    ax.set_xlabel("PM2.5 (µg m$^{-3}$)")
    ax.set_ylabel("% of category")

fig.suptitle(r"$PM_{2.5}$ distribution", fontsize=14, fontweight="bold")
plt.show()