import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable

from utils.metrics import (
    R2, RMSE, NMSE, grouped_metric
)
from utils.geoutil import (
    FIPS_list, StatePolygon, CountyPolygon, interest_domain_bound,
)
import pickle
from pyproj import Transformer

# PLOT FIGURES
# WGS84 lon/lat -> USA Contiguous Lambert Conformal Conic (ESRI:102004)
to_lcc = Transformer.from_crs("EPSG:4326", "ESRI:102004", always_xy=True).transform
save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/state_county_performance.pkl"
with open(save_path, "rb") as f:
    results = pickle.load(f)
LEVELS = ["state", "county"]
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

for metric_name in results:
    value_dicts = results[metric_name]
    # ---- shared color scale across both panels ----
    all_vals = np.array([v for vd in value_dicts.values() for v in vd.values()], float)
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin = np.nanpercentile(all_vals, 2)
    vmax = np.nanpercentile(all_vals, 98)
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # # ---- side-by-side plot ----
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(12, 6))
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, LEVELS):
        vd = value_dicts[level]
        if level == "state":
            id_poly = list(zip(FIPS_list(STATES), StatePolygon(STATES)))
        else:
            ids, polys = CountyPolygon(STATES)
            id_poly = list(zip(ids, polys))

        n_total = 0  # total regions plotted
        n_colored = 0  # regions with a real (finite) value

        xb, yb = [], []  # collect projected bounds
        for fid, poly in id_poly:
            val = vd.get(fid, np.nan)
            face = cmap(norm(val)) if np.isfinite(val) else "lightgrey"
            n_total += 1
            if np.isfinite(val):
                n_colored += 1
            geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
            for g in geoms:
                lon, lat = g.exterior.xy
                xs, ys = to_lcc(np.asarray(lon), np.asarray(lat))  # reproject to LCC
                ax.fill(xs, ys, facecolor=face, edgecolor="k", linewidth=0.4)
                xb.extend([xs.min(), xs.max()])
                yb.extend([ys.min(), ys.max()])

        print(f"[{metric_name}] {level}: {n_colored} of {n_total} regions colored "
              f"({n_total - n_colored} grey/missing)")

        ax.set_xlim(min(xb), max(xb))
        ax.set_ylim(min(yb), max(yb))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{level.capitalize()}", fontsize=18)
    # for ax, level in zip(axes, LEVELS):
    #     vd = value_dicts[level]
    #     if level == "state":
    #         id_poly = list(zip(FIPS_list(STATES), StatePolygon(STATES)))
    #     else:
    #         ids, polys = CountyPolygon(STATES)
    #         id_poly = list(zip(ids, polys))
    #
    #     xb, yb = [], []   # collect projected bounds
    #     for fid, poly in id_poly:
    #         val = vd.get(fid, np.nan)
    #         face = cmap(norm(val)) if np.isfinite(val) else "lightgrey"
    #         geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    #         for g in geoms:
    #             lon, lat = g.exterior.xy
    #             xs, ys = to_lcc(np.asarray(lon), np.asarray(lat))   # reproject to LCC
    #             ax.fill(xs, ys, facecolor=face, edgecolor="k", linewidth=0.4)
    #             xb.extend([xs.min(), xs.max()])
    #             yb.extend([ys.min(), ys.max()])
    #
    #     ax.set_xlim(min(xb), max(xb))
    #     ax.set_ylim(min(yb), max(yb))
    #     ax.set_aspect("equal")
    #     ax.axis("off")
    #     ax.set_title(f"{level.capitalize()}", fontsize=18)

    # ---- shared colorbar ----
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, extend="both")
    if metric_name == "R2":
        metric_name = "$R^2$"
    cb.set_label(metric_name + " Performance", fontsize=16)
    cb.ax.tick_params(labelsize=14, length=6, width=2)

    fig.suptitle(rf'UFS-AQM $PM_{{2.5}}$ — {metric_name} by Region',
                 fontsize=16, fontweight="bold", y=0.98)
    plt.show()