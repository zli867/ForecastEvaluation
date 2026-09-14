import pickle
import matplotlib.pyplot as plt

PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
temporal_path = f"{PUB}/site_r2_temporal_avg_subsets.pkl"      # from the subset temporal script
spatial_path  = f"{PUB}/metric_vs_resolution_subsets.pkl"      # from the subset spatial script

SUBSETS       = ["All", "Smoke", "Non-smoke"]
subset_colors = {"All": "black", "Smoke": "tab:red", "Non-smoke": "tab:blue"}
resolutions   = {"site": 5.079, "county": 50, "state": 400}
ylabel_map      = {"r": "r", "RMSE": "RMSE", "NMSE": "NMSE", "R2": r"$R^{2}$", "MB": "MB", "FE": "FE", "FB": "FB"}
with open(temporal_path, "rb") as f:
    td = pickle.load(f)

with open(spatial_path, "rb") as f:
    performance = pickle.load(f)

scales   = td["scales"]
temporal = td["results"]

# metrics present in BOTH pickles
metrics = [mn for mn in performance if mn in temporal]
print(metrics)
for METRIC in metrics:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # ── LEFT: temporal averaging (x = scale) ─────────────────────────────────
    for s in SUBSETS:
        axL.plot(scales, temporal[METRIC][s], marker="o", markersize=8,
                 markeredgecolor="k", markeredgewidth=0.6, linewidth=2.5,
                 color=subset_colors[s], label=s)
    axL.set_xlabel("Temporal Resolution", fontsize=16)
    axL.set_ylabel(ylabel_map[METRIC], fontsize=16)
    axL.set_title("Temporal Aggregation", fontsize=15, fontweight="bold")
    axL.tick_params(axis="both", which="major", labelsize=14, length=8, width=2)
    axL.legend(frameon=False, fontsize=12)

    # ── RIGHT: spatial resolution (x = resolution, log) ──────────────────────
    m = performance[METRIC]
    grid_res = sorted(k for k in m["All"] if isinstance(k, float))
    line_res = [resolutions["site"]] + grid_res
    for s in SUBSETS:
        c = subset_colors[s]
        curve = [m[s]["site"]] + [m[s][r] for r in grid_res]
        axR.plot(line_res, curve, marker="o", markersize=8, markeredgecolor="k",
                 markeredgewidth=0.6, linewidth=2.5, color=c, label=s)
        axR.scatter(resolutions["county"], m[s]["county"], color=c, marker="s",
                    s=110, edgecolor="k", linewidth=0.8, zorder=5)
        axR.scatter(resolutions["state"], m[s]["state"], color=c, marker="^",
                    s=140, edgecolor="k", linewidth=0.8, zorder=5)
    # marker-shape legend (grey, shape only)
    axR.scatter([], [], marker="o", s=70,  facecolor="grey", edgecolor="k", label="gridded incl. site (≈5 km)")
    axR.scatter([], [], marker="s", s=110, facecolor="grey", edgecolor="k", label="county (≈50 km)")
    axR.scatter([], [], marker="^", s=140, facecolor="grey", edgecolor="k", label="state (≈400 km)")
    axR.set_xscale("log")
    axR.set_xlabel("Spatial Resolution (km)", fontsize=16)
    axR.set_title("Spatial Aggregation", fontsize=15, fontweight="bold")
    axR.tick_params(axis="both", which="major", labelsize=14, length=8, width=2)
    axR.tick_params(axis="both", which="minor", length=4, width=2)
    axR.legend(frameon=False, fontsize=11, ncol=2)

    fig.suptitle(rf"UFS-AQM PM$_{{2.5}}$ {ylabel_map[METRIC]} Performance: temporal or spatial aggregation",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # fig.savefig(f"{PUB}/{METRIC}_temporal_spatial_subsets.png", dpi=200, bbox_inches="tight")
    plt.show()


# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
# PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
# temporal_path = f"{PUB}/site_r2_temporal_avg_subsets.pkl"
# spatial_path  = f"{PUB}/metric_vs_resolution_subsets.pkl"
#
# SUBSETS       = ["All", "Smoke", "Non-smoke"]
# subset_colors = {"All": "black", "Smoke": "tab:red", "Non-smoke": "tab:blue"}
# SITE_RES      = 5.079
# ylabel_map    = {"r2": r"$r^{2}$", "R2": r"$R^{2}$", "RMSE": "RMSE", "NMSE": "NMSE"}
#
# with open(temporal_path, "rb") as f:
#     td = pickle.load(f)
# with open(spatial_path, "rb") as f:
#     sp = pickle.load(f)
#
# scales      = td["scales"]
# temporal    = td["results"]
# performance = sp["performance"]      # site + gridded line
# region_perf = sp["region_perf"]      # per-state / per-county clouds
#
# metrics = [mn for mn in performance if mn in temporal]
#
# for METRIC in metrics:
#     fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
#
#     # ── LEFT: temporal averaging ─────────────────────────────────────────────
#     for s in SUBSETS:
#         axL.plot(scales, temporal[METRIC][s], marker="o", markersize=8,
#                  markeredgecolor="k", markeredgewidth=0.6, linewidth=2.5,
#                  color=subset_colors[s], label=s)
#     axL.set_xlabel("Temporal Resolution", fontsize=16)
#     axL.set_ylabel(ylabel_map[METRIC], fontsize=16)
#     axL.set_title("Temporal Aggregation", fontsize=15, fontweight="bold")
#     axL.tick_params(axis="both", which="major", labelsize=14, length=8, width=2)
#     axL.legend(frameon=False, fontsize=12)
#
#     # ── RIGHT: spatial — gridded line + per-region clouds ────────────────────
#     m = performance[METRIC]
#     grid_res = sorted(k for k in m["All"] if isinstance(k, float))
#     line_res = [SITE_RES] + grid_res
#     for s in SUBSETS:
#         c = subset_colors[s]
#         # gridded (incl. native site) line
#         curve = [m[s]["site"]] + [m[s][r] for r in grid_res]
#         axR.plot(line_res, curve, marker="o", markersize=8, markeredgecolor="k",
#                  markeredgewidth=0.6, linewidth=2.5, color=c, label=s, zorder=4)
#         # per-county cloud (small, translucent)
#         dc = region_perf[s]["counties"]
#         axR.scatter(dc["resolution_km"], dc[METRIC], s=12, alpha=0.25,
#                     color=c, edgecolor="none", zorder=2)
#         # per-state cloud (larger, outlined)
#         dst = region_perf[s]["states"]
#         axR.scatter(dst["resolution_km"], dst[METRIC], s=70, alpha=0.7,
#                     color=c, marker="s", edgecolor="k", linewidth=0.6, zorder=3)
#
#     # shape legend (grey)
#     axR.scatter([], [], marker="o", s=70,  facecolor="grey", edgecolor="k", label="gridded incl. site")
#     axR.scatter([], [], marker=".", s=40,  facecolor="grey", edgecolor="none", label="per-county")
#     axR.scatter([], [], marker="s", s=70,  facecolor="grey", edgecolor="k", label="per-state")
#
#     axR.set_xscale("log")
#     axR.set_xlabel("Spatial Resolution (km)", fontsize=16)
#     axR.set_title("Spatial Aggregation", fontsize=15, fontweight="bold")
#     axR.tick_params(axis="both", which="major", labelsize=14, length=8, width=2)
#     axR.tick_params(axis="both", which="minor", length=4, width=2)
#     axR.legend(frameon=False, fontsize=11, ncol=2)
#     axR.set_ylim([0, 4])
#     fig.suptitle(rf"UFS-AQM PM$_{{2.5}}$ {ylabel_map[METRIC]} Performance: temporal or spatial aggregation",
#                  fontsize=17, fontweight="bold")
#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     # fig.savefig(f"{PUB}/{METRIC}_temporal_spatial_subsets.png", dpi=200, bbox_inches="tight")
#     plt.show()