import pickle
import matplotlib.pyplot as plt

PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
temporal_path = f"{PUB}/temporal_avg_subsets.pkl"      # from the subset temporal script
spatial_path  = f"{PUB}/metric_vs_resolution_subsets.pkl"      # from the subset spatial script

SUBSETS       = ["All", "Smoke", "Non-smoke"]
subset_colors = {"All": "black", "Smoke": "tab:red", "Non-smoke": "tab:blue"}
resolutions   = {"site": 5.079, "county": 50, "state": 400}
ylabel_map      = {"r": "r", "RMSE": "RMSE", "NMSE": "NMSE", "COD": "COD", "MB": "MB", "FE": "FE", "FB": "FB"}
with open(temporal_path, "rb") as f:
    td = pickle.load(f)

with open(spatial_path, "rb") as f:
    performance = pickle.load(f)

scales   = td["scales"]
temporal = td["results"]

# metrics present in BOTH pickles
metrics = [mn for mn in performance if mn in temporal]
for METRIC in metrics:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # ── LEFT: temporal averaging (x = scale) ─────────────────────────────────
    for s in SUBSETS:
        axL.plot(scales, temporal[METRIC][s], marker="o", markersize=8,
                 markeredgecolor="k", markeredgewidth=0.6, linewidth=2.5,
                 color=subset_colors[s], label=s)
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
        print(s)
        axR.scatter(resolutions["state"], m[s]["state"], color=c, marker="^",
                    s=140, edgecolor="k", linewidth=0.8, zorder=5)
        print(m[s]["state"])
    # marker-shape legend (grey, shape only)
    axR.scatter([], [], marker="s", s=110, facecolor="grey", edgecolor="k", label="county (≈50 km)")
    axR.scatter([], [], marker="^", s=140, facecolor="grey", edgecolor="k", label="state (≈400 km)")
    axR.set_xscale("log")
    axR.set_title("Spatial Aggregation (km)", fontsize=15, fontweight="bold")
    axR.tick_params(axis="both", which="major", labelsize=14, length=8, width=2)
    axR.tick_params(axis="both", which="minor", length=4, width=2)
    axR.legend(frameon=False, fontsize=11, ncol=2)

    fig.suptitle(rf"UFS-AQM PM$_{{2.5}}$ {ylabel_map[METRIC]} Performance: Temporal or Spatial Aggregation",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()