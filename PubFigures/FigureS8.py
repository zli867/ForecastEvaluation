import pickle
import matplotlib.pyplot as plt

PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
temporal_path = f"{PUB}/temporal_avg_subsets.pkl"
spatial_path  = f"{PUB}/metric_vs_resolution_subsets.pkl"

SUBSETS       = ["All", "Smoke", "Non-smoke"]
subset_colors = {"All": "black", "Smoke": "tab:red", "Non-smoke": "tab:blue"}
resolutions   = {"site": 5.079, "county": 50, "state": 400}
ylabel_map    = {"r": "r", "RMSE": "RMSE", "NMSE": "NMSE", "COD": "COD",
                 "MB": "MB", "FE": "FE", "FB": "FB"}

with open(temporal_path, "rb") as f:
    td = pickle.load(f)
with open(spatial_path, "rb") as f:
    performance = pickle.load(f)

scales   = td["scales"]
temporal = td["results"]

# the 6 metrics other than r, present in BOTH pickles, in a fixed order
metric_order = ["MB", "RMSE", "NMSE", "FB", "FE", "COD"]
metrics = [m for m in metric_order if m in performance and m in temporal]
print(metrics)

nrows = len(metrics)
fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.0 * nrows), sharey="row")

for i, METRIC in enumerate(metrics):
    axL, axR = axes[i, 0], axes[i, 1]

    # ── LEFT: temporal averaging ─────────────────────────────────────────────
    for s in SUBSETS:
        axL.plot(scales, temporal[METRIC][s], marker="o", markersize=7,
                 markeredgecolor="k", markeredgewidth=0.6, linewidth=2.5,
                 color=subset_colors[s], label=s)
    axL.set_ylabel(ylabel_map[METRIC], fontsize=16)
    axL.tick_params(axis="both", which="major", labelsize=13, length=7, width=1.8)

    # ── RIGHT: spatial resolution (log x) ────────────────────────────────────
    m = performance[METRIC]
    grid_res = sorted(k for k in m["All"] if isinstance(k, float))
    line_res = [resolutions["site"]] + grid_res
    for s in SUBSETS:
        c = subset_colors[s]
        curve = [m[s]["site"]] + [m[s][r] for r in grid_res]
        axR.plot(line_res, curve, marker="o", markersize=7, markeredgecolor="k",
                 markeredgewidth=0.6, linewidth=2.5, color=c, label=s)
        axR.scatter(resolutions["county"], m[s]["county"], color=c, marker="s",
                    s=100, edgecolor="k", linewidth=0.8, zorder=5)
        if METRIC == "COD" and s == "Smoke":
            axR.scatter(resolutions["state"], m[s]["state"], color=c, marker="v",
                        s=130, edgecolor="k", linewidth=0.8, zorder=100)
        else:
            axR.scatter(resolutions["state"], m[s]["state"], color=c, marker="^",
                        s=130, edgecolor="k", linewidth=0.8, zorder=5)
        print(s)
        if METRIC == "FE" and s == "All":
            axR.scatter(resolutions["state"], m[s]["state"], color=c, marker="v",
                        s=130, edgecolor="k", linewidth=0.8, zorder=100)
        else:
            axR.scatter(resolutions["state"], m[s]["state"], color=c, marker="^",
                        s=130, edgecolor="k", linewidth=0.8, zorder=5)
    axR.set_xscale("log")
    axR.tick_params(axis="both", which="major", labelsize=13, length=7, width=1.8)
    axR.tick_params(axis="both", which="minor", length=4, width=1.8)

    # column titles on the top row only
    if i == 0:
        axL.set_title("Temporal Aggregation", fontsize=15, fontweight="bold")
        axR.set_title("Spatial Aggregation (km)", fontsize=15, fontweight="bold")
        axL.legend(frameon=False, fontsize=11)
        # marker-shape legend (grey, shape only)
        axR.scatter([], [], marker="s", s=100, facecolor="grey", edgecolor="k", label="county (≈50 km)")
        axR.scatter([], [], marker="^", s=130, facecolor="grey", edgecolor="k", label="state (≈400 km)")
        axR.legend(frameon=False, fontsize=10, ncol=2)

    # x labels on the bottom row only
    if i == nrows - 1:
        axL.set_xlabel("Temporal scale", fontsize=14)
        axR.set_xlabel("Spatial resolution (km)", fontsize=14)

fig.suptitle(r"UFS-AQM PM$_{2.5}$ Performance: Temporal or Spatial Aggregation",
             fontsize=17, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("Figure_S8.png", bbox_inches="tight")
# plt.show()