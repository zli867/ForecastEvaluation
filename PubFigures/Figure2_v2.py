import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from utils.Visualizations import plotComparisonIntercept_fast

# ── config / paths ───────────────────────────────────────────────
initial_time = 6
UTC_OFFSET   = 6
PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
scatter_path = os.path.join(PUB, "scatter_data_%02d.pkl" % initial_time)
leadr2_path  = os.path.join(PUB, "metrics_bylead_init_%d.pkl" % initial_time)

with open(scatter_path, "rb") as f:
    scat = pickle.load(f)
with open(leadr2_path, "rb") as f:
    results = pickle.load(f)

# ── layout: (a),(b) on top row; (c) spanning the full bottom row ──
fig, axd = plt.subplot_mosaic(
    [["a", "b"],
     ["c", "c"]],
    figsize=(12, 9),
    gridspec_kw={"height_ratios": [1.35, 1]},
)
lim = [0, 300]

# ── top row: scatter panels ──────────────────────────────────────
for ax, key, title in [(axd["a"], "Smoke",     "Smoke Days"),
                        (axd["b"], "Non-smoke", "Non-smoke Days")]:
    X, Y = scat[key]["obs"], scat[key]["model"]
    reg_txt = plotComparisonIntercept_fast(X, Y, bin_size=1000, ax=ax, fig=fig)
    ax.plot(lim, lim, "k--", linewidth=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Observed $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
    ax.set_ylabel(r"UFS-AQM $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
    ax.set_title(f"{title}\n{reg_txt}\nn={len(X):,}", fontsize=15, fontweight="bold")
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis="both", labelsize=14)
    ax.set_aspect("equal")

# ── bottom row: r vs lead time (full width) ──────────────────────
axC = axd["c"]
r = results["r"]
colors = {"Smoke": "tab:red", "Non-smoke": "tab:blue"}
for name in ["Smoke", "Non-smoke"]:
    d = r[name]
    axC.plot(d["index"], d["values"],
             marker="o", markersize=8, markeredgecolor="k", markeredgewidth=0.6,
             linewidth=2.5, color=colors[name],
             label=f"{name} (n={d['length']:,})")
axC.set_xlabel("Lead Time (hr)", fontsize=16)
axC.set_ylabel(r"r Performance", fontsize=16)
axC.set_ylim(0.2, 0.4)                       # full 0–1 range so the trend isn't visually exaggerated
axC.yaxis.set_major_locator(MultipleLocator(0.1))
axC.yaxis.set_minor_locator(MultipleLocator(0.05))
axC.xaxis.set_minor_locator(MultipleLocator(1))
axC.tick_params(axis="both", which="major", labelsize=14, length=8, width=2)
axC.tick_params(axis="both", which="minor", length=4, width=2)
axC.legend(frameon=False, fontsize=12)

# secondary x-axis: UTC time (lead time + UTC_OFFSET, wrapped to 0–23)
secax = axC.secondary_xaxis(
    "top",
    functions=(lambda x: x + UTC_OFFSET, lambda x: x - UTC_OFFSET))
secax.set_xlabel("UTC Time (hr)", fontsize=16)
secax.xaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f"{int(round(val)) % 24:02d}"))
secax.xaxis.set_minor_locator(MultipleLocator(1))
secax.tick_params(axis="x", which="major", labelsize=14, length=8, width=2)
secax.tick_params(axis="x", which="minor", length=4, width=2)

fig.tight_layout()
plt.show()