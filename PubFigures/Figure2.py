import os
import sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from utils.Visualizations import plotComparisonIntercept_fast

# ── config / paths ───────────────────────────────────────────────────────────
initial_time = 6
UTC_OFFSET   = 6
PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
scatter_path = os.path.join(PUB, "scatter_data_%02d.pkl" % initial_time)
leadr2_path  = os.path.join(PUB, "site_r2_smokenonsmoke_subsets_%02d_v7.pkl" % initial_time)

with open(scatter_path, "rb") as f:
    scat = pickle.load(f)
with open(leadr2_path, "rb") as f:
    results = pickle.load(f)


fig, axes = plt.subplots(1, 3, figsize=(20, 6))
lim = [0, 300]

for ax, key, title in [(axes[0], "Smoke",     "Smoke Days"),
                        (axes[1], "Non-smoke", "Non-smoke Days")]:
    X, Y = scat[key]["obs"], scat[key]["model"]
    reg_txt = plotComparisonIntercept_fast(X, Y, bin_size=1000, ax=ax, fig=fig)
    ax.plot(lim, lim, "k--", linewidth=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Observed $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
    ax.set_ylabel(r"UFS-AQM $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
    ax.set_title(f"{title}\n{reg_txt} (n={len(X):,})", fontsize=15, fontweight="bold")
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis="both", labelsize=14)
    ax.set_aspect("equal")

# ── Panel C: R2 by lead time — Smoke (red) vs Non-smoke (blue) ────────────────
axC = axes[2]
r2 = results["R2"]
colors = {"Smoke": "tab:red", "Non-smoke": "tab:blue"}
for name in ["Smoke", "Non-smoke"]:
    d = r2[name]
    axC.plot(d["index"], d["values"],
             marker="o", markersize=8, markeredgecolor="k", markeredgewidth=0.6,
             linewidth=2.5, color=colors[name],
             label=f"{name} (n={d['length']:,})")
axC.set_xlabel("Lead Time (hr)", fontsize=16)
axC.set_ylabel(r"$R^2$ Performance", fontsize=16)
axC.set_title(f"{initial_time:02d}z Initialization\n$R^2$ by Lead Time",
              fontsize=15, fontweight="bold")
axC.xaxis.set_minor_locator(MultipleLocator(1))
axC.tick_params(axis="both", which="major", labelsize=14, length=8, width=2)
axC.tick_params(axis="both", which="minor", length=4, width=2)
axC.legend(frameon=False, fontsize=12)

# ── secondary x-axis: UTC time (lead time + UTC_OFFSET, wrapped to 0–23) ──────
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
