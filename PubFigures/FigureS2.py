import os
import sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# ── config / paths ───────────────────────────────────────────────────────────
initial_time = 6
UTC_OFFSET   = 6
PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
leadr2_path  = os.path.join(PUB, "metrics_bylead_init_%d.pkl" % initial_time)

with open(leadr2_path, "rb") as f:
    results = pickle.load(f)

# metrics to plot (key in `results`, y-axis label)
metrics = [
    ("MB",   r"MB ($\mu g/m^3$)"),
    ("RMSE", r"RMSE ($\mu g/m^3$)"),
    ("NMSE", r"NMSE"),
    ("FB",   r"FB (%)"),
    ("FE",   r"FE (%)"),
    ("COD",  r"COD"),
]
colors = {"Smoke": "tab:red", "Non-smoke": "tab:blue"}

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

idx = 0
for ax, (mkey, ylabel) in zip(axes.flat, metrics):
    m = results[mkey]
    for name in ["Smoke", "Non-smoke"]:
        d = m[name]
        ax.plot(d["index"], d["values"],
                marker="o", markersize=7, markeredgecolor="k", markeredgewidth=0.6,
                linewidth=2.5, color=colors[name],
                label=f"{name}")
    if idx == 4:
        ax.set_xlabel("Lead Time (hr)", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", labelsize=13, length=8, width=2)
    ax.tick_params(axis="both", which="minor", length=4, width=2)
    ax.legend(frameon=False, fontsize=11)

    # secondary x-axis: UTC time (lead time + UTC_OFFSET, wrapped to 0–23)
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x + UTC_OFFSET, lambda x: x - UTC_OFFSET))
    if idx == 1:
        secax.set_xlabel("UTC Time (hr)", fontsize=15)
    secax.xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: f"{int(round(val)) % 24:02d}"))
    secax.xaxis.set_minor_locator(MultipleLocator(1))
    secax.tick_params(axis="x", which="major", labelsize=13, length=8, width=2)
    secax.tick_params(axis="x", which="minor", length=4, width=2)
    idx += 1
fig.tight_layout()
plt.show()