import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

inits      = [6, 12]
save_tmpl  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/r2_byleadtime_init_%d.pkl"
subsets    = ["All", "Smoke", "Non-smoke"]
init_color = {6: "tab:blue", 12: "tab:red"}
metric, metric_label = "r2", "$r^2$"

results_all = {}
for it in inits:
    with open(save_tmpl % it, "rb") as f:
        results_all[it] = pickle.load(f)

fig, axes = plt.subplots(1, len(subsets), figsize=(16, 5.2), sharey=True)
for i, (ax, sub) in enumerate(zip(axes, subsets)):
    for it in inits:
        d    = results_all[it][metric][sub]
        lead = np.asarray(d["index"],  float)
        val  = np.asarray(d["values"], float)
        utc  = (it + lead) % 24                      # UTC hour-of-day, 0–23

        # collapse duplicate hours (leads >24 h share a UTC hour) -> one clean value/hour
        hours = np.unique(utc)
        vmean = np.array([np.nanmean(val[utc == h]) for h in hours])
        order = np.argsort(hours)

        ax.plot(hours[order], vmean[order], marker="o", markersize=6,
                markeredgecolor="k", markeredgewidth=0.5,
                linewidth=2.4, color=init_color[it], label=f"{it:02d}z init")

    ax.set_title(sub, fontsize=15, fontweight="bold")
    ax.set_xlabel("UTC Time (hr)", fontsize=14)
    if i == 0:
        ax.set_ylabel(f"{metric_label} Performance", fontsize=14)
    ax.set_xlim(-0.5, 23.5)
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", labelsize=13, length=8, width=1.5)
    ax.tick_params(axis="both", which="minor", length=4, width=1.5)
    ax.legend(frameon=False, fontsize=12)
fig.suptitle(f"{metric_label} by UTC time: 06z vs 12z init",
             fontsize=17, fontweight="bold")
fig.tight_layout()
plt.show()