import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

lead_times = [6, 12]
save_tmpl = ("/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/metrics_bylead_init_%d.pkl")
colors = {"All": 'k', "Smoke": 'red', "Non-smoke": 'blue'}

results_all = {}
for lt in lead_times:
    with open(save_tmpl % lt, "rb") as f:
        results_all[lt] = pickle.load(f)

# desired metric order + axis labels (only keep those present in the pickle)
order  = ["MB", "RMSE", "NMSE", "FB", "FE", "r", "COD"]
labels = {"MB": r"MB ($\mu g\,m^{-3}$)", "RMSE": r"RMSE ($\mu g\,m^{-3}$)",
          "NMSE": "NMSE", "FB": "FB (%)", "FE": "FE (%)", "r": "r", "COD": "COD"}
present = list(results_all[lead_times[0]].keys())
metrics = [m for m in order if m in present]

nrows, ncols = len(metrics), len(lead_times)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows),
                         sharex=True, sharey="row")

for r, metric in enumerate(metrics):
    for c, lt in enumerate(lead_times):
        ax  = axes[r, c]
        res = results_all[lt][metric]
        for name in res:
            ax.plot(res[name]["index"], res[name]["values"],
                    marker="o", markersize=6, markeredgecolor="k", markeredgewidth=0.5,
                    linewidth=2.0, label=name, color=colors[name])

        ax.tick_params(axis="both", which="major", labelsize=12, length=7, width=1.6)
        ax.tick_params(axis="both", which="minor", length=4, width=1.6)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        if c == 0:
            ax.set_ylabel(labels[metric], fontsize=14)
        if r == 0:
            ax.set_title(f"{lt:02d}z Initialization", fontsize=15, fontweight="bold", pad=28)
        if r == nrows - 1:
            ax.set_xlabel("Lead Time (hr)", fontsize=14)

        # secondary UTC axis on the top row only (offset differs per column)
        if r == 0:
            secax = ax.secondary_xaxis(
                "top",
                functions=(lambda x, off=lt: x + off,
                           lambda x, off=lt: x - off))
            secax.set_xlabel("UTC Time (hr)", fontsize=13)
            secax.xaxis.set_major_locator(MultipleLocator(4))
            secax.xaxis.set_major_formatter(
                FuncFormatter(lambda val, pos: f"{int(round(val)) % 24:02d}"))
            secax.tick_params(axis="x", which="major", labelsize=12, length=7, width=1.6)

# single legend (from the first panel) at the top
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, frameon=False, fontsize=13, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 1.0))

fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()