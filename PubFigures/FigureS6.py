import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import AQI_ORDER

csv_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/metrics/regression_metrics_byAQI_06.csv"

# category order straight from the source constant (no drift vs the CSV)
CAT_ORDER = list(AQI_ORDER)
# short axis labels ("Unhealthy for Sensitive" -> "USG")
CAT_LABELS = ["USG" if c == "Unhealthy for Sensitive" else c for c in CAT_ORDER]

metric_order  = ["MB", "RMSE", "NMSE", "FB", "FE", "r", "COD"]
metric_labels = {"MB": r"MB ($\mu g\,m^{-3}$)", "RMSE": r"RMSE ($\mu g\,m^{-3}$)",
                 "NMSE": "NMSE", "FB": "FB (%)", "FE": "FE (%)", "r": "r", "COD": "COD"}
sub_colors = {"All": "k", "Smoke": "tab:red", "Non-smoke": "tab:blue"}

df = pd.read_csv(csv_path)
metrics = [m for m in metric_order if m in df["metric"].unique()]

ncols = 2
nrows = int(np.ceil(len(metrics) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows))
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    sub = df[df["metric"] == metric]
    for name, c in sub_colors.items():
        d = sub[sub["subset"] == name].set_index("category").reindex(CAT_ORDER)
        ax.plot(range(len(CAT_ORDER)), d["value"], "-o", color=c, markersize=6,
                markeredgecolor="k", markeredgewidth=0.4, linewidth=2.0, label=name)
    ax.set_ylabel(metric_labels[metric], fontsize=13)
    ax.set_xticks(range(len(CAT_ORDER)))
    ax.set_xticklabels(CAT_LABELS, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

# hide unused panels
for ax in axes[len(metrics):]:
    ax.set_visible(False)

# single legend
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, frameon=False, fontsize=13, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 1.0))

fig.suptitle("Performance within each observed AQI category (06z, all leads)",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()