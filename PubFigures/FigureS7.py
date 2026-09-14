import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import AQI_ORDER

csv_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/metrics/classification_metrics_06.csv"

CAT_ORDER  = list(AQI_ORDER)                                  # per-category only (drop "Overall")
CAT_LABELS = ["USG" if c == "Unhealthy for Sensitive" else c for c in CAT_ORDER]

metric_order  = ["hit_rate", "false_alarm_ratio", "F1"]
metric_labels = {"hit_rate": "Hit Rate", "false_alarm_ratio": "False Alarm Ratio", "F1": "F1"}
sub_colors    = {"All": "k", "Smoke": "red", "Non-smoke": "blue"}

df = pd.read_csv(csv_path)
metrics = [m for m in metric_order if m in df["metric"].unique()]

fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
if len(metrics) == 1:
    axes = [axes]

for ax, metric in zip(axes, metrics):
    sub = df[df["metric"] == metric]
    for name, c in sub_colors.items():
        d = sub[sub["subset"] == name].set_index("category").reindex(CAT_ORDER)
        ax.plot(range(len(CAT_ORDER)), d["value"], "-o", color=c, markersize=6,
                markeredgecolor="k", markeredgewidth=0.4, linewidth=2.0, label=name)
    ax.set_ylabel(metric_labels[metric], fontsize=13)
    ax.set_ylim(0, 1)                                         # all three are in [0, 1]
    ax.set_xticks(range(len(CAT_ORDER)))
    ax.set_xticklabels(CAT_LABELS, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

# single legend
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, frameon=False, fontsize=13, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Classification performance by observed AQI category (06z, all leads)",
             fontsize=15, fontweight="bold", y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()