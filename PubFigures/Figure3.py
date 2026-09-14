import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import aqi_category_agreement
from utils.Visualizations import plot_aqi_confusion_matrix_log

save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_aqi_subsets.pkl"
MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"

with open(save_path, "rb") as f:
    subsets_dict = pickle.load(f)
subsets = list(subsets_dict.items())

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
idx = 0
for subset_name, sdf in subsets:
    if subset_name == "All":
        continue
    if len(sdf) == 0:
        print(f"[{subset_name}] no records — skipping")
        continue

    agreement = aqi_category_agreement(sdf[MODEL_VAR].values, sdf[OBS_VAR].values)
    cm = agreement["confusion_matrix"]          # keep as DataFrame for the plot fn

    ax = axs[idx]
    plot_aqi_confusion_matrix_log(
        cm, ax=ax,
        title=f"UFS-AQM ({subset_name}) AQI Confusion Matrix"
    )

    # sums from the underlying array: rows = observed, cols = predicted
    row_sums = cm.values.sum(axis=1)
    col_sums = cm.values.sum(axis=0)

    xlabs = [t.get_text() for t in ax.get_xticklabels()]
    ylabs = [t.get_text() for t in ax.get_yticklabels()]
    ax.set_xticklabels([f"{lab}\n({int(s):,})" for lab, s in zip(xlabs, col_sums)])
    ax.set_yticklabels([f"{lab}\n({int(s):,})" for lab, s in zip(ylabs, row_sums)])

    ax.set_xlabel("Predicted Category", fontsize=11)
    if idx == 0:
        ax.set_ylabel("Observed Category", fontsize=11)
    idx += 1

fig.tight_layout()
plt.show()