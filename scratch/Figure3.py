
import pickle
import matplotlib.pyplot as plt
from utils.metrics import aqi_category_agreement
from utils.Visualizations import plot_aqi_confusion_matrix_log


save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_aqi_subsets.pkl"
MODEL_VAR   = "PM25_UFSAQM"
OBS_VAR     = "PM25"
# reload
with open(save_path, "rb") as f:
    subsets_dict = pickle.load(f)
subsets = list(subsets_dict.items())
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
idx = 0
for subset_name, sdf in subsets:
    if subset_name == "All": continue
    if len(sdf) == 0:
        print(f"[{subset_name}] no records — skipping")
        continue

    agreement = aqi_category_agreement(sdf[MODEL_VAR].values, sdf[OBS_VAR].values)
    plot_aqi_confusion_matrix_log(
        agreement["confusion_matrix"], ax=axs[idx],
        title=f"UFS-AQM ({subset_name}) AQI Confusion Matrix"
    )
    axs[idx].set_xlabel("Predicted Category", fontsize=11)
    if idx == 0:
        axs[idx].set_ylabel("Observed Category", fontsize=11)
    idx += 1
fig.tight_layout()
plt.show()