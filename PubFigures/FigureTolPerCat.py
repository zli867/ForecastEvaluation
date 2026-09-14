import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import remove_nan_values, assign_aqi_category, AQI_ORDER
from utils.Visualizations import plot_aqi_confusion_matrix_log
import pandas as pd

AQI_TOLERANCE = {
    "Good": 0.9,
    "Moderate": 2.65,
    "Unhealthy for Sensitive": 2.0,
    "Unhealthy": 7.0,
    "Very Unhealthy": 10.0,
    "Hazardous": 10.0,
}

def aqi_category_agreement_tol(prediction, observation, tol_by_category):
    """
    Same as aqi_category_agreement, but an adjacent cross-category prediction
    is counted as correct when its concentration difference from the observation
    is within the tolerance assigned to the observed AQI category.
    """
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    pred_clean, obs_clean = remove_nan_values(prediction, observation)

    pred_cats = assign_aqi_category(pred_clean)
    obs_cats = assign_aqi_category(obs_clean)

    order_idx = {c: i for i, c in enumerate(AQI_ORDER)}
    pred_idx = np.array([order_idx[c] for c in pred_cats])
    obs_idx = np.array([order_idx[c] for c in obs_cats])

    category_tol = np.array([tol_by_category[c] for c in obs_cats])

    # Forgive only adjacent category crossings within the observed-category tolerance.
    forgiven = (
        (np.abs(pred_idx - obs_idx) == 1)
        & (np.abs(pred_clean - obs_clean) <= category_tol)
    )
    eff_pred_idx = np.where(forgiven, obs_idx, pred_idx)

    hit_rate = np.mean(eff_pred_idx == obs_idx)
    over_rate = np.mean(eff_pred_idx > obs_idx)
    under_rate = np.mean(eff_pred_idx < obs_idx)

    n = len(AQI_ORDER)
    matrix = np.zeros((n, n), dtype=int)
    for oi, pi in zip(obs_idx, eff_pred_idx):
        matrix[oi, pi] += 1

    conf_df = pd.DataFrame(
        matrix,
        index=[f"Obs: {c}" for c in AQI_ORDER],
        columns=[f"Pred: {c}" for c in AQI_ORDER],
    )

    return {
        "hit_rate": hit_rate,
        "over_rate": over_rate,
        "under_rate": under_rate,
        "n_forgiven": int(forgiven.sum()),
        "confusion_matrix": conf_df,
    }

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

    agreement = aqi_category_agreement_tol(sdf[MODEL_VAR].values, sdf[OBS_VAR].values, AQI_TOLERANCE)
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