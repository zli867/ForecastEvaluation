# ── PART 1: generate temporary obs/model pair data for the scatter plot ─────
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

start_time, end_time = datetime(2021, 1, 1), datetime(2024, 12, 31)
MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"
initial_time = 6

obs_file     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time
save_path    = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/scatter_data_%02d.pkl" % initial_time
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"


with open(stn_locs_file, 'rb') as file:
    site_dict = pickle.load(file)
obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]

model_list = []
cur_time = start_time
while cur_time <= end_time:
    date_str = cur_time.strftime("%Y%m%d")
    cur_model_file = os.path.join(stats_folder, f"site_level_aqm_{date_str}_%02d.csv" % initial_time)
    if os.path.exists(cur_model_file):
        model_list.append(pd.read_csv(cur_model_file, parse_dates=["Time"], dtype={"siteCode": str}))
    cur_time += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)

# ── merge & clean ────────────────────────────────────────────────────────────
merged = model_df.merge(
    obs_df[["siteCode", "Time", OBS_VAR]],
    on=["siteCode", "Time"], how="inner"
)
merged = merged.dropna(subset=[MODEL_VAR, OBS_VAR])
merged = merged[(merged[OBS_VAR] >= 0) & (merged[MODEL_VAR] >= 0)]
# ── Smoke flag from latest stn_info (site_dict["smoke_dates"]) ──────────────
smoke_day_sets = {
    c: {d.date() for d in site_dict[c].get("smoke_dates", [])}
    for c in site_dict
}
merged["isSmokeDay"] = [
    code in smoke_day_sets and t.date() in smoke_day_sets[code]
    for code, t in zip(merged["siteCode"], merged["Time"])
]
subsets_dict = {
    "All":       merged,
    "Smoke":     merged[merged["isSmokeDay"]],
    "Non-smoke": merged[~merged["isSmokeDay"]],
}

data = {
    "All": {
        "obs":   subsets_dict["All"][OBS_VAR].to_numpy(float),
        "model": subsets_dict["All"][MODEL_VAR].to_numpy(float)
    },
    "Smoke": {
        "obs":   subsets_dict["Smoke"][OBS_VAR].to_numpy(float),
        "model": subsets_dict["Smoke"][MODEL_VAR].to_numpy(float)
    },
    "Non-smoke": {
        "obs": subsets_dict["Non-smoke"][OBS_VAR].to_numpy(float),
        "model": subsets_dict["Non-smoke"][MODEL_VAR].to_numpy(float)
    }
}

with open(save_path, "wb") as f:
    pickle.dump(data, f)


# # ── PART 2: hexbin density scatter of model vs obs, 06 & 12 lead times ──────
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from matplotlib.ticker import MultipleLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy import stats
# from sklearn.linear_model import LinearRegression
# from utils.Visualizations import plotComparisonIntercept, plotComparisonIntercept_fast
#
# lead_times = [6, 12]
# save_tmpl = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/scatter_data_%02d.pkl"
#
#
# def get_conf_intercept(alpha, lr, X, y):
#     """(1-alpha) 2-sided confidence intervals for LinearRegression coefficients."""
#     coefs = np.r_[[lr.intercept_], lr.coef_]
#     X_aux = np.zeros((X.shape[0], X.shape[1] + 1))
#     X_aux[:, 1:] = X
#     X_aux[:, 0] = 1
#     dof = -np.diff(X_aux.shape)[0]
#     mse = np.sum((y - lr.predict(X)) ** 2) / dof
#     var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
#     t_val = stats.t.isf(alpha / 2, dof)
#     gap = t_val * np.sqrt(mse * var_params)
#     return {"coeffs": coefs, "lower": coefs - gap, "upper": coefs + gap}
#
#
# # def plotComparisonHexbin(X, Y, bin_size, ax, fig, mincnt=100, vmax=None):
# #     hb = ax.hexbin(
# #         X, Y,
# #         gridsize=bin_size,
# #         mincnt=mincnt,
# #         linewidths=0.1,
# #         norm=LogNorm(vmax=vmax),
# #     )
# #     divider = make_axes_locatable(ax)
# #     d_cax = divider.append_axes('right', size='5%', pad=0.05)
# #     cb = fig.colorbar(hb, cax=d_cax)
# #     cb.set_label('Counts', fontsize=16, rotation=270, labelpad=15)
# #
# #     X_r = X.reshape(-1, 1)
# #     lin_model = LinearRegression().fit(X_r, Y)
# #     line_x = np.linspace(0, np.max(X_r)).reshape(-1, 1)
# #     line_y = lin_model.predict(line_x)
# #
# #     cf = get_conf_intercept(0.05, lin_model, X_r, Y)
# #     r2_score = lin_model.score(X_r, Y)
# #     reg_txt = r'y={:.2f}x+{:.1f}, $\mathbf{{R^{{2}}}}$ = {:.2f}'.format(
# #         cf["coeffs"][1], cf["coeffs"][0], r2_score
# #     )
# #
# #     ax.plot(line_x, line_y, 'r', linewidth=3)
# #     return reg_txt
#
#
# # # ── load both lead times ─────────────────────────────────────────────────────
# # data_all = {}
# # for lt in lead_times:
# #     with open(save_tmpl % lt, 'rb') as file:
# #         data_all[lt] = pickle.load(file)
# #     print(lt, len(data_all[lt]["obs"]))
# #
# # fig, axes = plt.subplots(1, len(lead_times), figsize=(13, 6))
# # lim = [0, 300]
# #
# # for i, (ax, lt) in enumerate(zip(axes, lead_times)):
# #     X = data_all[lt]["obs"]
# #     Y = data_all[lt]["model"]
# #
# #     reg_txt = plotComparisonIntercept_fast(X, Y, bin_size=1000, ax=ax, fig=fig)
# #     ax.plot(lim, lim, "k--", linewidth=3)
# #     ax.set_xlim(lim)
# #     ax.set_ylim(lim)
# #     ax.set_xlabel(r"Observed $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
# #     if i == 0:
# #         ax.set_ylabel(r"UFS-AQM $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
# #     ax.set_title(f"{lt:02d}z Initialization\n{reg_txt}", fontsize=16, fontweight='bold')
# #     ax.xaxis.set_major_locator(MultipleLocator(50))
# #     ax.yaxis.set_major_locator(MultipleLocator(50))
# #     ax.xaxis.set_minor_locator(MultipleLocator(10))
# #     ax.yaxis.set_minor_locator(MultipleLocator(10))
# #     ax.tick_params(axis='both', labelsize=16)
# #     ax.set_aspect('equal')
# #
# # plt.tight_layout()
# # plt.show()