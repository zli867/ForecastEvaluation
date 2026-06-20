# ── PART 1: generate temporary obs/model pair data for the scatter plot ─────
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# MODEL_VAR = "PM25_UFSAQM"
# OBS_VAR   = "PM25"
# lead_time = 6
# versions = {
#     5: [datetime(2021, 1, 1), datetime(2021, 7, 19)],
#     6: [datetime(2021, 7, 20), datetime(2024, 5, 13)],
#     7: [datetime(2024, 5, 14), datetime(2024, 12, 31)],
# }
# for ver in versions:
#     start_time, end_time = versions[ver][0], versions[ver][1]
#     obs_file     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
#     stats_folder = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % lead_time
#     save_path    = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/scatter_data_%02d_v%d.pkl" % (lead_time, ver)
#
#     # ── observations ─────────────────────────────────────────────────────────────
#     obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
#     obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]
#
#     # ── model files ──────────────────────────────────────────────────────────────
#     model_list = []
#     cur_time = start_time
#     while cur_time <= end_time:
#         date_str = cur_time.strftime("%Y%m%d")
#         cur_model_file = os.path.join(stats_folder, f"site_level_aqm_{date_str}_%02d.csv" % lead_time)
#         if os.path.exists(cur_model_file):
#             model_list.append(pd.read_csv(cur_model_file, parse_dates=["Time"], dtype={"siteCode": str}))
#         cur_time += timedelta(days=1)
#     model_df = pd.concat(model_list, ignore_index=True)
#
#     # ── merge & clean ────────────────────────────────────────────────────────────
#     merged = model_df.merge(
#         obs_df[["siteCode", "Time", OBS_VAR]],
#         on=["siteCode", "Time"], how="inner"
#     )
#     merged = merged.dropna(subset=[MODEL_VAR, OBS_VAR])
#     merged = merged[(merged[OBS_VAR] >= 0) & (merged[MODEL_VAR] >= 0)]
#
#     data = {
#         "obs":   merged[OBS_VAR].to_numpy(float),
#         "model": merged[MODEL_VAR].to_numpy(float),
#     }
#
#     with open(save_path, "wb") as f:
#         pickle.dump(data, f)
#     print(f"Saved {len(data['obs'])} obs/model pairs to {save_path}")



# ── PART 2: hexbin density scatter of model vs obs, v5/v6/v7 side by side ───
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from sklearn.linear_model import LinearRegression
from utils.Visualizations import plotComparisonIntercept, plotComparisonIntercept_fast

lead_time = 6
versions  = [5, 6, 7]
version_labels = {           # shown in panel titles
    5: "v5 (2021-01-01 to 2021-07-19)",
    6: "v6 (2021-07-20 to 2024-05-13)",
    7: "v7 (2024-05-14 to 2024-12-31)",
}
save_tmpl = ("/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/"
             "scatter_data_%02d_v%d.pkl")



# ── load all versions ────────────────────────────────────────────────────────
data_all = {}
for ver in versions:
    with open(save_tmpl % (lead_time, ver), 'rb') as file:
        data_all[ver] = pickle.load(file)
    print(f"v{ver}: {len(data_all[ver]['obs'])} pairs")

fig, axes = plt.subplots(1, len(versions), figsize=(19, 6))
lim = [0, 300]

for i, (ax, ver) in enumerate(zip(axes, versions)):
    X = data_all[ver]["obs"]
    Y = data_all[ver]["model"]
    n = len(X)
    if i == 0 or i == 1:
        reg_txt = plotComparisonIntercept_fast(X, Y, bin_size=1000, ax=ax, fig=fig, side=False)
    else:
        reg_txt = plotComparisonIntercept_fast(X, Y, bin_size=1000, ax=ax, fig=fig, side=True)
    ax.plot(lim, lim, "k--", linewidth=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    if i == 1:
        ax.set_xlabel(r"Observed $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
    if i == 0:
        ax.set_ylabel(r"UFS-AQM $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
    ax.set_title(f"{version_labels[ver]}\n{reg_txt}", fontsize=14, fontweight='bold')

    # number of points, upper-left corner
    ax.text(0.03, 0.97, f"n = {n:,}", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')

    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='both', labelsize=16)
    ax.set_aspect('equal')

fig.suptitle(f"{lead_time:02d}z Initialization", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
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