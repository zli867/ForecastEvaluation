# import pickle
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator, FuncFormatter
#
# lead_times = [6, 12]
# save_tmpl = ("/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/metrics_bylead_init_%d.pkl")
# colors = {"All": 'k', "Smoke": 'tab:red', "Non-smoke": 'tab:blue'}
#
# results_all = {}
# for lt in lead_times:
#     with open(save_tmpl % lt, "rb") as f:
#         results_all[lt] = pickle.load(f)
#
# # desired metric order + axis labels (only keep those present in the pickle)
# order  = ["MB", "RMSE", "NMSE", "FB", "FE", "r", "COD"]
# labels = {"MB": r"MB ($\mu g\,m^{-3}$)", "RMSE": r"RMSE ($\mu g\,m^{-3}$)",
#           "NMSE": "NMSE", "FB": "FB (%)", "FE": "FE (%)", "r": "r", "COD": "COD"}
# present = list(results_all[lead_times[0]].keys())
# metrics = [m for m in order if m in present]
#
# nrows, ncols = len(metrics), len(lead_times)
# fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows),
#                          sharex=True, sharey="row")
#
# for r, metric in enumerate(metrics):
#     for c, lt in enumerate(lead_times):
#         ax  = axes[r, c]
#         res = results_all[lt][metric]
#         for name in res:
#             ax.plot(res[name]["index"], res[name]["values"],
#                     marker="o", markersize=6, markeredgecolor="k", markeredgewidth=0.5,
#                     linewidth=2.0, label=name, color=colors[name])
#
#         ax.tick_params(axis="both", which="major", labelsize=12, length=7, width=1.6)
#         ax.tick_params(axis="both", which="minor", length=4, width=1.6)
#         ax.xaxis.set_major_locator(MultipleLocator(4))
#         ax.xaxis.set_minor_locator(MultipleLocator(1))
#
#         if c == 0:
#             ax.set_ylabel(labels[metric], fontsize=14)
#         if r == 0:
#             ax.set_title(f"{lt:02d}z Initialization", fontsize=15, fontweight="bold", pad=28)
#         if r == nrows - 1:
#             ax.set_xlabel("Lead Time (hr)", fontsize=14)
#
#         # secondary UTC axis on the top row only (offset differs per column)
#         if r == 0:
#             secax = ax.secondary_xaxis(
#                 "top",
#                 functions=(lambda x, off=lt: x + off,
#                            lambda x, off=lt: x - off))
#             secax.set_xlabel("UTC Time (hr)", fontsize=13)
#             secax.xaxis.set_major_locator(MultipleLocator(4))
#             secax.xaxis.set_major_formatter(
#                 FuncFormatter(lambda val, pos: f"{int(round(val)) % 24:02d}"))
#             secax.tick_params(axis="x", which="major", labelsize=12, length=7, width=1.6)
#
# # single legend (from the first panel) at the top
# handles, lbls = axes[0, 0].get_legend_handles_labels()
# fig.legend(handles, lbls, frameon=False, fontsize=13, ncol=3,
#            loc="upper center", bbox_to_anchor=(0.5, 1.0))
#
# fig.tight_layout(rect=[0, 0, 1, 0.97])
# plt.show()

# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
#
# lead_times = [6, 12]
# save_tmpl = ("/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/metrics_bylead_init_%d.pkl")
# init_colors = {6: "tab:blue", 12: "tab:red"}
# subsets = ["All", "Smoke", "Non-smoke"]
#
# results_all = {}
# for lt in lead_times:
#     with open(save_tmpl % lt, "rb") as f:
#         results_all[lt] = pickle.load(f)
#
# order  = ["MB", "RMSE", "NMSE", "FB", "FE", "r", "COD"]
# labels = {"MB": r"MB ($\mu g\,m^{-3}$)", "RMSE": r"RMSE ($\mu g\,m^{-3}$)",
#           "NMSE": "NMSE", "FB": "FB (%)", "FE": "FE (%)", "r": "r", "COD": "COD"}
# present = list(results_all[lead_times[0]].keys())
# metrics = [m for m in order if m in present]
#
# nrows, ncols = len(metrics), len(subsets)
# fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.7 * nrows),
#                          sharex=True, sharey="row")
#
# for r, metric in enumerate(metrics):
#     for c, sub in enumerate(subsets):
#         ax = axes[r, c]
#         for lt in lead_times:
#             res = results_all[lt][metric][sub]
#             idx = np.asarray(res["index"], dtype=float)      # lead time 1..24
#             val = np.asarray(res["values"], dtype=float)
#             utc = (idx + lt) % 24                             # convert to UTC hr
#             o   = np.argsort(utc)
#             ax.plot(utc[o], val[o], marker="o", markersize=5,
#                     markeredgecolor="k", markeredgewidth=0.4, linewidth=2.0,
#                     color=init_colors[lt], label=f"{lt:02d}z init")
#
#         ax.set_xlim(-0.5, 23.5)
#         ax.xaxis.set_major_locator(MultipleLocator(6))
#         ax.xaxis.set_minor_locator(MultipleLocator(1))
#         ax.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.4)
#         ax.tick_params(axis="both", which="minor", length=3, width=1.4)
#
#         if c == 0:
#             ax.set_ylabel(labels[metric], fontsize=13)
#         if r == 0:
#             ax.set_title(sub, fontsize=14, fontweight="bold")
#         if r == nrows - 1:
#             ax.set_xlabel("UTC Time (hr)", fontsize=13)
#
# # single legend
# handles, lbls = axes[0, 0].get_legend_handles_labels()
# fig.legend(handles, lbls, frameon=False, fontsize=12, ncol=2,
#            loc="upper center", bbox_to_anchor=(0.5, 1.0))
#
# fig.suptitle("Metrics by UTC time: 06z vs 12z initialization",
#              fontsize=15, fontweight="bold", y=1.005)
#
# fig.tight_layout(rect=[0, 0, 1, 0.97])
# plt.show()


import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

lead_times = [6, 12]
save_tmpl = ("/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/metrics_bylead_init_%d.pkl")
colors = {"All": 'k', "Smoke": 'red', "Non-smoke": 'blue'}

results_all = {}
for lt in lead_times:
    with open(save_tmpl % lt, "rb") as f:
        results_all[lt] = pickle.load(f)

order  = ["MB", "RMSE", "NMSE", "FB", "FE", "r", "COD"]
labels = {"MB": r"MB ($\mu g\,m^{-3}$)", "RMSE": r"RMSE ($\mu g\,m^{-3}$)",
          "NMSE": "NMSE", "FB": "FB (%)", "FE": "FE (%)", "r": "r", "COD": "COD"}
present = list(results_all[lead_times[0]].keys())
metrics = [m for m in order if m in present]

# top-axis formatter factory: UTC value -> lead time for a given init
def lead_formatter(off):
    def fmt(val, pos):
        lead = int(round(val) - off) % 24
        return f"{24 if lead == 0 else lead:02d}"
    return fmt

nrows, ncols = len(metrics), len(lead_times)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows),
                         sharex=True, sharey="row")

for r, metric in enumerate(metrics):
    for c, lt in enumerate(lead_times):
        ax  = axes[r, c]
        res = results_all[lt][metric]
        for name in res:
            idx = np.asarray(res[name]["index"], dtype=float)     # lead time 1..24
            val = np.asarray(res[name]["values"], dtype=float)
            utc = (idx + lt) % 24                                  # convert to UTC hr
            o   = np.argsort(utc)
            if metric == "NMSE" and name == "All":
                print(name)
                print(lt)
                idx = np.where(utc[o] == 7)[0][0]
                print(utc[o][idx])
                print(val[o][idx])
            ax.plot(utc[o], val[o],
                    marker="o", markersize=6, markeredgecolor="k", markeredgewidth=0.5,
                    linewidth=2.0, label=name, color=colors[name])

        ax.set_xlim(-0.5, 23.5)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis="both", which="major", labelsize=12, length=7, width=1.6)
        ax.tick_params(axis="both", which="minor", length=4, width=1.6)

        if c == 0:
            ax.set_ylabel(labels[metric], fontsize=14)
        if r == 0:
            ax.set_title(f"{lt:02d}z Initialization", fontsize=15, fontweight="bold", pad=28)
        if r == nrows - 1:
            ax.set_xlabel("UTC Time (hr)", fontsize=14)

        # top axis: lead time (different per column)
        if r == 0:
            secax = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
            secax.set_xlabel("Lead Time (hr)", fontsize=13)
            secax.xaxis.set_major_locator(MultipleLocator(4))
            secax.xaxis.set_major_formatter(FuncFormatter(lead_formatter(lt)))
            secax.tick_params(axis="x", which="major", labelsize=12, length=7, width=1.6)

# single legend
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, frameon=False, fontsize=13, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 1.0))

fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()