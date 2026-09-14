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
        reg_txt = plotComparisonIntercept_fast(X, Y, bin_size=int(n/5000), ax=ax, fig=fig, side=False)
    else:
        reg_txt = plotComparisonIntercept_fast(X, Y, bin_size=int(n/2500), ax=ax, fig=fig, side=True)
    ax.plot(lim, lim, "k--", linewidth=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    if i == 1:
        ax.set_xlabel(r"Observed $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
    if i == 0:
        ax.set_ylabel(r"UFS-AQM $PM_{2.5}$ ($\mu g / m^3$)", fontsize=16)
    ax.set_title(f"{version_labels[ver]}\n{reg_txt}", fontsize=14, fontweight='bold')

    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='both', labelsize=16)
    ax.set_aspect('equal')
plt.tight_layout()
plt.show()