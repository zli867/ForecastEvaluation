import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Bland–Altman density helper: (pred-obs) vs (obs+pred)/2 ──────────────────
def plotBlandAltman_fast(obs, pred, bin_size, ax, fig, side=True,
                         xlim=(1, 300), ylim=(-150, 150), cmap="jet", show_loa=True):
    obs  = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    mean = (obs + pred) / 2.0          # x
    diff = pred - obs                  # y  (model minus obs)
    m = np.isfinite(mean) & np.isfinite(diff)
    mean, diff = mean[m], diff[m]

    # log-spaced x edges, linear y edges
    xedges = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), bin_size + 1)
    yedges = np.linspace(ylim[0], ylim[1], bin_size + 1)

    # keep only points within the (positive) x range for log binning
    inx = (mean >= xlim[0]) & (mean <= xlim[1])
    mean_b, diff_b = mean[inx], diff[inx]

    hh, xe, ye = np.histogram2d(mean_b, diff_b, bins=[xedges, yedges])
    hh = hh.T / len(mean)              # density (normalised by N, all finite points)
    im = ax.pcolormesh(xe, ye, hh, cmap=cmap,
                       norm=colors.LogNorm(vmin=hh[hh > 0].min(), vmax=hh.max()))
    if side:
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Density", fontsize=10, rotation=270, labelpad=12)

    # red reference line at y = 0  (perfect agreement)
    ax.axhline(0, color="red", linewidth=3, zorder=5)

    # Bland–Altman stats: mean bias and ±1.96·SD limits of agreement
    bias, sd = diff.mean(), diff.std()
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    if show_loa:
        ax.axhline(bias, color="k", linewidth=2)
        ax.axhline(hi, color="k", linestyle="--", linewidth=1.5)
        ax.axhline(lo, color="k", linestyle="--", linewidth=1.5)

    # regression of difference on log10(mean): proportional-bias trend
    lx = np.log10(mean_b)
    a, b = np.polyfit(lx, diff_b, 1)
    r = np.corrcoef(lx, diff_b)[0, 1]
    xfit = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
    yfit = a * np.log10(xfit) + b
    ax.plot(xfit, yfit, color="blue", linewidth=2.5, zorder=6)
    sign = "+" if b >= 0 else "−"
    eq = rf"$y = {a:.2f}\,\log_{{10}}(x) {b:+.2f}$"
    ax.text(0.03, 0.04, eq, transform=ax.transAxes, fontsize=12,
            va="bottom", ha="left", color="k",
            bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85), zorder=7)

    return f"bias={bias:.2f}, LoA=[{lo:.1f}, {hi:.1f}]"


# ── config / paths ───────────────────────────────────────────────────────────
initial_time = 6
PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
with open(os.path.join(PUB, "scatter_data_%02d.pkl" % initial_time), "rb") as f:
    scat = pickle.load(f)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
XLIM, YLIM = (1, 300), (-150, 150)      # x lower bound must be > 0 for log scale
idx = 0
panels = [("All", "All Data"), ("Smoke", "Smoke Days"), ("Non-smoke", "Non-smoke Days")]
for ax, (key, title) in zip(axes, panels):
    X, Y = scat[key]["obs"], scat[key]["model"]
    txt = plotBlandAltman_fast(X, Y, bin_size=1000, ax=ax, fig=fig,
                               xlim=XLIM, ylim=YLIM)
    ax.set_xscale("log")
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    if idx == 1:
        ax.set_xlabel(r"Mean of UFS-AQM & Obs $PM_{2.5}$ ($\mu g/m^3$)", fontsize=16)
    if idx == 0:
        ax.set_ylabel(r"UFS-AQM $-$ Obs $PM_{2.5}$ ($\mu g/m^3$)", fontsize=16)
    ax.set_title(f"{title}\n{txt}", fontsize=15, fontweight="bold")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis="both", labelsize=14)
    idx += 1
fig.tight_layout()
plt.show()