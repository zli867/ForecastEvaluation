import pickle, gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.linear_model import LinearRegression
from utils.metrics import MB, RMSE, NMSE, FB, FE, pearson_r, COD

PUB = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
inits   = [6, 12]
subsets = ["All", "Smoke", "Non-smoke"]
AXMAX, NBINS = 300, 150            # 0-300 ug/m3, 2 ug/m3 cells

# ── precompute density + regression (one init pickle in memory at a time) ────
precomp = {}
for it in inits:
    with open(f"{PUB}/scatter_data_%02d.pkl" % it, "rb") as f:
        data = pickle.load(f)
    for sub in subsets:
        X = np.asarray(data[sub]["obs"],   dtype=float)
        Y = np.asarray(data[sub]["model"], dtype=float)
        m = np.isfinite(X) & np.isfinite(Y)
        X, Y = X[m], Y[m]
        # density on fixed 0-AXMAX grid, normalized by N (matches plotComparisonIntercept_fast)
        hh, xe, ye = np.histogram2d(X, Y, bins=NBINS, range=[[0, AXMAX], [0, AXMAX]])
        hh = hh.T / len(X)
        # regression + Pearson r on full (unclipped) data
        lin = LinearRegression().fit(X.reshape(-1, 1), Y)
        r = float(pearson_r(Y, X))                     # model, obs
        precomp[(it, sub)] = {
            "hh": hh, "xe": xe, "ye": ye,
            "slope": float(lin.coef_[0]), "intercept": float(lin.intercept_),
            "r": r, "n": int(len(X)),
        }
        del X, Y, m, lin; gc.collect()
    del data; gc.collect()
    print(f"init {it:02d} done", flush=True)

# shared log-norm across all panels
pos = np.concatenate([p["hh"][p["hh"] > 0].ravel() for p in precomp.values()])
norm = colors.LogNorm(vmin=pos.min(), vmax=pos.max())

# ── plot 2x3 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4), sharex=True, sharey=True)
im = None
for r_idx, it in enumerate(inits):
    for c, sub in enumerate(subsets):
        ax = axes[r_idx, c]
        p = precomp[(it, sub)]
        im = ax.pcolormesh(p["xe"], p["ye"], p["hh"], cmap="jet", norm=norm)
        ax.plot([0, AXMAX], [0, AXMAX], "k--", lw=1.6)                 # 1:1
        lx = np.linspace(0, AXMAX, 100)
        ax.plot(lx, p["slope"] * lx + p["intercept"], "r", lw=2)       # regression
        sign = "+" if p["intercept"] >= 0 else "-"
        ax.set_title(f"y={p['slope']:.2f}x{sign}{abs(p['intercept']):.2f}, "
                     f"$r$={p['r']:.2f}", fontsize=11)
        ax.set_xlim(0, AXMAX); ax.set_ylim(0, AXMAX)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)
        if r_idx == 0:   # column header
            ax.annotate(sub, xy=(0.5, 1.18), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=14, fontweight="bold")
        if c == 0:   # row label
            ax.set_ylabel(f"{it:02d}z init\nUFS-AQM PM$_{{2.5}}$ ($\\mu g\\,m^{{-3}}$)",
                          fontsize=11)
        if r_idx == 1:
            ax.set_xlabel("Observed PM$_{2.5}$ ($\\mu g\\,m^{-3}$)", fontsize=11)

fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.08, wspace=0.12, hspace=0.18)
cax = fig.add_axes([0.915, 0.08, 0.018, 0.82])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Density", fontsize=11, rotation=270, labelpad=15)
plt.show()