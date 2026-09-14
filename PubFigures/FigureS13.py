import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# ── config ───────────────────────────────────────────────────────────────────
save_dir = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
r_csv     = save_dir + "/"+ "spatial_autocorr_pearson.csv"
cod_csv   = save_dir + "/" + "spatial_autocorr_cod.csv"
max_shift = 24

# ── load saved results (columns = lag "1".."24", rows = hours) ───────────────
r_df   = pd.read_csv(r_csv)
cod_df = pd.read_csv(cod_csv)

shifts = list(range(1, max_shift + 1))
res_r   = {k: r_df[str(k)].dropna().values   for k in shifts}
res_cod = {k: cod_df[str(k)].dropna().values for k in shifts}

# ── plotting helper ──────────────────────────────────────────────────────────
def density_panel(ax, res, bins, ylabel, title):
    mean_v = [np.nanmean(res[k]) for k in shifts]
    hist2d = []
    for k in shifts:
        col = res[k][np.isfinite(res[k])]
        h, _ = np.histogram(col, bins=bins, density=True)
        hist2d.append(h)
    hist2d = np.array(hist2d).T
    im = ax.imshow(hist2d, aspect="auto", origin="lower",
                   extent=[shifts[0], shifts[-1], bins[0], bins[-1]])
    ax.plot(shifts, mean_v, color="red", linewidth=3, label="Mean")
    ax.set_ylim(bins[0], bins[-1])
    ax.set_xlabel("Lag (hours)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=15, length=9, width=2)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", which="minor", length=5, width=1.5)
    return im

# fixed y-ranges: r in [0, 1], COD in [-1, 1]
r_bins   = np.linspace(0, 1, 120)
cod_bins = np.linspace(-1, 1, 120)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))
imL = density_panel(axL, res_r,   r_bins,   "Pearson $r$", "$r$ Persistence")
imR = density_panel(axR, res_cod, cod_bins, "COD",         "COD Persistence")
fig.colorbar(imL, ax=axL, label="Probability Density")
fig.colorbar(imR, ax=axR, label="Probability Density")
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()