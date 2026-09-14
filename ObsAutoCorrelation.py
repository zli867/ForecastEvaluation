import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ── config ───────────────────────────────────────────────────────────────────
obs_file  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
OBS_VAR   = "PM25"
start, end = datetime(2021, 1, 1), datetime(2024, 12, 31)
max_shift = 24
min_sites = 20
save_dir = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data"
r_csv     = save_dir + "/"+ "spatial_autocorr_pearson.csv"
cod_csv   = save_dir + "/" + "spatial_autocorr_cod.csv"

# ── load + reshape to a regular hourly grid (rows=Time, cols=site) ───────────
df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
df = df[(df["Time"] >= start) & (df["Time"] <= end)]

pivot = df.pivot_table(index="Time", columns="siteCode", values=OBS_VAR)
full_idx = pd.date_range(start, end, freq="h")
vals = pivot.reindex(full_idx).values.astype("float32")   # (T, S)
print(f"hours={vals.shape[0]}, sites={vals.shape[1]}")

# ── per-row (per-hour) statistics across sites ───────────────────────────────
def rowwise_corr(A, B, min_n):
    """Pearson r between row t (pred, time t) and row t+k (truth), across sites."""
    both = np.isfinite(A) & np.isfinite(B)
    A = np.where(both, A, np.nan); B = np.where(both, B, np.nan)
    n = both.sum(axis=1)
    mA = np.nanmean(A, axis=1); mB = np.nanmean(B, axis=1)
    dA = A - mA[:, None];       dB = B - mB[:, None]
    num = np.nansum(dA * dB, axis=1)
    den = np.sqrt(np.nansum(dA**2, axis=1) * np.nansum(dB**2, axis=1))
    return np.where((den > 0) & (n >= min_n), num / den, np.nan)

def rowwise_cod(A, B, min_n):
    """COD = 1 - SS_res/SS_tot, treating field at t (A) as the persistence
       prediction of the field at t+k (B), across sites."""
    both = np.isfinite(A) & np.isfinite(B)
    A = np.where(both, A, np.nan); B = np.where(both, B, np.nan)
    n = both.sum(axis=1)
    ss_res = np.nansum((A - B)**2, axis=1)
    dB = B - np.nanmean(B, axis=1)[:, None]
    ss_tot = np.nansum(dB**2, axis=1)
    return np.where((ss_tot > 0) & (n >= min_n), 1 - ss_res / ss_tot, np.nan)

res_r, res_cod = {}, {}
for k in range(1, max_shift + 1):
    A = vals[:-k]; B = vals[k:]
    res_r[k]   = rowwise_corr(A, B, min_sites)
    res_cod[k] = rowwise_cod(A, B, min_sites)

pd.DataFrame({str(k): pd.Series(res_r[k])   for k in range(1, max_shift + 1)}).to_csv(r_csv, index=False)
pd.DataFrame({str(k): pd.Series(res_cod[k]) for k in range(1, max_shift + 1)}).to_csv(cod_csv, index=False)
print(f"saved -> {r_csv}\nsaved -> {cod_csv}")

# ── plotting helper ──────────────────────────────────────────────────────────
shifts = list(range(1, max_shift + 1))

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
    ax.plot(shifts, mean_v, color="red", linewidth=3, label="Mean (over time)")
    ax.set_xlabel("Lag (hours)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    return im

# bins: r is bounded [-1, 1]; COD set adaptively (can be strongly negative)
r_bins = np.linspace(-1, 1, 120)
all_cod = np.concatenate([res_cod[k][np.isfinite(res_cod[k])] for k in shifts])
cod_lo  = np.floor(np.nanpercentile(all_cod, 1))
cod_bins = np.linspace(cod_lo, 1, 120)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))
imL = density_panel(axL, res_r,   r_bins,   "Pearson $r$", "Spatial $r$ persistence")
imR = density_panel(axR, res_cod, cod_bins, "COD",         "Spatial COD persistence")
fig.colorbar(imL, ax=axL, label="Probability Density")
fig.colorbar(imR, ax=axR, label="Probability Density")

fig.suptitle("Spatial PM$_{2.5}$ persistence: all-site field at t vs t+lag (AQS, 2021–2024)",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()