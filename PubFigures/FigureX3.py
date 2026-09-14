import os, pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# ── config / paths ───────────────────────────────────────────────────────────
initial_time = 6
UTC_OFFSET   = 6
MODEL_VAR    = "PM25_UFSAQM"
OBS_VAR      = "PM25"
start_time, end_time = datetime(2024, 5, 14), datetime(2024, 12, 31)

BASE          = "/Users/zongrunli/Desktop/ForecastEvaluation/data"
stn_locs_file = os.path.join(BASE, "static/stn_info.pkl")
obs_file      = os.path.join(BASE, "obs/PM25_obs_2021_2024.csv")
stats_folder  = os.path.join(BASE, "eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % initial_time)

# ── load obs ─────────────────────────────────────────────────────────────────
with open(stn_locs_file, "rb") as f:
    site_dict = pickle.load(f)

obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]

# ── load model ───────────────────────────────────────────────────────────────
model_list = []
cur = start_time
while cur <= end_time:
    f = os.path.join(stats_folder, f"site_level_aqm_{cur:%Y%m%d}_%02d.csv" % initial_time)
    if os.path.exists(f):
        model_list.append(pd.read_csv(f, parse_dates=["Time"], dtype={"siteCode": str}))
    cur += timedelta(days=1)
model_df = pd.concat(model_list, ignore_index=True)

merged = model_df.merge(obs_df[["siteCode", "Time", OBS_VAR]],
                        on=["siteCode", "Time"], how="inner")

# ── smoke flag ───────────────────────────────────────────────────────────────
smoke_day_sets = {c: {d.date() for d in site_dict[c].get("smoke_dates", [])} for c in site_dict}
merged["isSmokeDay"] = [
    code in smoke_day_sets and t.date() in smoke_day_sets[code]
    for code, t in zip(merged["siteCode"], merged["Time"])
]

# keep only finite pairs, compute per-row bias
merged = merged[np.isfinite(merged[MODEL_VAR]) & np.isfinite(merged[OBS_VAR])]
merged["diff"] = merged[MODEL_VAR] - merged[OBS_VAR]

subsets = {"Smoke":     merged[merged["isSmokeDay"]],
           "Non-smoke": merged[~merged["isSmokeDay"]]}

# ── per-lead bias arrays + means (ordered by lead, placed at 1..24) ──────────
def lead_box_data(sub):
    leads = sorted(sub["lead_time"].unique())
    data  = [sub.loc[sub["lead_time"] == L, "diff"].values for L in leads]
    means = [d.mean() for d in data]
    pos   = np.arange(1, len(leads) + 1)        # force x positions 1..24
    return pos, data, means

# ── plot: per-lead box plots + mean-bias line ────────────────────────────────
print("Plotting...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
c = "blue"

for ax, name in zip(axes, ["Smoke", "Non-smoke"]):
    pos, data, means = lead_box_data(subsets[name])
    bp = ax.boxplot(data, positions=pos, widths=0.6,
                    whis=(10, 90), showfliers=False, patch_artist=True,
                    medianprops=dict(color="k", linewidth=2),
                    whiskerprops=dict(color=c, linewidth=1.5),
                    capprops=dict(color=c, linewidth=1.5),
                    boxprops=dict(edgecolor=c, linewidth=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor(c); patch.set_alpha(0.45)

    # mean-bias line on top
    ax.plot(pos, means, "-o", color="g", lw=2.5, ms=6,
            markerfacecolor="white", markeredgecolor="k", zorder=6, label="Mean bias")

    ax.axhline(0, color="red", lw=2, zorder=5)            # perfect agreement
    ax.set_ylabel(r"UFS-AQM $-$ Obs $PM_{2.5}$ ($\mu g/m^3$)", fontsize=15)
    if name == "Smoke":
        ax.set_title(f"AQMv7 {name} Days\n(%s to %s)" % (start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d")), fontsize=15, fontweight="bold")
    else:
        ax.set_title(f"{name} Days", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=13, length=8, width=2)

axes[1].set_xlabel("Lead Time (hr)", fontsize=15)
axes[0].set_xlim(0.5, 24.5)
axes[1].set_xticks(range(1, 25))
axes[1].set_xticklabels(range(1, 25))

# secondary UTC axis on the top panel
secax = axes[0].secondary_xaxis("top",
    functions=(lambda x: x + UTC_OFFSET, lambda x: x - UTC_OFFSET))
secax.set_xlabel("UTC Time (hr)", fontsize=15)
secax.xaxis.set_major_locator(MultipleLocator(1))
secax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v)) % 24:02d}"))
secax.tick_params(axis="x", which="major", labelsize=13, length=8, width=2)
axes[0].set_ylim(-16, 11)
axes[1].set_ylim(-7, 8)

fig.tight_layout()
plt.show()