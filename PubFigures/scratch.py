import pickle
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Load processed event data
# ---------------------------------------------------------------------
with open(
    "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/"
    "plot_data_by_year_new.pkl",
    "rb",
) as f:
    plot_data = pickle.load(f)

years = sorted(plot_data.keys())

fig, axes = plt.subplots(
    len(years),
    1,
    figsize=(12, 3.2 * len(years)),
    sharex=False,
)

if len(years) == 1:
    axes = [axes]

for ax, year in zip(axes, years):
    d = plot_data[year]
    merged = d["merged"].sort_values("Time")

    # Regional mean concentration across paired monitor locations
    ax.plot(
        merged["Time"],
        merged["PM25"],
        color="black",
        linewidth=1.2,
        label=r"Observed $PM_{2.5}$",
        zorder=3,
    )

    ax.plot(
        merged["Time"],
        merged["PM25_UFSAQM_stn_avg"],
        color="red",
        linewidth=1.2,
        label=r"UFS-AQM $PM_{2.5}$ (monitor average)",
        zorder=3,
    )

    # Spread across the monitor locations used in the paired evaluation
    ax.fill_between(
        merged["Time"],
        merged["obs_p25"],
        merged["obs_p75"],
        color="gray",
        alpha=0.35,
        label=r"Observed $PM_{2.5}$ P25–P75 range",
        zorder=1,
    )

    ax.fill_between(
        merged["Time"],
        merged["model_p25"],
        merged["model_p75"],
        color="purple",
        alpha=0.25,
        label=r"UFS-AQM $PM_{2.5}$ P25–P75 range (monitor locations)",
        zorder=2,
    )

    # Display "r" without changing the dictionary stored in plot_data.
    metric_labels = {
        "MB": "MB",
        "RMSE": "RMSE",
        "corr": "r",
        "r": "r",
    }

    metrics_label = "\n".join(
        f"{metric_labels.get(key, key)}: {value:.2f}"
        for key, value in d["metrics"].items()
    )

    ax.text(
        0.01,
        0.97,
        metrics_label,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
        zorder=4,
    )

    region_name = ", ".join(str(region) for region in d["region_numbers"])

    ax.set_title(f"{year} — EPA Region {region_name}")
    ax.set_ylabel(r"$PM_{2.5}$ ($\mu g\,m^{-3}$)")
    ax.grid(True, linestyle="--", alpha=0.5)

axes[-1].set_xlabel("UTC Time")

# Show one shared legend.
axes[0].legend(
    loc="upper right",
    fontsize=8,
    frameon=True,
)

fig.tight_layout()
plt.show()

# Optional:
# fig.savefig("Figure_SXX_monitor_colocated_ranges.png",
#             dpi=300, bbox_inches="tight")