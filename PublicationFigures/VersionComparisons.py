import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

lead_time = 6
utc_offset = lead_time   # lead-time hr + offset = UTC hr (change if init differs)
versions = ["v5", "v6", "v7"]
save_tmpl = ("/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/"
             "site_r2_smokenonsmoke_subsets_%02d_%s.pkl")

# ── Load all versions ───────────────────────────────────────────────────────
results_all = {}
for ver in versions:
    with open(save_tmpl % (lead_time, ver), "rb") as f:
        results_all[ver] = pickle.load(f)

metrics = list(results_all[versions[0]].keys())
subset_colors = {"All": "tab:blue", "Smoke": "tab:red", "Non-smoke": "tab:green"}
version_styles = {"v5": ":", "v6": "--", "v7": "-"}
colors = {"v5": "r", "v6": "g", "v7": "k"}
# ── One figure per metric, all versions overlaid ────────────────────────────
for metric in metrics:
    metric_label = "$R^2$" if metric == "R2" else metric
    fig, ax = plt.subplots(figsize=(8, 5))

    for ver in versions:
        for name in results_all[ver][metric]:
            if name != "All":
                continue
            res = results_all[ver][metric][name]
            ax.plot(res["index"], res["values"],
                    marker="o", markersize=6, markeredgecolor="k",
                    markeredgewidth=0.6, linewidth=2.5,
                    color=colors.get(name),
                    linestyle=version_styles[ver],
                    label=f"{name} {ver} (n={res['length']})")

    ax.set_xlabel("Lead Time (hr)", fontsize=16)
    ax.set_ylabel(f"{metric_label} Performance", fontsize=16)
    ax.set_title(f"{metric_label} by Lead Time: Smoke vs Non-Smoke Start",
                 fontsize=16, fontweight="bold")

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", which="major", labelsize=16, length=8, width=2)
    ax.tick_params(axis="both", which="minor", length=4, width=2)
    ax.legend(frameon=False, fontsize=10, ncol=len(versions))

    # ── secondary x-axis: UTC time (lead time + offset, wrapped to 0–23) ──
    secax = ax.secondary_xaxis("top",
                               functions=(lambda x: x + utc_offset,
                                          lambda x: x - utc_offset))
    secax.set_xlabel("UTC Time (hr)", fontsize=16)
    secax.xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: f"{int(round(val)) % 24:02d}"))
    secax.xaxis.set_minor_locator(MultipleLocator(1))
    secax.tick_params(axis="x", which="major", labelsize=16, length=8, width=2)
    secax.tick_params(axis="x", which="minor", length=4, width=2)

    plt.tight_layout()
    plt.show()