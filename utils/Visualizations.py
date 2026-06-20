import numpy as np
from matplotlib.ticker import Locator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.linear_model import LinearRegression
from utils.utils import AQI_CATEGORIES, AQI_ORDER, assign_aqi_category, aqi_category_color
from utils.metrics import metrics_by_aqi_category, aqi_category_agreement, remove_nan_values, MB, ME, RMSE, NMB, NME, IOA, pearson_r, R2
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

def discrete_cmap(invertals, base_color_scheme="Spectral_r", low_value_color='gray', high_value_color='purple'):
    cmap = plt.get_cmap(base_color_scheme, invertals)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = colors.to_rgba("white")
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    cmap.set_over(color=high_value_color, alpha=1.0)
    # cmap.set_under(color=low_value_color, alpha=1.0)
    return cmap


def discrete_pos_neg_cmap(invertals, base_color_scheme="seismic", neutral_color='white', low_value_color='black', high_value_color='purple'):
    invertals = (invertals // 2) * 2 + 1
    cmap = plt.get_cmap(base_color_scheme, invertals)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    cmaplist[invertals // 2] = colors.to_rgba(neutral_color)
    cmap.set_over(color=high_value_color, alpha=1.0)
    cmap.set_under(color=low_value_color, alpha=1.0)
    return cmap


def plotPolygons(polygon_list, ax, color, crs=None, linewidth=1.0):
    for current_polygon in polygon_list:
        if current_polygon.geom_type == "MultiPolygon":
            for geom in current_polygon.geoms:
                xs, ys = geom.exterior.xy
                if crs is not None:
                    xs, ys = crs(xs, ys)
                ax.plot(xs, ys, color, linewidth=linewidth)

        else:
            xs, ys = current_polygon.exterior.xy
            if crs is not None:
                xs, ys = crs(xs, ys)
            ax.plot(xs, ys, color, linewidth=linewidth)


def fillPolygons(polygon_list, ax, color, crs=None):
    for current_polygon in polygon_list:
        if current_polygon.geom_type == "MultiPolygon":
            for geom in current_polygon.geoms:
                xs, ys = geom.exterior.xy
                if crs is not None:
                    xs, ys = crs(xs, ys)
                ax.fill(xs, ys, color)
                ax.plot(xs, ys, 'k')
        else:
            xs, ys = current_polygon.exterior.xy
            if crs is not None:
                xs, ys = crs(xs, ys)
            ax.fill(xs, ys, color)
            ax.plot(xs, ys, 'k')


def plotGeopandas(gdf, ax, col_name, cmap, colorbar_name=None, crs=None):
    plotting = gdf.plot(column=col_name, cmap=cmap, ax=ax, vmax=1000, legend=False)
    norm = colors.Normalize(gdf[col_name].min(), vmax=gdf[col_name].max())
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []
        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                          '%s type.' % type(self))



def get_conf_intercept(alpha, lr, X, y):
    """
    Returns (1-alpha) 2-sided confidence intervals
    for sklearn.LinearRegression coefficients
    as a pandas DataFrame
    """
    coefs = np.r_[[lr.intercept_], lr.coef_]
    X_aux = np.zeros((X.shape[0], X.shape[1] + 1))
    X_aux[:, 1:] = X
    X_aux[:, 0] = 1
    dof = -np.diff(X_aux.shape)[0]
    mse = np.sum((y - lr.predict(X)) ** 2) / dof
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    t_val = stats.t.isf(alpha / 2, dof)
    gap = t_val * np.sqrt(mse * var_params)
    return {
        "coeffs": coefs,
        "lower": coefs - gap,
        "upper": coefs + gap
    }

def plotComparisonIntercept(X, Y, bin_size, ax, fig):
    # scatter
    #histogram definition
    N = len(X)
    bins = [bin_size, bin_size] # number of bins
    # histogram the data
    hh, locx, locy = np.histogram2d(X, Y, bins=bins)
    # Sort the points by density, so that the densest points are plotted last
    Z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(X, Y)])
    idx = Z.argsort()
    # normalize z2
    x2, y2, z2 = X[idx], Y[idx], Z[idx] / N
    z2_min = np.min(z2[z2 > 0])
    cax = ax.scatter(x2, y2, c=z2, cmap='jet', marker='.', s=2, norm=colors.LogNorm(vmin=z2_min, vmax=z2.max()))
    divider = make_axes_locatable(ax)
    d_cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(cax, cax=d_cax, extend='both')
    cb.ax.set_title('Density', fontsize=10)
    # reg
    X = X.reshape(-1, 1)
    lin_model = LinearRegression().fit(X, Y)
    line_x = np.linspace(0, np.max(X)).reshape(-1, 1)
    line_y = lin_model.predict(line_x)
    # Confident intervals
    cf = get_conf_intercept(0.05, lin_model, X, Y)
    r2_score = lin_model.score(X, Y)
    reg_txt = 'y={:.2f}x+{:.2f}, $R^2$ = {:.2f}'.format(cf["coeffs"][1], cf["coeffs"][0], r2_score)
    # ax.text(0.8, 0.2, reg_txt, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.plot(line_x, line_y, 'r')
    return reg_txt


# ---------------------------------------------------------------------------
# AQI colour palette helpers
# ---------------------------------------------------------------------------

def aqi_cmap_and_norm():
    """
    Return a ListedColormap and BoundaryNorm spanning all 6 AQI categories.
    Boundaries are the PM2.5 breakpoints from AQI_CATEGORIES.
    """
    boundaries = [AQI_CATEGORIES[c]["pm25"][0] for c in AQI_ORDER] + [np.inf]
    cat_colors  = [AQI_CATEGORIES[c]["color"]  for c in AQI_ORDER]
    cmap = colors.ListedColormap(cat_colors, name="AQI")
    norm = colors.BoundaryNorm(boundaries, ncolors=len(cat_colors))
    return cmap, norm


def aqi_legend_patches(fontsize=10):
    """Return a list of Patch objects for an AQI legend."""
    patches = []
    for cat in AQI_ORDER:
        info = AQI_CATEGORIES[cat]
        patches.append(
            mpatches.Patch(facecolor=info["color"],
                           edgecolor="black", linewidth=0.4,
                           label=info["label"])
        )
    return patches


def plot_aqi_scatter(prediction, observation, ax,
                     title="Model vs Obs (AQI Category)",
                     model_label="Model PM$_{2.5}$ (µg/m³)",
                     obs_label="Obs PM$_{2.5}$ (µg/m³)",
                     alpha=0.35, s=8):
    """
    Scatter plot of model vs obs PM2.5, with points coloured by the
    *observed* AQI category.  Overlays a 1:1 line and OLS fit.
    """
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    pred_clean, obs_clean = remove_nan_values(prediction, observation)

    obs_cats = assign_aqi_category(obs_clean)

    for cat in AQI_ORDER:
        mask = obs_cats == cat
        if mask.sum() == 0:
            continue
        c = AQI_CATEGORIES[cat]["color"]
        ax.scatter(obs_clean[mask], pred_clean[mask],
                   color=c, alpha=alpha, s=s, linewidths=0, label=cat)

    lim_max = max(obs_clean.max(), pred_clean.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="1:1")

    # OLS
    X_fit = obs_clean.reshape(-1, 1)
    lin_model = LinearRegression().fit(X_fit, pred_clean)
    x_line = np.linspace(0, lim_max, 200).reshape(-1, 1)
    ax.plot(x_line, lin_model.predict(x_line), "r-", linewidth=1.5)
    cf = get_conf_intercept(0.05, lin_model, X_fit, pred_clean)
    r2_score = lin_model.score(X_fit, pred_clean)
    reg_txt = 'y={:.2f}x+{:.2f}, $R^2$ = {:.2f}'.format(cf["coeffs"][1], cf["coeffs"][0], r2_score)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel(obs_label, fontsize=12)
    ax.set_ylabel(model_label, fontsize=12)
    ax.set_title(title + "\n" + reg_txt, fontsize=13)
    ax.legend(handles=aqi_legend_patches(), fontsize=8, loc="upper left", framealpha=0.8)
    return ax


# ---------------------------------------------------------------------------
# AQI metrics bar charts
# ---------------------------------------------------------------------------

def plot_aqi_metrics_bars(metrics_dict_by_cat, metrics_to_plot=None,
                          figsize=None, title_prefix=""):
    """
    For each metric in `metrics_to_plot`, draw a grouped bar chart with one bar
    per AQI category, coloured by AQI colour.

    Parameters
    ----------
    metrics_dict_by_cat : dict returned by metrics.metrics_by_aqi_category()
    metrics_to_plot     : list of metric names; defaults to MB/ME/RMSE/NMB/NME/IOA/R2
    figsize             : tuple, auto-computed if None

    Returns
    -------
    fig, axes
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["MB", "ME", "RMSE", "NMB", "NME", "IOA", "R2"]

    n_metrics = len(metrics_to_plot)
    ncols = min(4, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 4.5, nrows * 3.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).ravel()

    cat_labels = [AQI_CATEGORIES[c]["label"] for c in AQI_ORDER]
    cat_colors = [AQI_CATEGORIES[c]["color"] for c in AQI_ORDER]
    # dark edge for visibility of bright/yellow bars
    edge_colors = ["#555555" if AQI_CATEGORIES[c]["color"] in ("#FFFF00", "#00E400")
                   else "white" for c in AQI_ORDER]

    for ax_i, metric in enumerate(metrics_to_plot):
        ax = axes[ax_i]
        values = []
        for cat in AQI_ORDER:
            row = metrics_dict_by_cat.get(cat, {})
            values.append(row.get(metric, np.nan))

        bars = ax.bar(range(len(AQI_ORDER)), values,
                      color=cat_colors, edgecolor=edge_colors, linewidth=0.6)

        # value annotation
        for bar, v in zip(bars, values):
            if np.isnan(v):
                continue
            va = "bottom" if v >= 0 else "top"
            offset = 0.01 * max(abs(v) for v in values if not np.isnan(v)) if any(
                not np.isnan(v) for v in values) else 0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (offset if v >= 0 else -offset),
                    f"{v:.2f}", ha="center", va=va, fontsize=7.5)

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(range(len(AQI_ORDER)))
        ax.set_xticklabels(cat_labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(f"{title_prefix}{metric}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    for ax_j in range(ax_i + 1, len(axes)):
        axes[ax_j].set_visible(False)

    fig.suptitle("Performance Metrics by AQI Category", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# AQI category hit-rate summary bar chart
# ---------------------------------------------------------------------------

def plot_aqi_hit_rate(agreement_dict, ax=None, title="AQI Category Agreement"):
    """
    Three-bar summary: Hit Rate, Over-prediction Rate, Under-prediction Rate.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    labels = ["Hit Rate\n(Exact Match)", "Over-prediction\nRate", "Under-prediction\nRate"]
    vals = [agreement_dict["hit_rate"],
            agreement_dict["over_rate"],
            agreement_dict["under_rate"]]
    bar_colors = ["#2ecc71", "#e74c3c", "#3498db"]

    bars = ax.bar(labels, [v * 100 for v in vals], color=bar_colors,
                  edgecolor="white", linewidth=0.8, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_aqi_confusion_matrix(confusion_df, ax=None,
                              title="AQI Category Confusion Matrix\n(rows=Obs, cols=Pred)"):
    """
    Plot the AQI confusion matrix as a heatmap with cell-count annotations.
    Diagonal cells (correct predictions) are shaded differently.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    matrix = confusion_df.values.astype(float)
    # row-normalise to show fractions
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = matrix / row_sums

    im = ax.imshow(norm_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    n = len(AQI_ORDER)
    short_labels = [c.replace("Unhealthy for Sensitive", "USG") for c in AQI_ORDER]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    # ax.set_xlabel("Predicted Category", fontsize=11)
    # ax.set_ylabel("Observed Category", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # annotate cells with counts
    for i in range(n):
        for j in range(n):
            count = int(matrix[i, j])
            frac = norm_matrix[i, j]
            text_color = "white" if frac > 0.55 else "black"
            ax.text(j, i, f"{count}\n({frac * 100:.0f}%)",
                    ha="center", va="center", fontsize=7.5,
                    color=text_color, fontweight="bold" if i == j else "normal")

    # highlight diagonal
    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="#2ecc71",
                                   linewidth=2.5, zorder=3))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Row-normalised fraction", fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_aqi_prediction_bias(confusion_df, ax=None,
                              title="AQI Prediction Bias\n(per-category error breakdown)"):
    """
    Diverging stacked bar chart of prediction errors per true category.

    For each true category (y-axis):
      LEFT  (negative): days that WERE this category but predicted as something else
                        → stacked by which class they were confused INTO
      RIGHT (positive): days that WERE NOT this category but predicted as this
                        → stacked by which class they actually CAME FROM

    Stack order: Hazardous closest to zero, Good furthest — so severity
    reads outward from the center line.
    Uses symlog scale to handle wide dynamic range.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))   # slightly wider to accommodate legend
    else:
        fig = ax.get_figure()

    matrix       = confusion_df.values.astype(float)
    n            = len(AQI_ORDER)
    short_labels = [c.replace("Unhealthy for Sensitive", "USG") for c in AQI_ORDER]
    y            = np.arange(n)
    h            = 0.35

    cat_colors = ["#1D9E75", "#BA7517", "#E24B4A", "#534AB7",
                  "#888780", "#185FA5", "#639922"][:n]

    # ── Stack order: reversed AQI_ORDER so most severe (Hazardous) is
    #    closest to the zero line, least severe (Good) furthest out ────────────
    stack_order = list(reversed(range(n)))
    false_neg_total = np.array([matrix[i, :].sum() - matrix[i, i] for i in range(n)])
    false_pos_total = np.array([matrix[:, i].sum() - matrix[i, i] for i in range(n)])

    # ── Bars ──────────────────────────────────────────────────────────────────
    for i in range(n):
        left_neg = 0.0
        left_pos = 0.0
        for j in stack_order:
            if i == j:
                continue

            fn = matrix[i, j]   # true i, predicted j  → missed (left)
            fp = matrix[j, i]   # true j, predicted i  → falsely predicted (right)

            if fn > 0:
                ax.barh(y[i] + h / 2, -fn, left=-left_neg,
                        height=h, color=cat_colors[j], alpha=0.85, zorder=2)
                left_neg += fn

            if fp > 0:
                ax.barh(y[i] - h / 2, fp, left=left_pos,
                        height=h, color=cat_colors[j], alpha=0.85, zorder=2)
                left_pos += fp

    # ── Symlog scale ──────────────────────────────────────────────────────────
    ax.set_xscale("symlog", linthresh=10, linscale=0.5)

    # ── X-axis ticks: both sides increase away from zero ─────────────────────
    max_val = max(false_neg_total.max(), false_pos_total.max())
    tick_counts = [t for t in [10, 100, 1_000, 10_000, 100_000, 1_000_000]
                   if t <= max_val * 1.5]

    ax.set_xticks([-t for t in tick_counts] + [0] + tick_counts)
    ax.set_xticklabels(
        [f"{t:,}" for t in tick_counts] + ["0"] +
        [f"{t:,}" for t in tick_counts],
        fontsize=7.5, rotation=30, ha="right"
    )

    # ── Zero line ─────────────────────────────────────────────────────────────
    ax.axvline(0, color="black", linewidth=0.9, zorder=3)

    # ── Y-axis: alternating coloured category labels ──────────────────────────
    ax.set_yticks([])
    ax.set_yticklabels([])

    for i, (label, color) in enumerate(zip(short_labels, cat_colors)):
        if i % 2 == 0:
            y_text = y[i] + h / 2 + 0.15
            va     = "bottom"
        else:
            y_text = y[i] - h / 2 - 0.15
            va     = "top"

    # ── Total count labels at the tip of each bar ─────────────────────────────
    for i in range(n):
        fn_tot = false_neg_total[i]
        fp_tot = false_pos_total[i]

        if fn_tot > 0:
            ax.text(-fn_tot, y[i] + h / 2,
                    f" {fn_tot:,.0f} ", va="center", ha="right",
                    fontsize=7.5, color="k", fontweight="bold", zorder=5)

        if fp_tot > 0:
            ax.text(fp_tot, y[i] - h / 2,
                    f" {fp_tot:,.0f} ", va="center", ha="left",
                    fontsize=7.5, color="k", fontweight="bold", zorder=5)

    # ── Direction labels ──────────────────────────────────────────────────────
    xlim = ax.get_xlim()
    ax.text(xlim[0], n - 0.15, "← missed as...",
            ha="left", va="top", fontsize=8, color="black")
    ax.text(xlim[1], n - 0.15, "falsely predicted from →",
            ha="right", va="top", fontsize=8, color="black")

    # ── Axes labels & grid ────────────────────────────────────────────────────
    ax.set_xlabel("Error count (symlog scale)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1)

    # ── Legend outside the axes, anchored to the right ───────────────────────
    handles = [Patch(color=cat_colors[j], alpha=0.85, label=short_labels[j])
               for j in range(n)]
    legend = ax.legend(
        handles=handles,
        title="Confused with",
        title_fontsize=8,
        fontsize=7.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),   # just right of the axes
        borderaxespad=0,
        frameon=False,
    )

    # Ensure legend is not clipped when saving
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)   # leave room on the right for the legend

    return ax
# ---------------------------------------------------------------------------
# Comprehensive AQI evaluation dashboard (single-call convenience wrapper)
# ---------------------------------------------------------------------------

def plot_aqi_evaluation_dashboard(prediction, observation,
                                  metrics_to_plot=None,
                                  model_name="UFSAQM",
                                  figsize_scatter=(7, 6)):
    """
    Convenience function: generate the full AQI evaluation suite.

    Returns
    -------
    figures : dict with keys 'scatter', 'metrics_bars', 'hit_rate', 'confusion'
    results : dict with keys 'metrics_by_cat', 'agreement'
    """
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)

    # ---- compute ----
    metrics_dict_by_cat = metrics_by_aqi_category(prediction, observation)
    agreement = aqi_category_agreement(prediction, observation)

    # ---- scatter ----
    fig_sc, ax_sc = plt.subplots(figsize=figsize_scatter)
    plot_aqi_scatter(prediction, observation, ax_sc, fig_sc,
                     title=f"{model_name}: Model vs Obs (AQI Categories)")
    fig_sc.tight_layout()

    # ---- metric bars ----
    fig_mb, axes_mb = plot_aqi_metrics_bars(metrics_dict_by_cat,
                                            metrics_to_plot=metrics_to_plot,
                                            title_prefix=f"{model_name} – ")

    # ---- hit rate ----
    fig_hr, ax_hr = plot_aqi_hit_rate(agreement,
                                      title=f"{model_name}: AQI Category Agreement")

    # ---- confusion ----
    fig_cf, ax_cf = plot_aqi_confusion_matrix(
        agreement["confusion_matrix"],
        title=f"{model_name}: AQI Category Confusion Matrix\n(rows=Obs, cols=Pred)"
    )

    figures = {
        "scatter": fig_sc,
        "metrics_bars": fig_mb,
        "hit_rate": fig_hr,
        "confusion": fig_cf,
    }
    results = {
        "metrics_by_cat": metrics_dict_by_cat,
        "agreement": agreement,
    }
    return figures, results


def plotComparisonIntercept_fast(X, Y, bin_size, ax, fig, side=True):
    hh, xedges, yedges = np.histogram2d(X, Y, bins=bin_size)
    hh = hh.T / len(X)
    im = ax.pcolormesh(
        xedges, yedges, hh,
        cmap="jet",
        norm=colors.LogNorm(vmin=hh[hh > 0].min(), vmax=hh.max())
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    if side:
        cb.set_label("Density", fontsize=10, rotation=270, labelpad=12)

    Xr = X.reshape(-1, 1)
    lin_model = LinearRegression().fit(Xr, Y)

    line_x = np.linspace(0, np.nanmax(X), 100).reshape(-1, 1)
    line_y = lin_model.predict(line_x)

    r2 = lin_model.score(Xr, Y)
    slope = lin_model.coef_[0]
    intercept = lin_model.intercept_

    reg_txt = f"y={slope:.2f}x+{intercept:.2f}, $R^2$={r2:.2f}"
    ax.plot(line_x, line_y, 'r', lw=2)

    return reg_txt