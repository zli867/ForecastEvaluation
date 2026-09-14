import numpy as np
import scipy.stats
from utils.utils import assign_aqi_category, AQI_ORDER
import pandas as pd
from sklearn.metrics import r2_score

# reference: https://www.tandfonline.com/doi/full/10.1080/10962247.2016.1265027
def remove_nan_values(prediction, observation):
    valid_idx = (~np.isnan(prediction)) & (~np.isnan(observation))
    valid_p = prediction[valid_idx]
    valid_o = observation[valid_idx]
    return valid_p, valid_o

# Regressions
def MB(prediction, observation):
    return np.sum(prediction - observation) / len(prediction)


def RMSE(prediction, observation):
    return np.sqrt(np.sum((prediction - observation) ** 2) / len(prediction))


def FB(prediction, observation, dl=0.1):
    p = np.maximum(np.asarray(prediction, float), dl)
    o = np.maximum(np.asarray(observation, float), dl)
    return (2 / len(p)) * np.sum((p - o) / (p + o)) * 100

def FE(prediction, observation, dl=0.1):
    p = np.maximum(np.asarray(prediction, float), dl)
    o = np.maximum(np.asarray(observation, float), dl)
    return (2 / len(p)) * np.sum(np.abs(p - o) / (p + o)) * 100


def pearson_r(prediction, observation):
    r = scipy.stats.pearsonr(prediction, observation)
    return r[0]


def NMSE(prediction, observation):
    mse = np.mean((observation - prediction) ** 2)
    norm = np.mean(prediction) * np.mean(observation)
    return mse / norm


def COD(prediction, observation):
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)

    obs_mean = np.mean(observation)
    ss_res = np.sum((observation - prediction) ** 2)   # residual sum of squares
    ss_tot = np.sum((observation - obs_mean) ** 2)      # total sum of squares

    return 1 - ss_res / ss_tot


# Classifications
def _confusion_counts(obs_cat, mod_cat, category):
    """One-vs-rest TP, FP, FN for a single category."""
    obs_cat, mod_cat = np.asarray(obs_cat), np.asarray(mod_cat)
    obs_is, mod_is = (obs_cat == category), (mod_cat == category)
    TP = int(np.sum(obs_is & mod_is))
    FP = int(np.sum(~obs_is & mod_is))
    FN = int(np.sum(obs_is & ~mod_is))
    return TP, FP, FN


def precision(obs_cat, mod_cat, category):
    """Precision for one category: TP / (TP + FP)."""
    TP, FP, _ = _confusion_counts(obs_cat, mod_cat, category)
    return TP / (TP + FP) if (TP + FP) > 0 else np.nan


def recall(obs_cat, mod_cat, category):
    """Recall for one category: TP / (TP + FN)."""
    TP, _, FN = _confusion_counts(obs_cat, mod_cat, category)
    return TP / (TP + FN) if (TP + FN) > 0 else np.nan


def f1(obs_cat, mod_cat, category):
    """F1 for one category: 2TP / (2TP + FP + FN)."""
    TP, FP, FN = _confusion_counts(obs_cat, mod_cat, category)
    return 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else np.nan


def false_alarm_ratio(obs_cat, mod_cat, category):
    """False alarm ratio for one category: 1 - precision."""
    p = precision(obs_cat, mod_cat, category)
    return 1 - p if not np.isnan(p) else np.nan


def classification_accuracy(obs_cat, mod_cat):
    """Overall multiclass accuracy: fraction of samples in the correct category."""
    obs_cat, mod_cat = np.asarray(obs_cat), np.asarray(mod_cat)
    return float(np.mean(obs_cat == mod_cat)) if len(obs_cat) else np.nan


def metrics_by_aqi_category(prediction, observation, metric_funcs=None):
    if metric_funcs is None:
        metric_funcs = {
            "MB": MB, "RMSE": RMSE,
            "Pearson R": pearson_r, "R2": R2,
            "FB": FB, "FE": FE, "NMSE": NMSE
        }
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    pred_clean, obs_clean = remove_nan_values(prediction, observation)
    obs_cats = assign_aqi_category(obs_clean)
    results = {}
    for cat in AQI_ORDER:
        mask = obs_cats == cat
        if mask.sum() < 2:  # skip categories with too few samples
            results[cat] = {"N": int(mask.sum())}
            continue
        p_sub = pred_clean[mask]
        o_sub = obs_clean[mask]
        row = {"N": int(mask.sum())}
        for name, func in metric_funcs.items():
            try:
                row[name] = func(p_sub, o_sub)
            except Exception:
                row[name] = np.nan
        results[cat] = row
    return results


# ---------------------------------------------------------------------------
# AQI category agreement / confusion
# ---------------------------------------------------------------------------
def aqi_category_agreement(prediction, observation):
    """
    Compute hit rate, over-prediction rate, and under-prediction rate
    for AQI category assignment.

    Returns dict with keys: hit_rate, over_rate, under_rate, confusion_matrix_df
    """
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    pred_clean, obs_clean = remove_nan_values(prediction, observation)

    pred_cats = assign_aqi_category(pred_clean)
    obs_cats = assign_aqi_category(obs_clean)

    order_idx = {c: i for i, c in enumerate(AQI_ORDER)}
    pred_idx = np.array([order_idx[c] for c in pred_cats])
    obs_idx = np.array([order_idx[c] for c in obs_cats])

    hit_rate = np.mean(pred_cats == obs_cats)
    over_rate = np.mean(pred_idx > obs_idx)
    under_rate = np.mean(pred_idx < obs_idx)

    # confusion matrix (rows = observed, cols = predicted)
    n = len(AQI_ORDER)
    matrix = np.zeros((n, n), dtype=int)
    for oi, pi in zip(obs_idx, pred_idx):
        matrix[oi, pi] += 1

    conf_df = pd.DataFrame(
        matrix,
        index=[f"Obs: {c}" for c in AQI_ORDER],
        columns=[f"Pred: {c}" for c in AQI_ORDER],
    )

    return {
        "hit_rate": hit_rate,
        "over_rate": over_rate,
        "under_rate": under_rate,
        "confusion_matrix": conf_df,
    }


def metrics_by_lead(df, model_var, obs_var, metrics):
    out = {}
    for lt, g in df.groupby("lead_time"):
        mod, obs = g[model_var].values, g[obs_var].values
        mask = np.isfinite(mod) & np.isfinite(obs)
        out[lt] = metrics(mod[mask], obs[mask])
    return pd.Series(out).sort_index()


def metrics_by_lead_per_cat(df, model_var, obs_var, metrics, cat_col="obs_aqi_cat"):
    rows = []
    for cat in AQI_ORDER:
        sub = df[df[cat_col] == cat]
        grp = (
            sub.groupby("lead_time")
               .apply(lambda g: metrics(g[model_var].values[np.isfinite(g[model_var].values) & np.isfinite(g[obs_var].values)],
                                        g[obs_var].values[np.isfinite(g[model_var].values) & np.isfinite(g[obs_var].values)]))
               .reset_index(name="value")
        )
        grp["Category"] = cat
        rows.append(grp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def grouped_metric(merged, MODEL_VAR, OBS_VAR, level, metric, lead_time=None):
    df = merged.copy()
    df["siteCode"] = df["siteCode"].astype(str).str.zfill(9)
    df["state_fips"] = df["siteCode"].str[:2]
    df["county_fips"] = df["siteCode"].str[:5]
    if lead_time is not None and "lead_time" in df.columns:
        df = df[df["lead_time"] == lead_time]

    key = "state_fips" if level == "state" else "county_fips"
    out = {}
    for gid, g in df.groupby(key):
        p, o = remove_nan_values(g[MODEL_VAR].to_numpy(float), g[OBS_VAR].to_numpy(float))
        out[gid] = float(metric(p, o))
    return out


def grouped_site_count(merged, level):
    """Number of unique monitoring sites in each state/county."""
    df = merged.copy()
    df["siteCode"]    = df["siteCode"].astype(str).str.zfill(9)
    df["state_fips"]  = df["siteCode"].str[:2]
    df["county_fips"] = df["siteCode"].str[:5]
    key = "state_fips" if level == "state" else "county_fips"
    return df.groupby(key)["siteCode"].nunique().to_dict()

def time_bin(df, scale):
    _season = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
               6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    t = df["Time"]
    if scale == "Hourly":
        return t
    if scale == "Daily":
        return t.dt.floor("D")
    if scale == "Weekly":
        return t.dt.to_period("W").astype(str)   # ISO Mon–Sun weeks
    if scale == "Monthly":
        return t.dt.to_period("M").astype(str)
    if scale == "Seasonal":
        yr = t.dt.year + (t.dt.month == 12).astype(int)   # Dec -> next year's DJF
        return yr.astype(str) + "-" + t.dt.month.map(_season)
    if scale == "Yearly":
        return t.dt.year.astype(str)
    raise ValueError(scale)


def overall_metrics_at_scale(df, scale, metrics, model_var, obs_var):
    # TODO:
    """
    Week 1 — an active fire week: 5 smoke days, each with bias −4.
    Week 2 — an isolated smoke day: 1 smoke day, bias −1.
    Hourly (or daily) MB — every smoke day weighted equally:
    MB = (5·(−4) + 1·(−1)) / 6 = −21/6 = −3.5
    Weekly MB — each week collapsed to one bin, then averaged:
    Week 1 bin = −4, Week 2 bin = −1 → MB = (−4 + −1)/2 = −2.5
    """
    #TODO:
    g = df.copy()
    if scale == "Hourly":
        avg = g
    else:
        g["_bin"] = time_bin(g, scale)
        avg = g.groupby(["siteCode", "_bin"])[[model_var, obs_var]].mean().reset_index()
    mod, obs = avg[model_var].values, avg[obs_var].values
    mask = np.isfinite(mod) & np.isfinite(obs)
    return metrics(mod[mask], obs[mask])


def spatial_group_metric(df, key, model_var, obs_var, lead_times, metric_fn):
    """Spatial mean per (group, Time, lead_time) — hourly resolution kept — then metric across all pairs."""
    ts = df.groupby([key, "Time", "lead_time"])[[model_var, obs_var]].mean().reset_index()
    vals = []
    for lt in lead_times:
        sub = ts if lt is None else ts[ts["lead_time"] == lt]
        mod, obs = sub[model_var].to_numpy(float), sub[obs_var].to_numpy(float)
        m = np.isfinite(mod) & np.isfinite(obs)
        vals.append(metric_fn(mod[m], obs[m]))
    return vals


def aqi_category_agreement_tol(prediction, observation, tol=1.0):
    """
    Same as aqi_category_agreement, but a cross-category pair is forgiven
    (counted as a hit on the observed category) when |pred - obs| <= tol.
    This only forgives genuine boundary-huggers, e.g. pred=8.9 vs obs=9.1.
    tol is in µg/m³ and should be justified by measurement uncertainty.
    """
    prediction = np.asarray(prediction, dtype=float)
    observation = np.asarray(observation, dtype=float)
    pred_clean, obs_clean = remove_nan_values(prediction, observation)

    pred_cats = assign_aqi_category(pred_clean)
    obs_cats = assign_aqi_category(obs_clean)

    order_idx = {c: i for i, c in enumerate(AQI_ORDER)}
    pred_idx = np.array([order_idx[c] for c in pred_cats])
    obs_idx = np.array([order_idx[c] for c in obs_cats])

    # forgive boundary crossings within tol: snap predicted -> observed
    forgiven = (pred_idx != obs_idx) & (np.abs(pred_clean - obs_clean) <= tol)
    eff_pred_idx = np.where(forgiven, obs_idx, pred_idx)

    hit_rate = np.mean(eff_pred_idx == obs_idx)
    over_rate = np.mean(eff_pred_idx > obs_idx)
    under_rate = np.mean(eff_pred_idx < obs_idx)

    n = len(AQI_ORDER)
    matrix = np.zeros((n, n), dtype=int)
    for oi, pi in zip(obs_idx, eff_pred_idx):
        matrix[oi, pi] += 1

    conf_df = pd.DataFrame(
        matrix,
        index=[f"Obs: {c}" for c in AQI_ORDER],
        columns=[f"Pred: {c}" for c in AQI_ORDER],
    )
    return {
        "hit_rate": hit_rate,
        "over_rate": over_rate,
        "under_rate": under_rate,
        "n_forgiven": int(forgiven.sum()),
        "confusion_matrix": conf_df,
    }