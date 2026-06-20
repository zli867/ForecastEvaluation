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


def MB(prediction, observation):
    return np.sum(prediction - observation) / len(prediction)


def ME(prediction, observation):
    return np.sum(np.abs(prediction - observation)) / len(prediction)


def RMSE(prediction, observation):
    return np.sqrt(np.sum((prediction - observation) ** 2) / len(prediction))


def CRMSE(prediction, observation):
    p_mean, o_mean = np.mean(prediction), np.mean(observation)
    return np.sqrt((1 / len(prediction) * np.sum(((prediction - p_mean) - (observation - o_mean)) ** 2)))


def NMB(prediction, observation):
    return (np.sum(prediction - observation) / np.sum(observation)) * 100


def NME(prediction, observation):
    return ((np.sum(np.abs(prediction - observation))) / np.sum(observation)) * 100


def MNB(prediction, observation):
    return (1 / len(prediction)) * np.sum((prediction - observation) / observation) * 100


def MNE(prediction, observation):
    return (1 / len(prediction)) * np.sum(np.abs(prediction - observation) / observation) * 100


def FB(prediction, observation):
    return (2 / len(prediction)) * np.sum((prediction - observation) / (prediction + observation)) * 100


def FE(prediction, observation):
    return (2 / len(prediction)) * np.sum(np.abs(prediction - observation) / (prediction + observation)) * 100


def IOA(prediction, observation):
    numerator = np.sum((prediction - observation) ** 2)
    prediction_shift = np.abs((prediction - np.mean(observation)))
    observation_shift = np.abs((observation - np.mean(observation)))
    denominator = np.sum((prediction_shift + observation_shift) ** 2)
    return 1 - numerator / denominator


def pearson_r(prediction, observation):
    r = scipy.stats.pearsonr(prediction, observation)
    return r[0]


def pearson_r2(prediction, observation):
    return pearson_r(prediction, observation) ** 2

def NMSE(obs, pred):
    """
    Normalized Mean Square Error
    NMSE = MSE / mean(|obs|^2)
    Returns NMSE in linear scale (not dB)
    """
    mse = np.mean(np.abs(obs - pred)**2)
    norm = np.mean(np.abs(obs)**2)
    return mse / norm

def R2(prediction, observation):
    return r2_score(prediction, observation)


def metrics_by_aqi_category(prediction, observation, metric_funcs=None):
    if metric_funcs is None:
        metric_funcs = {
            "MB": MB, "ME": ME, "RMSE": RMSE,
            "NMB": NMB, "NME": NME, "IOA": IOA,
            "Pearson R": pearson_r, "R2": R2,
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
    g = df.copy()
    if scale == "Hourly":
        avg = g
    else:
        g["_bin"] = time_bin(g, scale)
        avg = g.groupby(["siteCode", "_bin"])[[model_var, obs_var]].mean().reset_index()
    mod, obs = avg[model_var].values, avg[obs_var].values
    mask = np.isfinite(mod) & np.isfinite(obs)
    return metrics(obs[mask], mod[mask])


def spatial_group_metric(df, key, model_var, obs_var, lead_times, metric_fn):
    """Spatial mean per (group, lead_time), then R2 across groups per lead time."""
    ts = df.groupby([key, "lead_time"])[[model_var, obs_var]].mean().reset_index()
    vals = []
    for lt in lead_times:
        sub = ts if lt is None else ts[ts["lead_time"] == lt]
        vals.append(metric_fn(sub[model_var].to_numpy(float), sub[obs_var].to_numpy(float)))
    return vals