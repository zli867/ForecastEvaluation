import os, pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from utils.utils import assign_aqi_category, AQI_ORDER
from utils.metrics import classification_accuracy, false_alarm_ratio, recall, f1

# ── config ───────────────────────────────────────────────────────────────────
START, END = datetime(2021, 1, 1), datetime(2024, 12, 31)
MODEL_VAR, OBS_VAR = "PM25_UFSAQM", "PM25"
INIT = 6
STN_FILE  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
OBS_FILE  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
STATS_DIR = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % INIT
CSV_ALL   = "/Users/zongrunli/Desktop/ForecastEvaluation/data/metrics/classification_metrics_%02d.csv" % INIT
CSV_LEAD  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/metrics/classification_metrics_byLead_%02d.csv" % INIT

SUBSETS        = ["All", "Smoke", "Non-smoke"]
CLASS_METRICS  = ["Accuracy", "false_alarm_ratio", "hit_rate", "F1"]
CAT_ORDER      = ["Overall"] + AQI_ORDER
CAT_METRIC_FNS = {"false_alarm_ratio": false_alarm_ratio, "hit_rate": recall, "F1": f1}


def load_merged():
    """Load model + obs, merge, and add the smoke-day flag."""
    with open(STN_FILE, "rb") as f:
        site_dict = pickle.load(f)

    obs = pd.read_csv(OBS_FILE, parse_dates=["Time"], dtype={"siteCode": str})
    obs = obs[(obs["Time"] >= START) & (obs["Time"] <= END)]

    frames, cur = [], START
    while cur <= END:
        fp = os.path.join(STATS_DIR, f"site_level_aqm_{cur:%Y%m%d}_{INIT:02d}.csv")
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp, parse_dates=["Time"], dtype={"siteCode": str}))
        cur += timedelta(days=1)
    model = pd.concat(frames, ignore_index=True)

    merged = model.merge(obs[["siteCode", "Time", OBS_VAR]], on=["siteCode", "Time"], how="inner")
    smoke = {c: {d.date() for d in site_dict[c].get("smoke_dates", [])} for c in site_dict}
    merged["isSmokeDay"] = [code in smoke and t.date() in smoke[code]
                            for code, t in zip(merged["siteCode"], merged["Time"])]
    return merged


def classification_rows(obs_vals, mod_vals, base):
    """Long-format rows: overall Accuracy + per-AQI-category Precision/Recall/F1."""
    m = np.isfinite(obs_vals) & np.isfinite(mod_vals)
    obs_cat = assign_aqi_category(obs_vals[m])
    mod_cat = assign_aqi_category(mod_vals[m])

    rows = [{**base, "category": "Overall", "metric": "Accuracy",
             "value": classification_accuracy(obs_cat, mod_cat), "n": int(len(obs_cat))}]
    for cat in AQI_ORDER:
        n = int(np.sum(obs_cat == cat))                  # observed count in this category
        for name, fn in CAT_METRIC_FNS.items():
            rows.append({**base, "category": cat, "metric": name,
                         "value": fn(obs_cat, mod_cat, cat), "n": n})
    return rows


def finalize(rows, sort_keys):
    df = pd.DataFrame(rows)
    df["subset"]   = pd.Categorical(df["subset"],   SUBSETS,       ordered=True)
    df["category"] = pd.Categorical(df["category"], CAT_ORDER,     ordered=True)
    df["metric"]   = pd.Categorical(df["metric"],   CLASS_METRICS, ordered=True)
    return df.sort_values(sort_keys).reset_index(drop=True)


def main():
    merged = load_merged()
    subsets = {"All":       merged,
               "Smoke":     merged[merged["isSmokeDay"]],
               "Non-smoke": merged[~merged["isSmokeDay"]]}

    # CSV 1 — pooled over all lead times
    rows = []
    for name, sdf in subsets.items():
        rows += classification_rows(sdf[OBS_VAR].values, sdf[MODEL_VAR].values, {"subset": name})
    df_all = finalize(rows, ["subset", "category", "metric"])

    # CSV 2 — broken out by lead time
    lead_rows = []
    for name, sdf in subsets.items():
        for lead, g in sdf.groupby("lead_time"):
            lead_rows += classification_rows(g[OBS_VAR].values, g[MODEL_VAR].values,
                                             {"lead": int(lead), "subset": name})
    df_lead = finalize(lead_rows, ["subset", "category", "metric", "lead"])

    os.makedirs(os.path.dirname(CSV_ALL), exist_ok=True)
    df_all.to_csv(CSV_ALL, index=False)
    df_lead.to_csv(CSV_LEAD, index=False)
    print(f"saved -> {CSV_ALL}")
    print(f"saved -> {CSV_LEAD}")
    return df_all, df_lead


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    df_all, _ = main()
    print(df_all)