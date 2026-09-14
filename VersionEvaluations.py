import os
import pickle
import pandas as pd
from datetime import datetime, timedelta


MODEL_VAR = "PM25_UFSAQM"
OBS_VAR   = "PM25"
lead_time = 6
versions = {
    5: [datetime(2021, 1, 1), datetime(2021, 7, 19)],
    6: [datetime(2021, 7, 20), datetime(2024, 5, 13)],
    7: [datetime(2024, 5, 14), datetime(2024, 12, 31)],
}
for ver in versions:
    start_time, end_time = versions[ver][0], versions[ver][1]
    obs_file     = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
    stats_folder = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % lead_time
    save_path    = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/scatter_data_%02d_v%d.pkl" % (lead_time, ver)

    # ── observations ─────────────────────────────────────────────────────────────
    obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
    obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]

    # ── model files ──────────────────────────────────────────────────────────────
    model_list = []
    cur_time = start_time
    while cur_time <= end_time:
        date_str = cur_time.strftime("%Y%m%d")
        cur_model_file = os.path.join(stats_folder, f"site_level_aqm_{date_str}_%02d.csv" % lead_time)
        if os.path.exists(cur_model_file):
            model_list.append(pd.read_csv(cur_model_file, parse_dates=["Time"], dtype={"siteCode": str}))
        cur_time += timedelta(days=1)
    model_df = pd.concat(model_list, ignore_index=True)

    # ── merge & clean ────────────────────────────────────────────────────────────
    merged = model_df.merge(
        obs_df[["siteCode", "Time", OBS_VAR]],
        on=["siteCode", "Time"], how="inner"
    )
    merged = merged.dropna(subset=[MODEL_VAR, OBS_VAR])
    merged = merged[(merged[OBS_VAR] >= 0) & (merged[MODEL_VAR] >= 0)]

    data = {
        "obs":   merged[OBS_VAR].to_numpy(float),
        "model": merged[MODEL_VAR].to_numpy(float),
    }

    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data['obs'])} obs/model pairs to {save_path}")