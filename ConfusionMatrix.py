import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
from utils.utils import assign_aqi_category


start_time = datetime(2021, 1, 1)
end_time   = datetime(2024, 12, 31)
init_time = 6

MODEL_VAR   = "PM25_UFSAQM"
OBS_VAR     = "PM25"
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stats_folder  = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/%02d" % init_time
save_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/site_aqi_subsets.pkl"

with open(stn_locs_file, "rb") as f:
    site_dict = pickle.load(f)

obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] <= end_time)]
obs_df["stateCode"]  = obs_df["siteCode"].str[:2]
obs_df["countyCode"] = obs_df["siteCode"].str[:5]

model_list = []
cur_time = start_time
while cur_time <= end_time:
    date_str = cur_time.strftime("%Y%m%d")
    path = os.path.join(stats_folder, f"site_level_aqm_{date_str}_{init_time:02d}.csv")
    if os.path.exists(path):
        dtype = {"siteCode": str}
        model_list.append(pd.read_csv(path, parse_dates=["Time"], dtype=dtype))
    cur_time += timedelta(days=1)
model_df  = pd.concat(model_list,  ignore_index=True)

merge_cols = ["siteCode", "Time", OBS_VAR]
merged_site = model_df.merge(
    obs_df[merge_cols],
    on=["siteCode", "Time"], how="inner"
)

for df in [merged_site]:
    df["obs_aqi_cat"]  = assign_aqi_category(df[OBS_VAR].values)
    df["pred_aqi_cat"] = assign_aqi_category(df[MODEL_VAR].values)

if "smoke" not in merged_site.columns:
    smoke_day_sets = {
        c: {d.date() for d in site_dict[c].get("smoke_dates", [])}
        for c in site_dict
    }
    merged_site["smoke"] = [
        code in smoke_day_sets and t.date() in smoke_day_sets[code]
        for code, t in zip(merged_site["siteCode"], merged_site["Time"])
    ]

merged_site["smoke"] = merged_site["smoke"].astype(bool)
subsets = [
    ("All",       merged_site),
    ("Smoke",     merged_site[merged_site["smoke"]]),
    ("Non-smoke", merged_site[~merged_site["smoke"]]),
]

# save
subsets_dict = {name: sdf for name, sdf in subsets}
with open(save_path, "wb") as f:
    pickle.dump(subsets_dict, f)