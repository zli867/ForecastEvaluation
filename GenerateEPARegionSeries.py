import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
from utils.geoutil import StatePolygon, CountyPolygon, FIPS_list
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from utils.Visualizations import plotComparisonIntercept
from utils.metrics import MB, RMSE, FB, FE, pearson_r, NMSE, COD
import pyproj
from utils.ZonalStats import polygon_statistics
from utils.utils import build_kdtree, findSpatialIndex_kdtree
from sklearn.metrics import r2_score

from utils.geoutil import StatePolygon, FIPS_list
plot_data = {}
epa_regions = {
    1: {
        "name": "New England",
        "headquarters": "Boston",
        "states": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"]
    },
    2: {
        "name": "New York",
        "headquarters": "New York City",
        "states": ["New Jersey", "New York"]
    },
    3: {
        "name": "Mid-Atlantic",
        "headquarters": "Philadelphia",
        "states": ["Delaware", "District of Columbia", "Maryland", "Pennsylvania", "Virginia", "West Virginia"]
    },
    4: {
        "name": "Southeast",
        "headquarters": "Atlanta",
        "states": ["Alabama", "Florida", "Georgia", "Kentucky", "Mississippi", "North Carolina", "South Carolina", "Tennessee"]
    },
    5: {
        "name": "Great Lakes",
        "headquarters": "Chicago",
        "states": ["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"]
    },
    6: {
        "name": "South Central",
        "headquarters": "Dallas",
        "states": ["Arkansas", "Louisiana", "New Mexico", "Oklahoma", "Texas"]
    },
    7: {
        "name": "Midwest",
        "headquarters": "Kansas City",
        "states": ["Iowa", "Kansas", "Missouri", "Nebraska"]
    },
    8: {
        "name": "Mountains and Plains",
        "headquarters": "Denver",
        "states": ["Colorado", "Montana", "North Dakota", "South Dakota", "Utah", "Wyoming"]
    },
    9: {
        "name": "Pacific Southwest",
        "headquarters": "San Francisco",
        "states": ["Arizona", "California", "Nevada"]
    },
    10: {
        "name": "Pacific Northwest",
        "headquarters": "Seattle",
        "states": ["Idaho", "Oregon", "Washington"]
    }
}

fire_checks = {
    2021: (9, datetime(2021, 8, 1), datetime(2021, 8, 31)),
    2022: (6, datetime(2022, 5, 1), datetime(2022, 5, 31)),
    2023: (10, datetime(2023, 8, 1), datetime(2023, 8, 31)),
    2024: (10, datetime(2024, 8, 1), datetime(2024, 8, 31)),
}

metrics_dict = {
    "MB": {"func": MB},
    "RMSE": {"func": RMSE},
    "corr": {"func": pearson_r},
}

for year in fire_checks:
    print(f"year: {year}")
    region_numbers = [fire_checks[year][0]]
    start_time = fire_checks[year][1]
    end_time = fire_checks[year][2]
    states = []
    for n in region_numbers:
        states.extend(epa_regions[n]["states"])
    states_polys = StatePolygon(states)
    state_ids = FIPS_list(states)
    county_ids, county_polygons = CountyPolygon(states)
    state_geom_dict  = dict(zip(state_ids, states_polys))
    county_geom_dict = dict(zip(county_ids, county_polygons))

    # modeled stats info
    static_file = "/Users/zongrunli/Desktop/SmkFcstEval/data/fcst_data/static/aqm_static.npz"
    file_folder = "/Volumes/T9/UFSAQM/Evaluation/values"
    static_info = np.load(static_file)
    lon, lat = static_info["lon"], static_info["lat"]
    x, y = static_info["x"], static_info["y"]
    crs_proj4 = static_info["crs_proj4"].item()
    crs = pyproj.Proj(crs_proj4)
    DX = x[0, 1] - x[0, 0]
    DY = y[1, 0] - y[0, 0]
    tree = build_kdtree(x, y)
    tree_lonlat = build_kdtree(lon, lat)
    model_info = {"crs": crs, "DX": DX, "DY": DY, "tree": tree, "X_ctr": x, "Y_ctr": y}
    lon_min, lon_max = np.min(lon), np.max(lon)
    lat_min, lat_max = np.min(lat), np.max(lat)

    stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
    obs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
    stats_folder = "/Users/zongrunli/Desktop/ForecastEvaluation/data/eval_data/UFSAQM_RAW_NEW/site_levels/06"
    with open(stn_locs_file, 'rb') as file:
        site_dict = pickle.load(file)

    obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
    obs_df = obs_df[(obs_df["Time"] >= start_time) & (obs_df["Time"] < end_time)]
    obs_df["stateCode"] = obs_df["siteCode"].str[:2]
    obs_df["countyCode"] = obs_df["siteCode"].str[:5]

    cur_time = start_time
    model_list = []
    while cur_time < end_time:
        date_str = cur_time.strftime("%Y%m%d")
        cur_model_file = os.path.join(stats_folder, f"site_level_aqm_{date_str}_06.csv")
        if os.path.exists(cur_model_file):
            model_list.append(pd.read_csv(cur_model_file, parse_dates=["Time"], dtype={"siteCode": str}))
        cur_time += timedelta(days=1)
    model_df = pd.concat(model_list, ignore_index=True)
    merged_df = model_df.merge(obs_df[["siteCode", "Time", "PM25"]], on=["siteCode", "Time"], how="inner")

    merged_df["stateCode"] = merged_df["siteCode"].str[:2]
    merged_df = merged_df[merged_df["stateCode"].isin(state_ids)]

    paired_df = merged_df.dropna(subset=["PM25", "PM25_UFSAQM"]).copy()
    merged = (
        paired_df.groupby("Time")
        .agg(
            PM25=("PM25", "mean"),
            PM25_UFSAQM_stn_avg=("PM25_UFSAQM", "mean"),
            obs_p25=("PM25", lambda x: x.quantile(0.25)),
            obs_p75=("PM25", lambda x: x.quantile(0.75)),
            model_p25=("PM25_UFSAQM", lambda x: x.quantile(0.25)),
            model_p75=("PM25_UFSAQM", lambda x: x.quantile(0.75)),
            n_sites=("siteCode", "nunique"),
        ).reset_index()
    )
    pred = merged["PM25_UFSAQM_stn_avg"].values
    obs = merged["PM25"].values
    metrics = {
        name: float(info["func"](pred, obs))
        for name, info in metrics_dict.items()
    }
    plot_data[year] = {
        "region_numbers": region_numbers,
        "start_time": start_time,
        "end_time": end_time,
        "merged": merged.copy(),
        "metrics": metrics,
    }

with open("/Users/zongrunli/Desktop/ForecastEvaluation/data/pub_data/plot_data_by_year.pkl", "wb") as f:
    pickle.dump(plot_data, f)