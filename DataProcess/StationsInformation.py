import os
import pickle
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from utils.utils import build_kdtree, findSpatialIndex_kdtree

start_year = 2021
end_year = 2024
smoke_filedir = "/Users/zongrunli/Desktop/ForecastEvaluation/data/smoke_data"
obs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
stn_output_path = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
THRESHOLD = 0.5

df = pd.read_csv(obs_file, usecols=["siteCode", "Latitude", "Longitude"], dtype={"siteCode": str})
df = df.drop_duplicates(subset="siteCode")
site_dict = {
    row.siteCode: {"coords": (float(row.Longitude), float(row.Latitude))}
    for row in df.itertuples(index=False)
}
print(f" Found {len(site_dict)} unique sites.")

init_smoke_file = os.path.join(smoke_filedir, "krigedPM25_%d_v2_statfix_meanbk.nc" % start_year)
ds = nc.Dataset(init_smoke_file)
lon = ds["lon"][:]
lat = ds["lat"][:]
lon_min, lon_max = np.min([np.min(ds["we_lon"]), np.min(ds["ns_lon"])]), np.max([np.max(ds["we_lon"]), np.max(ds["ns_lon"])])
lat_min, lat_max = np.min([np.min(ds["we_lat"]), np.min(ds["ns_lat"])]), np.max([np.max(ds["we_lat"]), np.max(ds["ns_lat"])])
tree = build_kdtree(lon, lat)

stn_loc_idx = {}
# get index
for k in site_dict:
    cur_lon, cur_lat = site_dict[k]["coords"]
    if lon_min <= cur_lon <= lon_max and lat_min <= cur_lat <= lat_max:
        stn_loc_idx[k] = findSpatialIndex_kdtree(cur_lon, cur_lat, tree, lon.shape)

# accumulate smoke exceedance dates per site across all years
smoke_dates = {code: [] for code in stn_loc_idx}
for year in range(start_year, end_year + 1):
    smoke_data_file = os.path.join(smoke_filedir, "krigedPM25_%d_v2_statfix_meanbk.nc" % year)
    ds = nc.Dataset(smoke_data_file)
    pm25 = ds["PM25"][:]
    bkg_pm25 = ds["Background_PM25"][:]
    smoke_pm25 = pm25 - bkg_pm25
    doy_arr = ds["doy"][:]
    date_list = [datetime(year, 1, 1) + timedelta(days=float(d) - 1) for d in doy_arr]

    for code in stn_loc_idx:
        row, col = stn_loc_idx[code]
        ts = smoke_pm25[:, row, col]
        t_total = pm25[:, row, col]

        exceed_idx = np.where((ts / t_total) >= THRESHOLD)[0]
        if exceed_idx.size == 0:
            continue
        smoke_dates[code].extend(date_list[i] for i in exceed_idx)

# merge into site_dict: attach "smoke_dates" only when smoke data exists for that site
for code, dates in smoke_dates.items():
    if dates:
        site_dict[code]["smoke_dates"] = dates

with open(stn_output_path, "wb") as f:
    pickle.dump(site_dict, f)
