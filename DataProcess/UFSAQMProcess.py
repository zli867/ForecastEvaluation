import pickle
from datetime import datetime
import numpy as np
import os
import pandas as pd
from utils.utils import build_kdtree, findSpatialIndex_kdtree
import pyproj

static_file = "/Users/zongrunli/Desktop/SmkFcstEval/data/fcst_data/static/aqm_static.npz"
file_folder = "/Users/zongrunli/Desktop/SmkFcstEval/data/fcst_data/values"
stn_locs_file = "/Users/zongrunli/Desktop/SmkFcstEval/processed_data/stn_info.pkl"
output_dir = "/Users/zongrunli/Desktop/SmkFcstEval/processed_data"
with open(stn_locs_file, 'rb') as file:
    site_dict = pickle.load(file)

static_info = np.load(static_file)
lon, lat = static_info["lon"], static_info["lat"]
x, y = static_info["x"], static_info["y"]
crs_proj4 = static_info["crs_proj4"].item()  # extracts plain Python string
crs = pyproj.Proj(crs_proj4)
DX = x[0, 1] - x[0, 0]
DY = y[1, 0] - y[0, 0]

# required info, crs, tree, DX, DY, X_ctr, Y_ctr
tree = build_kdtree(x, y)
tree_lonlat = build_kdtree(lon, lat)
model_info = {"crs": crs, "DX": DX, "DY": DY, "tree": tree, "X_ctr": x, "Y_ctr": y}
lon_min, lon_max = np.min(lon), np.max(lon)
lat_min, lat_max = np.min(lat), np.max(lat)
stn_loc_idx = {}

for k in site_dict:
    cur_lon, cur_lat = site_dict[k]["coords"]
    if lon_min <= cur_lon <= lon_max and lat_min <= cur_lat <= lat_max:
        stn_loc_idx[k] = findSpatialIndex_kdtree(cur_lon, cur_lat, tree_lonlat, lon.shape)


for filename in os.listdir(file_folder):
    cur_path = os.path.join(file_folder, filename)
    cur_values = np.load(cur_path)
    cur_time = cur_values["time"][0:24]
    time_datetime = pd.to_datetime(cur_time, unit="s").to_pydatetime().tolist()
    cur_pm = cur_values["PM25_UFSAQM"][0:24, :, :]
    date_str = datetime.strftime(time_datetime[0], "%Y%m%d")
    # station level
    codes = list(stn_loc_idx.keys())
    rows = np.array([stn_loc_idx[c][0] for c in codes])
    cols = np.array([stn_loc_idx[c][1] for c in codes])
    pm_matrix = cur_pm[:, rows, cols]
    n_stns = len(codes)
    df = pd.DataFrame({
        "Time": np.repeat(time_datetime, n_stns),  # ← fix: repeat not tile
        "PM25_UFSAQM": pm_matrix.ravel(),  # ← fix: ravel("C")
        "lead_time": np.repeat(np.arange(1, 25), n_stns),
        "siteCode": np.tile(codes, 24),  # ← fix: tile not repeat
    })
    df["siteCode"] = df["siteCode"].astype(str).str.zfill(9)
    df.to_csv(os.path.join(output_dir, "site_level_aqm_%s.csv" % date_str), index=False)