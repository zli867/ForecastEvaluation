import pickle
from math import radians

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils.utils import build_kdtree, findSpatialIndex_kdtree
import pyproj

# # ── Paths & config ──────────────────────────────────────────────────────────
# static_file   = "/home/zli/radsense/Evaluation/static/aqm_static.npz"
# file_folder   = "/home/zli/radsense/Evaluation/values"
# stn_locs_file = "/home/zli/radsense/Evaluation/static/stn_info.pkl"
# obs_file      = "/home/zli/radsense/Evaluation/obs/PM25_obs_2021_2024.csv"
# output_dir    = "/home/zli/radsense/Evaluation/resolutions"
static_file   = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/aqm_static.npz"
file_folder   = "/Users/zongrunli/Desktop/SmkFcstEval/data/fcst_data/values"
stn_locs_file = "/Users/zongrunli/Desktop/ForecastEvaluation/data/static/stn_info.pkl"
obs_file      = "/Users/zongrunli/Desktop/ForecastEvaluation/data/obs/PM25_obs_2021_2024.csv"
output_dir    = "/Users/zongrunli/Desktop/ForecastEvaluation/data"

RESOLUTIONS = [10, 50, 100]   # N×N block reduction
MODEL_VAR   = "PM25_UFSAQM"
OBS_VAR     = "PM25"

os.makedirs(output_dir, exist_ok=True)

# ── Load static grid & station locations ────────────────────────────────────
with open(stn_locs_file, 'rb') as file:
    site_dict = pickle.load(file)

static_info = np.load(static_file)
lon, lat = static_info["lon"], static_info["lat"]
ny, nx = lon.shape
x, y = static_info["x"], static_info["y"]
crs_proj4 = static_info["crs_proj4"].item()
crs = pyproj.Proj(crs_proj4)
dx = (x[0, 1] - x[0, 0]) / 1000

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    return 6371.0 * np.arccos(np.clip(
        np.sin(lat1) * np.sin(lat2) +
        np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2), -1, 1))


def coarse_mean(field, n):
    """NaN-aware block-reduce of an (T, ny, nx) field into (T, ny//n, nx//n)."""
    T = field.shape[0]
    nyc, nxc = ny // n, nx // n
    f = field[:, :nyc * n, :nxc * n].reshape(T, nyc, n, nxc, n)
    with np.errstate(invalid="ignore"):
        return np.nanmean(f, axis=(2, 4))   # average over the two N-blocks


def coarse_centers(arr2d, n):
    """Block-mean a 2D lat/lon array into coarse-cell center coordinates."""
    nyc, nxc = ny // n, nx // n
    a = arr2d[:nyc * n, :nxc * n].reshape(nyc, n, nxc, n)
    return np.nanmean(a, axis=(1, 3))       # shape (nyc, nxc)


# ── Pre-list the daily files once ───────────────────────────────────────────
filenames = sorted(os.listdir(file_folder))
obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df["siteCode"] = obs_df["siteCode"].astype(str).str.zfill(9)
lookup_rows = []
merged_list = []

# ════════════════════════════════════════════════════════════════════════════
# RESOLUTION LOOP (outermost)
# ════════════════════════════════════════════════════════════════════════════
for n in RESOLUTIONS:
    print(f"\n=== Resolution {n}×{n} ===")

    # 2a. coarse-cell center coordinates + KD-tree on those centers
    clon = coarse_centers(lon, n)
    clat = coarse_centers(lat, n)
    cshape = clon.shape
    ctree = build_kdtree(clon, clat)
    clon_min, clon_max = np.nanmin(clon), np.nanmax(clon)
    clat_min, clat_max = np.nanmin(clat), np.nanmax(clat)

    # 2b. for each station, find its nearest coarse cell
    stn_cell = {}
    for k in site_dict:
        slon, slat = site_dict[k]["coords"]
        if clon_min <= slon <= clon_max and clat_min <= slat <= clat_max:
            r, c = findSpatialIndex_kdtree(slon, slat, ctree, cshape)
            stn_cell[k] = (int(r), int(c))

    codes = list(stn_cell.keys())
    crows = np.array([stn_cell[k][0] for k in codes])
    ccols = np.array([stn_cell[k][1] for k in codes])
    cell_ids = np.array([f"{r}_{c}" for r, c in zip(crows, ccols)])
    n_stns = len(codes)
    print(f"  {n_stns} stations matched to {len(set(cell_ids))} coarse cells")

    # 3. lookup table for this resolution
    site_to_cell = {}     # for obs aggregation below
    for k, r, c, cid in zip(codes, crows, ccols, cell_ids):
        slon, slat = site_dict[k]["coords"]
        site_to_cell[str(k).zfill(9)] = cid
        lookup_rows.append({
            "siteCode": str(k).zfill(9),
            "resolution": n,
            "stn_lon": slon,
            "stn_lat": slat,
            "cell_row": r,
            "cell_col": c,
            "cell_id": cid,
            "cell_lon": float(clon[r, c]),
            "cell_lat": float(clat[r, c]),
            "match_dist_km": float(haversine_km(slat, slon, clat[r, c], clon[r, c])),
        })

    # 1. process UFS-AQM at this resolution → model cell-level records
    model_frames = []
    for filename in filenames:
        cur_values = np.load(os.path.join(file_folder, filename))
        cur_time = cur_values["time"][0:24]
        time_datetime = pd.to_datetime(cur_time, unit="s").to_pydatetime().tolist()
        cur_pm = cur_values["PM25_UFSAQM"][0:24, :, :]
        coarse_pm = coarse_mean(cur_pm, n)            # (24, nyc, nxc)
        plt.pcolor(clon, clat, coarse_pm[0, :, :])
        plt.colorbar()
        plt.show()
        pm_matrix = coarse_pm[:, crows, ccols]        # (24, n_stns)
        model_frames.append(pd.DataFrame({
            "Time": np.repeat(time_datetime, n_stns),
            MODEL_VAR: pm_matrix.ravel(),
            "lead_time": np.repeat(np.arange(1, 25), n_stns),
            "cell_id": np.tile(cell_ids, 24),
        }))

    model_cell = pd.concat(model_frames, ignore_index=True)
    # 4a. observations → mean within each coarse cell (via this resolution's map)
    obs_n = obs_df.copy()
    obs_n["cell_id"] = obs_n["siteCode"].map(site_to_cell)
    obs_n = obs_n.dropna(subset=["cell_id"])
    obs_cell = (obs_n.groupby(["cell_id", "Time"])[OBS_VAR].mean().reset_index())

    # 4b. merge cell-level model with cell-level obs
    m = model_cell.merge(obs_cell, on=["cell_id", "Time"], how="inner")
    m["resolution"] = n * dx
    merged_path = os.path.join(output_dir, "merged_cell_level_multires_%d.csv" % n)
    m.to_csv(merged_path, index=False)