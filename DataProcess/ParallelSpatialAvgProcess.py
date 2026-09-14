import pickle
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from utils.utils import build_kdtree, findSpatialIndex_kdtree
import pyproj

# ── Paths & config ──────────────────────────────────────────────────────────
static_file   = "/home/zli/radsense/Evaluation/static/aqm_static.npz"
file_folder   = "/home/zli/radsense/Evaluation/values"
stn_locs_file = "/home/zli/radsense/Evaluation/static/stn_info.pkl"
obs_file      = "/home/zli/radsense/Evaluation/obs/PM25_obs_2021_2024.csv"
output_dir    = "/home/zli/radsense/Evaluation/resolutions"

RESOLUTIONS = [2, 3, 4, 5, 6, 8, 10, 20, 30, 50, 80, 100]
MODEL_VAR   = "PM25_UFSAQM"
OBS_VAR     = "PM25"
SUBSETS     = ["All", "Smoke", "Non-smoke"]      # NEW

os.makedirs(output_dir, exist_ok=True)

with open(stn_locs_file, "rb") as file:
    site_dict = pickle.load(file)

static_info = np.load(static_file)
lon, lat = static_info["lon"], static_info["lat"]
ny, nx = lon.shape
x, y = static_info["x"], static_info["y"]
crs_proj4 = static_info["crs_proj4"].item()
crs = pyproj.Proj(crs_proj4)
dx = (x[0, 1] - x[0, 0]) / 1000   # native cell size (km)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — match every station to its NEAREST NATIVE model cell (no regridding)
# ════════════════════════════════════════════════════════════════════════════
ntree = build_kdtree(lon, lat)
lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

stn_native = {}
for k in site_dict:
    slon, slat = site_dict[k]["coords"]
    if lon_min <= slon <= lon_max and lat_min <= slat <= lat_max:
        r, c = findSpatialIndex_kdtree(slon, slat, ntree, lon.shape)
        stn_native[str(k).zfill(9)] = (int(r), int(c))

codes = list(stn_native.keys())
nrows = np.array([stn_native[k][0] for k in codes])
ncols = np.array([stn_native[k][1] for k in codes])
n_stns = len(codes)
print(f"{n_stns} stations matched to native model cells")

# ── Build per-station (obs, sim) pairs at native resolution ─────────────────
filenames = sorted(os.listdir(file_folder))
model_frames = []
for filename in filenames:
    cur_values = np.load(os.path.join(file_folder, filename), mmap_mode="r")
    cur_time = cur_values["time"][0:24]
    time_datetime = pd.to_datetime(cur_time, unit="s").to_pydatetime().tolist()
    cur_pm = cur_values["PM25_UFSAQM"][0:24, :, :]
    pm_matrix = cur_pm[:, nrows, ncols]
    model_frames.append(pd.DataFrame({
        "Time": np.repeat(time_datetime, n_stns),
        "siteCode": np.tile(codes, 24),
        MODEL_VAR: pm_matrix.ravel(),
        "lead_time": np.repeat(np.arange(1, 25), n_stns),
    }))
model_stn = pd.concat(model_frames, ignore_index=True)

# 1b. observations
obs_df = pd.read_csv(obs_file, parse_dates=["Time"], dtype={"siteCode": str})
obs_df["siteCode"] = obs_df["siteCode"].astype(str).str.zfill(9)

pairs = model_stn.merge(obs_df[["siteCode", "Time", OBS_VAR]],
                        on=["siteCode", "Time"], how="inner")

# ── NEW: smoke flag at the native station-hour level ────────────────────────
# key smoke_dates by the SAME zero-filled 9-digit code used in `pairs`
smoke_day_sets = {
    str(c).zfill(9): {d.date() for d in site_dict[c].get("smoke_dates", [])}
    for c in site_dict
}
pairs["isSmokeDay"] = [
    code in smoke_day_sets and t.date() in smoke_day_sets[code]
    for code, t in zip(pairs["siteCode"], pairs["Time"])
]

# ── NEW: the three subsets, separated on the OBSERVATIONS' smoke flag ────────
subset_pairs = {
    "All":       pairs,
    "Smoke":     pairs[pairs["isSmokeDay"]],
    "Non-smoke": pairs[~pairs["isSmokeDay"]],
}
for name, df in subset_pairs.items():
    print(f"  subset {name:10s}: {len(df):,} native pairs")

# station coords aligned to `codes`
stn_lons = np.array([site_dict[k]["coords"][0] for k in stn_native])
stn_lats = np.array([site_dict[k]["coords"][1] for k in stn_native])
slon_map = dict(zip(codes, stn_lons))
slat_map = dict(zip(codes, stn_lats))


def coarse_centers(arr2d, n):
    """Block-mean a 2D lat/lon array into coarse-cell center coordinates."""
    nyc, nxc = ny // n, nx // n
    a = arr2d[:nyc * n, :nxc * n].reshape(nyc, n, nxc, n)
    return np.nanmean(a, axis=(1, 3))


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — per resolution: define coarse grid, rematch stations to coarse cells
#          (ONCE), then aggregate each subset (All / Smoke / Non-smoke)
# ════════════════════════════════════════════════════════════════════════════
def process_resolution(n):
    print(f"\n=== Resolution {n}×{n}  (~{n * dx:.0f} km) ===")

    # 2a. coarse grid + KD-tree on its centers
    clon = coarse_centers(lon, n)
    clat = coarse_centers(lat, n)
    ctree = build_kdtree(clon, clat)
    cshape = clon.shape

    # 2b. station -> coarse cell (geometry only; identical for every subset)
    stn_cell = {}
    for k in codes:
        r, c = findSpatialIndex_kdtree(slon_map[k], slat_map[k], ctree, cshape)
        stn_cell[k] = f"{int(r)}_{int(c)}"

    # 2c. aggregate each subset separately
    for subset_name in SUBSETS:
        p = subset_pairs[subset_name].copy()
        if len(p) == 0:
            print(f"  [{subset_name}] empty — skipping")
            continue
        p["cell_id"] = p["siteCode"].map(stn_cell)

        agg = (p.groupby(["cell_id", "Time", "lead_time"])
                 .agg(**{
                     MODEL_VAR: (MODEL_VAR, "mean"),
                     OBS_VAR:   (OBS_VAR, "mean"),
                     "n_sites": ("siteCode", "nunique"),
                 })
                 .reset_index())
        agg["resolution"] = n * dx
        agg["subset"]     = subset_name
        out_path = os.path.join(
            output_dir,
            "merged_cell_level_nearest_%s_%d.csv" % (subset_name.lower().replace("-", ""), n))
        agg.to_csv(out_path, index=False)
        print(f"  [{subset_name:10s}] {p['siteCode'].nunique()} stns -> "
              f"{agg['cell_id'].nunique()} cells, {len(agg)} pairs -> {out_path}")


# run all resolutions in parallel; each job now writes 3 CSVs (one per subset)
Parallel(n_jobs=-1)(delayed(process_resolution)(n) for n in RESOLUTIONS)