import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import pyproj
from datetime import datetime, timedelta
import netCDF4 as nc
from scipy.spatial import cKDTree
from utils.metrics import remove_nan_values, COD
from utils.Visualizations import plotComparisonIntercept_fast
import pickle
from sklearn.metrics import r2_score

def columnCheck(pollutant, dataframe):
    """

    :param dataframe: dataframe read from observation files
    :return: True: the data is valid, False: the data is invalid
    """
    valid = True
    required_fields = ["Time", "siteCode", "Latitude", "Longitude", pollutant]
    for required_field in required_fields:
        valid = valid and (required_field in dataframe.columns)
    # check datetime format
    return valid


def extractOBSHourly(pollutant, filename, cmaq_dict):
    """
    We firstly remove the observations out of the CMAQ boundary, remove the date we do not have any CMAQ simulations (it
    means date range of observation <= date range of CMAQ), the data points which have negative pollutant concentration.
    Then, the concentration is rearraged into the format: (timeSeries * site). Notice that, if we calculate
    np.nanmean(conc, axis=0) or np.nanmean(conc, axis=1), there will be no nan values.
    :param pollutant: pollutant name in the csv file
    :param filename: csv filename which includes the observation information
    :param cmaq_dict: the extractMultipleCMAQ returned results
    :return: a list of a dictionary. The dictionary contains daily, yearly observation data
    """
    df = pd.read_csv(filename, dtype={"siteCode": str, "Latitude": float,
                                      "Longitude": float, pollutant: float},
                     parse_dates=['Time'], date_format='%Y-%m-%d-%H:%M')
    column_check = columnCheck(pollutant, df)
    if not column_check:
        sys.exit("The observation data format is not correct, check the columns.")

    # Remove the negative pollutant data (really simple quality check, quality check should be
    # user's responsibility)
    df[df[pollutant] < 0] = np.nan
    df = df.dropna()
    df = df.reset_index(drop=True)

    crs, cmaq_time = cmaq_dict["crs"], cmaq_dict["time"]

    # Add X, Y coordinates
    loc_x, loc_y = crs(df["Longitude"].to_numpy(), df["Latitude"].to_numpy(), inverse=False)
    df["X"], df["Y"] = loc_x, loc_y

    # Filter by CMAQ boundaries and time
    x0, x1 = cmaq_dict["X_bdry"]
    y0, y1 = cmaq_dict["Y_bdry"]
    df = df[(df["X"] >= x0) & (df["X"] <= x1) & (df["Y"] >= y0) & (df["Y"] <= y1)]

    # Convert CMAQ time to set for fast lookup
    cmaq_times = pd.to_datetime(cmaq_dict["time"])
    df["Time"] = pd.to_datetime(df["Time"])
    df = df[(df["Time"] >= cmaq_times.min()) & (df["Time"] <= cmaq_times.max())]

    df.sort_values(by="Time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create a mapping from datetime to index
    time_idx_map = {t: i for i, t in enumerate(cmaq_times)}

    # Group by siteCode and build the dictionary
    obs_combined_dict = {}
    for site, group in df.groupby("siteCode"):
        conc = np.full(len(cmaq_times), np.nan)
        for t, val in zip(group["Time"], group[pollutant]):
            idx = time_idx_map.get(t)
            if idx is not None:
                conc[idx] = val
        obs_combined_dict[site] = {
            "Lat": group["Latitude"].iloc[0],
            "Lon": group["Longitude"].iloc[0],
            "X": group["X"].iloc[0],
            "Y": group["Y"].iloc[0],
            "time": cmaq_dict["time"],
            "conc": conc
        }
    return obs_combined_dict


def AQMGridInfo(cmaq_ds):
    time_data = cmaq_ds['TFLAG'][:]
    lat_1 = cmaq_ds.getncattr('P_ALP')
    lat_2 = cmaq_ds.getncattr('P_BET')
    lat_0 = cmaq_ds.getncattr('YCENT')
    lon_0 = cmaq_ds.getncattr('XCENT')
    crs = pyproj.Proj("+proj=lcc +a=6370000.0 +b=6370000.0 +lat_1=" + str(lat_1)
                      + " +lat_2=" + str(lat_2) + " +lat_0=" + str(lat_0) +
                      " +lon_0=" + str(lon_0))
    xcell = cmaq_ds.getncattr('XCELL')
    ycell = cmaq_ds.getncattr('YCELL')
    xorig = cmaq_ds.getncattr('XORIG')
    yorig = cmaq_ds.getncattr('YORIG')

    ncols = cmaq_ds.getncattr('NCOLS')
    nrows = cmaq_ds.getncattr('NROWS')

    # > for X, Y cell centers
    x_center_range = np.linspace(xorig + xcell / 2, (xorig + xcell / 2) + xcell * (ncols - 1), ncols)
    y_center_range = np.linspace(yorig + ycell / 2, (yorig + ycell / 2) + ycell * (nrows - 1), nrows)

    Xcenters, Ycenters = np.meshgrid(x_center_range, y_center_range)

    # > for X, Y cell boundaries (i.e., cell corners)
    x_bound_range = np.linspace(xorig, xorig + xcell * ncols, ncols + 1)
    y_bound_range = np.linspace(yorig, yorig + ycell * nrows, nrows + 1)

    Xbounds, Ybounds = np.meshgrid(x_bound_range, y_bound_range)

    x_max = np.max(Xbounds)
    x_min = np.min(Xbounds)
    y_max = np.max(Ybounds)
    y_min = np.min(Ybounds)

    cmaq_time_array = []

    # Due to errors in AQM
    time_data_tmp = time_data[0, 0, :]
    time_str = str(time_data_tmp[0]) + str(time_data_tmp[1]).rjust(6, '0')
    parsed = datetime.strptime(time_str, '%Y%j%H%M%S')
    cmaq_time_array.extend(parsed + timedelta(hours=t) for t in range(time_data.shape[0]))

    lon_ctr, lat_ctr = crs(Xcenters, Ycenters, inverse=True)
    lon_bdry_ctr, lat_bdry_ctr = crs(Xbounds, Ybounds, inverse=True)
    lon_max = np.max(lon_bdry_ctr)
    lon_min = np.min(lon_bdry_ctr)
    lat_max = np.max(lat_bdry_ctr)
    lat_min = np.min(lat_bdry_ctr)

    res_dict = {
        "crs": crs, "X": Xcenters, "Y": Ycenters,
        "time": cmaq_time_array,
        "Lat": lat_ctr, "Lon": lon_ctr,
        "XCELL": xcell, "YCELL": ycell,
        "X_bdry": [x_min, x_max], "Y_bdry": [y_min, y_max],
        "Lat_bdry": [lat_min, lat_max], "Lon_bdry": [lon_min, lon_max]
    }
    return res_dict


def extractMultipleCMAQ(pollutant, filenames):
    """
    :param pollutant: the variable name need to do the data fusion
    :param filenames: CMAQ file names
    :return: a list of a dictionary. The dictionary contains daily CMAQ data and unit of the variable
    """
    # TODO: Test CMAQ files consistency
    global_time = []
    global_conc = []
    cmaq_daily_dict = {}
    for i in range(0, len(filenames)):
        file_data = nc.Dataset(filenames[i])
        cur_cmaq_info = AQMGridInfo(file_data)
        polltant_conc = file_data.variables[pollutant]
        conc_surface_layer = polltant_conc[:][:, 0, :, :]
        global_time.extend(cur_cmaq_info["time"])
        global_conc.append(conc_surface_layer)
        # Initialize static info
        if i == 0:
            for key in cur_cmaq_info.keys():
                if key != "time":
                    cmaq_daily_dict[key] = cur_cmaq_info[key]
            cmaq_daily_dict["var"] = pollutant
    # sort time
    sort_idx = np.argsort(global_time)
    global_time.sort()
    global_conc = np.concatenate(global_conc, axis=0)
    cmaq_daily_dict["time"] = global_time
    cmaq_daily_dict["conc"] = global_conc[sort_idx, :, :]
    return cmaq_daily_dict


def findSpatialIndex(obs_x, obs_y, X_ctr, Y_ctr):
    """
    :param obs_x: X of obs location in CMAQ projection
    :param obs_y: Y of obs location in CMAQ projection
    :param X_ctr: CMAQ grid X center
    :param Y_ctr: CMAQ grid Y center
    :return: x_idx, y_idx which are the fire location in CMAQ grid
    """
    dist = np.sqrt((X_ctr - obs_x) ** 2 + (Y_ctr - obs_y) ** 2)
    x_idx, y_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    return x_idx, y_idx


def CMAQatOBS(cmaq_dict, obs_x, obs_y):
    x_idx, y_idx = findSpatialIndex(obs_x, obs_y, cmaq_dict["X"], cmaq_dict["Y"])
    return np.squeeze(cmaq_dict["conc"][:, x_idx, y_idx])

cmaq_dir = "/Volumes/T9/EQUATES/CMAQ"
cmaq_filenames = glob.glob(os.path.join(cmaq_dir, '*'))
for filename in cmaq_filenames:
    if not filename.endswith(".nc"):
        cmaq_filenames.remove(filename)
obs_filename = "/Volumes/T9/EQUATES/obs/PM25_obs_2019.csv"
cmaq_pollutant = "ATOTIJ"
obs_pollutant = "PM25"

CMAQ_res = extractMultipleCMAQ(cmaq_pollutant, cmaq_filenames)
OBS_res = extractOBSHourly(obs_pollutant, obs_filename, CMAQ_res)

all_cmaq_vals, all_obs_vals = [], []
all_obs_x, all_obs_y = [], []
all_obs_lon, all_obs_lat = [], []
for obs_name in OBS_res.keys():
    cur_obs_x, cur_obs_y = OBS_res[obs_name]["X"], OBS_res[obs_name]["Y"]
    cur_obs_lon, cur_obs_lat = OBS_res[obs_name]["Lon"], OBS_res[obs_name]["Lat"]
    cmaq_conc = CMAQatOBS(CMAQ_res, cur_obs_x, cur_obs_y)
    valid_p, valid_o = remove_nan_values(cmaq_conc, OBS_res[obs_name]["conc"])
    all_cmaq_vals.append(valid_p)
    all_obs_vals.append(valid_o)
all_cmaq_vals, all_obs_vals = np.concatenate(all_cmaq_vals), np.concatenate(all_obs_vals)


# density plot
unit = "$\mu g/m^3$"
fig, ax = plt.subplots(figsize=(7, 7))
txt = plotComparisonIntercept_fast(all_obs_vals, all_cmaq_vals, 1000, ax, fig)
cod = COD(all_cmaq_vals, all_obs_vals)
r2_value = r2_score(y_true=all_obs_vals, y_pred=all_cmaq_vals)
print(r2_value)
ax.set_ylabel("EQUATES Hourly $PM_{2.5}$ (%s)" % unit, fontsize=16)
ax.set_xlabel("Observation Hourly $PM_{2.5}$ (%s)" % unit, fontsize=16)
ax.plot([0, 300], [0, 300], '--k')
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
ax.set_title(txt + ", COD=%.2f" % cod, fontsize=16)
ax.tick_params(axis='both', labelsize=14)  # 'both' = x and y ticks
ax.set_aspect('equal', adjustable='box')
plt.show()