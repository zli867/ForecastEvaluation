import numpy as np
import os
from scipy.spatial import cKDTree
import pandas as pd


def extractUFSAQM(file_dir, geo_info, base_time="1970-01-01"):
    """
    :param pollutant: the variable name need to do the data fusion
    :param filenames: CMAQ file names
    :return: a list of a dictionary. The dictionary contains daily CMAQ data and unit of the variable
    """
    filenames = sorted([
        os.path.join(file_dir, f)
        for f in os.listdir(file_dir)
        if f.endswith(".npz")
    ])

    all_conc = []
    all_time = []

    for filename in filenames:
        data = np.load(filename)
        cur_conc = data["PM25_UFSAQM"][0:24, :, :]  # select the first 24 hours
        all_conc.append(cur_conc)
        time_seconds = data["time"].astype(np.int64)
        time_datetime = pd.to_datetime(time_seconds, unit="s", origin=base_time)
        all_time.extend(time_datetime[0:24])

    # concatenate along time dimension
    conc_cat = np.concatenate(all_conc, axis=0)

    aqm_dict = {
        "conc": conc_cat,
        "time": all_time
    }
    aqm_dict.update(geo_info)
    aqm_dict["X_bdry"] = [np.min(aqm_dict["X_ctr"]) - 6000, np.max(aqm_dict["X_ctr"]) + 6000]
    aqm_dict["Y_bdry"] = [np.min(aqm_dict["Y_ctr"]) - 6000, np.max(aqm_dict["Y_ctr"]) + 6000]
    return aqm_dict
