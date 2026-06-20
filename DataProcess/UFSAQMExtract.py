import os
import netCDF4 as nc
import numpy as np
import subprocess
import tempfile
import pygrib
import pyproj
from datetime import datetime, timedelta

start_time = datetime(2020, 12, 31)
end_time   = datetime(2020, 12, 31)
x_s, x_e, y_s, y_e = 163, 770, 312, 1310

output_dir  = "/home/zli/radsense/Evaluation"
static_path = os.path.join(output_dir, "aqm_static.npz")
os.makedirs(output_dir, exist_ok=True)
static_saved = False

cur_time = start_time
while cur_time <= end_time:
    if cur_time <= datetime(2021, 7, 19):
        version = "AQMv5"
    elif cur_time <= datetime(2024, 5, 13):
        version = "AQMv6"
    else:
        version = "AQMv7"

    date_str  = cur_time.strftime("%Y%m%d")
    grib_path = "/home/zli/radsense/NAQFC_PM25/%s/aqm.%s.t06z.ave_1hr_pm25.grib2" % (version, date_str)
    out_path  = os.path.join(output_dir, "aqm_%s.npz" % date_str)

    if not os.path.exists(grib_path):
        print("File not found: %s" % grib_path)
        cur_time += timedelta(days=1)
        continue

    tmp_nc = tempfile.NamedTemporaryFile(suffix=".nc", delete=False, dir="/home/zli/radsense/Evaluation").name
    try:
        subprocess.run(
            ["wgrib2", grib_path, "-match", "PMTF", "-netcdf", tmp_nc],
            check=True,
        )
        ds = nc.Dataset(tmp_nc, "r")

        pm25 = np.ma.filled(ds["PMTF_1sigmalevel"][:], np.nan)   # (T, H, W)
        pm25 = pm25[:, x_s:x_e, y_s:y_e]
        time = ds["time"][:].astype(np.int64)                      # seconds since epoch

        if not static_saved:
            lat = np.array(ds["latitude"][x_s:x_e, y_s:y_e])
            lon = np.array(ds["longitude"][x_s:x_e, y_s:y_e])
            if lon.max() > 180:
                lon = lon - 360

            # extract real projection params from grib2
            grbs = pygrib.open(grib_path)
            grb  = grbs.select()[0]
            p    = grb.projparams
            grbs.close()

            crs = pyproj.Proj(
                "+proj=" + p["proj"]      +
                " +a="     + str(p["a"])     +
                " +b="     + str(p["b"])     +
                " +lat_1=" + str(p["lat_1"]) +
                " +lat_2=" + str(p["lat_2"]) +
                " +lat_0=" + str(p["lat_0"]) +
                " +lon_0=" + str(p["lon_0"])
            )
            x, y = crs(lon, lat)

            np.savez(
                static_path,
                lat=lat, lon=lon,
                x=x, y=y,
                crs_proj4=np.array(crs.srs),
            )
            print("Static grid saved -> %s  shape=%s" % (static_path, lat.shape))
            static_saved = True

        np.savez(out_path, PM25_UFSAQM=pm25, time=time)
        print("%s  shape=%s  max=%.2f" % (date_str, pm25.shape, np.nanmax(pm25)))

        ds.close()
    except Exception as e:
        print("Error processing %s: %s" % (date_str, e))
    finally:
        if os.path.exists(tmp_nc):
            os.remove(tmp_nc)

    cur_time += timedelta(days=1)