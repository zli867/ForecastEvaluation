import numpy as np
import shapefile                      # pyshp
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

state_shp  = "/Volumes/Shield/HealthAnalysis/data/geo/US/cb_2018_us_state_20m.shp"
county_shp = "/Volumes/Shield/HealthAnalysis/data/geo/County/cb_2018_us_county_20m.shp"

# WGS84 (lon/lat) -> US Albers Equal Area, meters
to_albers = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform


def read_areas(path):
    """Return list of (state_fips, area_km2) for every feature in the shapefile."""
    sf = shapefile.Reader(path)
    fields = [f[0] for f in sf.fields[1:]]
    rows = []
    for sr in sf.shapeRecords():
        rec = dict(zip(fields, sr.record))
        geom = shape(sr.shape.__geo_interface__)
        area_km2 = transform(to_albers, geom).area / 1e6
        rows.append((rec.get("STATEFP", ""), area_km2))
    return rows


def summarize(rows, label):
    a = np.array([r[1] for r in rows])
    print(f"{label:>9}: n={len(a):>4}  mean={a.mean():>10,.0f} km^2  "
          f"median={np.median(a):>10,.0f} km^2 avg resolution={np.sqrt(a.mean()):.2f} km")


if __name__ == "__main__":
    states   = read_areas(state_shp)
    counties = read_areas(county_shp)

    excl = ("02", "15", "72")
    states2   = [r for r in states   if r[0] not in excl]
    counties2 = [r for r in counties if r[0] not in excl]
    summarize(states2,   "States")
    summarize(counties2, "Counties")