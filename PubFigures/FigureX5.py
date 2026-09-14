import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.ops import transform, unary_union
from pyproj import CRS, Transformer

from utils.geoutil import StatePolygon, interest_domain_bound


epa_regions = {
    1: {"name": "New England",
        "states": ["Connecticut", "Maine", "Massachusetts",
                   "New Hampshire", "Rhode Island", "Vermont"]},
    2: {"name": "New York",
        "states": ["New Jersey", "New York"]},
    3: {"name": "Mid-Atlantic",
        "states": ["Delaware", "District of Columbia", "Maryland",
                   "Pennsylvania", "Virginia", "West Virginia"]},
    4: {"name": "Southeast",
        "states": ["Alabama", "Florida", "Georgia", "Kentucky",
                   "Mississippi", "North Carolina", "South Carolina",
                   "Tennessee"]},
    5: {"name": "Great Lakes",
        "states": ["Illinois", "Indiana", "Michigan", "Minnesota",
                   "Ohio", "Wisconsin"]},
    6: {"name": "South Central",
        "states": ["Arkansas", "Louisiana", "New Mexico", "Oklahoma",
                   "Texas"]},
    7: {"name": "Midwest",
        "states": ["Iowa", "Kansas", "Missouri", "Nebraska"]},
    8: {"name": "Mountains and Plains",
        "states": ["Colorado", "Montana", "North Dakota",
                   "South Dakota", "Utah", "Wyoming"]},
    9: {"name": "Pacific Southwest",
        "states": ["Arizona", "California", "Nevada"]},
    10: {"name": "Pacific Northwest",
         "states": ["Idaho", "Oregon", "Washington"]},
}

region_colors = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#8c564b",
    7: "#e377c2",
    8: "#7f7f7f",
    9: "#bcbd22",
    10: "#17becf",
}

# Lambert Conformal Conic projection centered over CONUS.
# Census state shapefiles generally use NAD83 geographic coordinates.
lcc_crs = CRS.from_proj4(
    "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 "
    "+lon_0=-96 +datum=NAD83 +units=m +no_defs"
)

project_to_lcc = Transformer.from_crs(
    "EPSG:4269",       # NAD83 longitude/latitude
    lcc_crs,
    always_xy=True
).transform

fig, ax = plt.subplots(figsize=(8, 5))
all_state_polygons_lcc = []

for region_id, region_info in epa_regions.items():

    # Read states in longitude/latitude, then project each geometry to Lambert.
    state_polygons = StatePolygon(region_info["states"])
    state_polygons_lcc = [
        transform(project_to_lcc, geom)
        for geom in state_polygons
    ]

    all_state_polygons_lcc.extend(state_polygons_lcc)
    patches = []

    for geom in state_polygons_lcc:
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = geom.geoms
        else:
            continue

        # Each state is drawn independently, preserving state boundaries.
        for poly in polygons:
            patches.append(
                MplPolygon(
                    np.asarray(poly.exterior.coords),
                    closed=True
                )
            )

    ax.add_collection(
        PatchCollection(
            patches,
            facecolor=region_colors[region_id],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.85,
            zorder=1,
        )
    )

    # Union is used only to place one label per EPA region.
    region_geometry = unary_union(state_polygons_lcc)
    label_point = region_geometry.representative_point()

    ax.text(
        label_point.x,
        label_point.y,
        str(region_id),
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="black",
        zorder=2,
    )

# Determine map bounds in projected coordinates.
x_min, x_max, y_min, y_max = interest_domain_bound(
    all_state_polygons_lcc,
    tol=0.03,
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal")
ax.set_axis_off()

ax.set_title(
    "U.S. EPA Regions across CONUS",
    fontsize=14,
    fontweight="bold",
)

plt.tight_layout()
plt.show()

# fig.savefig("EPA_regions_CONUS_lambert.png", dpi=300, bbox_inches="tight")