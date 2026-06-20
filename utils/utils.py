from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import numpy as np

def doy_to_datetime(base, d):
    """Convert fractional day-of-year to datetime (1-based)."""
    return base + timedelta(days=float(d) - 1)


def build_kdtree(X_ctr, Y_ctr):
    """
    Build a KDTree from 2D grid centers.
    """
    points = np.column_stack((X_ctr.ravel(), Y_ctr.ravel()))
    tree = cKDTree(points)
    return tree


def findSpatialIndex_kdtree(fire_x, fire_y, tree, X_ctr_shape):
    """
    Use KDTree to find the closest index of a fire location in the grid.
    :param fire_x: X coordinate of the fire
    :param fire_y: Y coordinate of the fire
    :param tree: KDTree built from X_ctr and Y_ctr
    :param X_ctr_shape: original shape of X_ctr for unraveling
    :return: x_idx, y_idx
    """
    dist, idx_flat = tree.query([fire_x, fire_y])
    x_idx, y_idx = np.unravel_index(idx_flat, X_ctr_shape)
    return x_idx, y_idx

# https://www.airnow.gov/aqi/aqi-basics/, left include, right not include
# Convention: [lo, hi)  — left inclusive, right exclusive.
# Hazardous has no upper bound (pm25 >= 225.5).

AQI_CATEGORIES = {
    "Good":                    {"range": (0,   50),  "pm25": (0.0,   9.0),   "color": "#00E400", "label": "Good"},
    "Moderate":                {"range": (50,  100), "pm25": (9.0,   35.5),  "color": "#FFFF00", "label": "Moderate"},
    "Unhealthy for Sensitive": {"range": (100, 150), "pm25": (35.5,  55.5),  "color": "#FF7E00", "label": "USG"},
    "Unhealthy":               {"range": (150, 200), "pm25": (55.5,  125.5), "color": "#FF0000", "label": "Unhealthy"},
    "Very Unhealthy":          {"range": (200, 300), "pm25": (125.5, 225.5), "color": "#8F3F97", "label": "Very Unhealthy"},
    "Hazardous":               {"range": (300, np.inf), "pm25": (225.5, np.inf), "color": "#7E0023", "label": "Hazardous"},
}

AQI_ORDER = [
    "Good", "Moderate", "Unhealthy for Sensitive",
    "Unhealthy", "Very Unhealthy", "Hazardous",
]


def assign_aqi_category(pm25_array):
    """
    Vectorised assignment of AQI category strings for a numpy array or Series.
    Convention: [lo, hi) — left inclusive, right exclusive.
    Hazardous upper bound is np.inf, so no special-case branch is needed.
    Negative values default to 'Good'.

    Returns a numpy array of category name strings (dtype=object).
    """
    pm25_array = np.asarray(pm25_array, dtype=float)
    result = np.full(pm25_array.shape, "Good", dtype=object)
    for cat in AQI_ORDER:
        lo, hi = AQI_CATEGORIES[cat]["pm25"]
        result[(pm25_array >= lo) & (pm25_array < hi)] = cat
    return result


def pm25_to_aqi_category(pm25_value):
    """
    Map a single PM2.5 concentration (µg/m³) to its AQI category name.
    Convention: [lo, hi) — left inclusive, right exclusive.
    Hazardous upper bound is np.inf, so no special-case branch is needed.
    Negative values return 'Good'.
    """
    for cat in AQI_ORDER:
        lo, hi = AQI_CATEGORIES[cat]["pm25"]
        if lo <= pm25_value < hi:
            return cat
    return "Good"


def aqi_category_color(cat_name):
    """Return the hex color string for a given AQI category name."""
    return AQI_CATEGORIES.get(cat_name, {}).get("color", "#CCCCCC")