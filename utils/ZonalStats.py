import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform
from utils.utils import findSpatialIndex_kdtree
from scipy.sparse import csr_matrix

def create_grid_cell(center_x, center_y, dx, dy):
    corners = [(center_x - dx / 2, center_y + dy / 2), (center_x + dx / 2, center_y + dy / 2),
               (center_x + dx / 2, center_y - dy / 2), (center_x - dx / 2, center_y - dy / 2)]
    rectangle = Polygon(corners)
    return rectangle



def search_idx(polygon, tree, X_ctr_shape):
    x_min, y_min, x_max, y_max = polygon.bounds
    corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
    y_range_idx = []
    x_range_idx = []
    for corner in corners:
        corner_x, corner_y = findSpatialIndex_kdtree(corner[0], corner[1], tree, X_ctr_shape)
        y_range_idx.append(corner_y)
        x_range_idx.append(corner_x)
    start_y_idx = min(y_range_idx)
    end_y_idx = max(y_range_idx)
    start_x_idx = min(x_range_idx)
    end_x_idx = max(x_range_idx)
    return start_x_idx, end_x_idx, start_y_idx, end_y_idx


def polygon_statistics(coords_info, lonlat_polygons):
    """
    Polygons must in the search region
    :param coords_info: grid info, crs, dx, dy, X_ctr, Y_ctr
    :param lonlat_polygons: list of polygons in lon lat coordinate system
    :return: list: [[(x_idx, y_idx, portion for polygon_1), ...], [(x_idx, y_idx, portion for polygon_n), ...]]
    """
    res = []
    cell_area = coords_info["DX"] * coords_info["DY"]
    m, n = coords_info["X_ctr"].shape
    tree = coords_info["tree"]
    for lonlat_polygon in lonlat_polygons:
        portion_list = []
        trans_polygon = transform(coords_info["crs"], lonlat_polygon)
        start_x_idx, end_x_idx, start_y_idx, end_y_idx = search_idx(trans_polygon, tree, coords_info["X_ctr"].shape)
        start_x_idx = np.max([0, start_x_idx - 1])
        start_y_idx = np.max([0, start_y_idx - 1])
        end_x_idx = np.min([m, end_x_idx + 1])
        end_y_idx = np.min([n, end_y_idx + 1])
        for i in range(start_x_idx, end_x_idx + 1):
            for j in range(start_y_idx, end_y_idx + 1):
                grid_polygon = create_grid_cell(coords_info["X_ctr"][i, j], coords_info["Y_ctr"][i, j], coords_info["DX"], coords_info["DY"])
                is_intersection = trans_polygon.intersects(grid_polygon)
                if is_intersection:
                    portion_list.append((i, j, trans_polygon.intersection(grid_polygon).area / cell_area))
        res.append(portion_list)
    return res


def build_weight_matrix(portion_lists, grid_shape):
    m, n = grid_shape
    n_counties = len(portion_lists)

    rows, cols, data = [], [], []
    for k, portion_list in enumerate(portion_lists):
        if not portion_list:
            continue
        idxs = np.array([i * n + j for i, j, _ in portion_list])
        fracs = np.array([frac for _, _, frac in portion_list])
        # normalize so weights sum to 1 → weighted average
        frac_sum = fracs.sum()
        if frac_sum > 0:
            fracs /= frac_sum
        rows.extend([k] * len(idxs))
        cols.extend(idxs)
        data.extend(fracs)

    W = csr_matrix((data, (rows, cols)), shape=(n_counties, m * n))
    return W

def zonal_avg(values, W):
    """
    values : (T, m, n)
    W      : (n_counties, m*n) sparse
    Returns: (T, n_counties)
    """
    T = values.shape[0]
    flat = values.reshape(T, -1)        # (T, m*n)
    return (W @ flat.T).T               # (n_counties, T).T → (T, n_counties)