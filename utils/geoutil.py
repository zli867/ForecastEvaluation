import fiona
import numpy as np
from shapely.geometry import shape
from shapely.ops import transform


def StatePolygon(state_name_list):
    """
    List of States:
        Maryland, Iowa, Delaware, Ohio, Pennsylvania, Nebraska, Washington, Puerto Rico, Alabama,
        Arkansas, New Mexico, Texas, California, Kentucky, Georgia, Wisconsin, Oregon,
        Missouri, Virginia, Tennessee, Louisiana, New York, Michigan, Idaho, Florida,
        Alaska, Illinois, Montana, Minnesota, Indiana, Massachusetts, Kansas, Nevada,
        Vermont, Connecticut, New Jersey, District of Columbia, North Carolina, Utah,
        North Dakota, South Carolina, Mississippi, Colorado, South Dakota, Oklahoma,
        Wyoming, West Virginia, Maine, Hawaii, New Hampshire, Arizona, Rhode Island
    :param state_name: input the state name in the list
    :return: polygon of the state
    """
    us_shape_name = "/Users/zongrunli/Desktop/ForecastEvaluation/data/geo/US/cb_2018_us_state_20m.shp"
    us_states = fiona.open(us_shape_name)
    state_geo = []
    for state_name in state_name_list:
        for us_state in us_states:
            cur_name = us_state['properties']['NAME']
            if cur_name == state_name:
                state_geo.append(shape(us_state['geometry']))
    return state_geo


def FIPS_list(state_name_list):
    fips_list = []
    fips_dict = {'Alaska': '02', 'Alabama': '01', 'Arkansas': '05', 'Arizona': '04', 'California': '06', 'Colorado': '08',
                 'Connecticut': '09', 'District of Columbia': '11', 'Delaware': '10', 'Florida': '12', 'Georgia': '13',
                 'Hawaii': '15', 'Iowa': '19', 'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Kansas': '20',
                 'Kentucky': '21', 'Louisiana': '22', 'Massachusetts': '25', 'Maryland': '24', 'Maine': '23',
                 'Michigan': '26', 'Minnesota': '27', 'Missouri': '29', 'Mississippi': '28', 'Montana': '30',
                 'North Carolina': '37', 'North Dakota': '38', 'Nebraska': '31', 'New Hampshire': '33',
                 'New Jersey': '34', 'New Mexico': '35', 'Nevada': '32', 'New York': '36', 'Ohio': '39',
                 'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42', 'Puerto Rico': '72', 'Rhode Island': '44',
                 'South Carolina': '45', 'South Dakota': '46', 'Tennessee': '47', 'Texas': '48', 'Utah': '49',
                 'Virginia': '51', 'Vermont': '50', 'Washington': '53', 'Wisconsin': '55', 'West Virginia': '54',
                 'Wyoming': '56'}
    for state_name in state_name_list:
        fips_list.append(fips_dict[state_name])
    return fips_list


def CountyPolygon(state_name_list):
    fips_list = FIPS_list(state_name_list)
    county_shape_name = "/Users/zongrunli/Desktop/ForecastEvaluation/data/geo/County/cb_2018_us_county_20m.shp"
    us_counties = fiona.open(county_shape_name)
    county_ids = []
    polygons = []
    for us_county in us_counties:
        state_fips = us_county['properties']["STATEFP"]
        county_fips = us_county['properties']["COUNTYFP"]
        county_id = state_fips + county_fips
        if state_fips in fips_list:
            county_ids.append(county_id)
            polygons.append(shape(us_county['geometry']))
    return county_ids, polygons


def interest_domain_bound(polygon_list, tol=0.01, crs=None):
    if crs is None:
        x_bound = []
        y_bound = []
        for cur_poly in polygon_list:
            cur_bound = cur_poly.bounds
            x_bound.append(cur_bound[0])
            x_bound.append(cur_bound[2])
            y_bound.append(cur_bound[1])
            y_bound.append(cur_bound[3])
        x_bound = np.array(x_bound)
        y_bound = np.array(y_bound)
        x_min, x_max, y_min, y_max = np.min(x_bound), np.max(x_bound), np.min(y_bound), np.max(y_bound)
        x_min = min([(1 - tol) * x_min, (1 + tol) * x_min])
        x_max = max([(1 - tol) * x_max, (1 + tol) * x_max])
        y_min = min([(1 - tol) * y_min, (1 + tol) * y_min])
        y_max = max([(1 - tol) * y_max, (1 + tol) * y_max])
    else:
        x_bound = []
        y_bound = []
        for cur_poly in polygon_list:
            cur_poly = transform(crs, cur_poly)
            cur_bound = cur_poly.bounds
            x_bound.append(cur_bound[0])
            x_bound.append(cur_bound[2])
            y_bound.append(cur_bound[1])
            y_bound.append(cur_bound[3])
        x_bound = np.array(x_bound)
        y_bound = np.array(y_bound)
        x_min, x_max, y_min, y_max = np.min(x_bound), np.max(x_bound), np.min(y_bound), np.max(y_bound)
        x_min = min([(1 - tol) * x_min, (1 + tol) * x_min])
        x_max = max([(1 - tol) * x_max, (1 + tol) * x_max])
        y_min = min([(1 - tol) * y_min, (1 + tol) * y_min])
        y_max = max([(1 - tol) * y_max, (1 + tol) * y_max])
    return x_min, x_max, y_min, y_max