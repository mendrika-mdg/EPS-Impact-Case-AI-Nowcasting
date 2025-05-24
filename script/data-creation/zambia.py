import sys
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import os
from datetime import datetime, timedelta
import snflics
import numpy as np      
from netCDF4 import Dataset              # type: ignore               
from scipy.ndimage import label
from scipy.ndimage import zoom

def prepare_core(file):

    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    try:
        # Open the NetCDF file using a context manager to ensure proper file closure
        with Dataset(file, "r") as data:
            cores = data.variables["cores"][0, :, :]
    except OSError as e:
        raise OSError(f"Error opening NetCDF file: {file}. {e}")
    
    return cores


def update_hour(date_dict, hours_to_add):
    """
    Add hours to a datetime dictionary and return the updated dict and a generated file path.

    Args:
        date_dict (dict): Keys: 'year', 'month', 'day', 'hour', 'minute' as strings, e.g. "01", "23"
        hours_to_add (int): Number of hours to add.

    Returns:
        tuple:
            - dict: Updated datetime dictionary with all fields as zero-padded strings.
            - str: File path in the format YYYY/MM/YYYYMMDDHHMM.nc
    """
    # Parse the original time
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )

    # Add hours
    updated = time_obj + timedelta(hours=hours_to_add)

    # Format updated dictionary
    new_date_dict = {
        "year": f"{updated.year:04d}",
        "month": f"{updated.month:02d}",
        "day": f"{updated.day:02d}",
        "hour": f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}"
    }

    # Generate file path
    file_path = f"{new_date_dict['year']}/{new_date_dict['month']}/{new_date_dict['year']}{new_date_dict['month']}{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}.nc"

    return new_date_dict, file_path


def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Compute Haversine distance between two points or arrays of points.
        Inputs are in degrees. Output is in kilometers.
        
        Supports both scalar and array inputs (NumPy).
        """
        R = 6371.0  # Earth radius in kilometers

        # Convert degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

def generate_storm_feature_header(n):
    """
    Generate a list of header names for n closest storms.
    Format: [year, month, day, hour, minute,
             lat1, ..., latN, lon1, ..., lonN,
             wp1, ..., wpN, size1, ..., sizeN,
             d1, ..., dN, mask1, ..., maskN,
             Cb_{location}_t0, ..., Cb_{location}_t6]
    """
    headers = ['year', 'month', 'day', 'hour', 'minute']
    
    for prefix in ['lat', 'lon', 'wp', 'size', 'd', 'mask']:
        headers.extend([f'{prefix}{i}' for i in range(1, n + 1)])
        
    return headers

# core within the context domain    
def create_storm_database(location_lat, location_lon, x0_lat, x0_lon, mask_in, data_t0, lats, lons):
    """
    Identify storm cores within the context domain and extract features for each core.

    Parameters:
        location_lat (float): Latitude of the reference location (e.g. city centre).
        location_lon (float): Longitude of the reference location.
        x0_lat (np.ndarray): Latitude coordinates of core points.
        x0_lon (np.ndarray): Longitude coordinates of core points.
        mask_in (np.ndarray): Boolean mask to filter valid points inside the context domain.
        data_t0 (netCDF4.Dataset): Dataset at t0 containing 'cores' variable.
        lats, lons (np.ndarray): 2D arrays of grid latitudes and longitudes.

    Returns:
        dict: Dictionary of storm features indexed by core label.
    """
    x0_lat, x0_lon = x0_lat[mask_in], x0_lon[mask_in]
    cores_t0 = data_t0["cores"][0, :, :]

    labeled_array, _ = label(cores_t0 != 0)
    core_labels = np.unique(labeled_array[labeled_array != 0])

    dict_storm_size = {
        core_label: np.sum(labeled_array == core_label) * 9
        for core_label in core_labels
    }

    dict_storm_intensity = {
        core_label: np.mean(cores_t0[labeled_array == core_label])
        for core_label in core_labels
    }

    storm_database = {}
    for lat, lon in zip(x0_lat, x0_lon):
        try:
            x0_y, x0_x = snflics.to_yx(lat, lon, lats, lons)
        except IndexError:
            continue  # Skip points that fall outside the coordinate grid

        lab = labeled_array[x0_y, x0_x]
        distance = haversine_distance(location_lat, location_lon, lat, lon)

        if lab == 0 or lab in storm_database:
            continue

        storm_database[int(lab)] = {
            "lat": lat,
            "lon": lon,
            "wp": dict_storm_intensity[lab],
            "size": dict_storm_size[lab],
            "distance": distance,
            "mask": 1
        }

    return storm_database


def flatten_storm_features(t0, X0_features):
    """
    Given a list of closest storms [(id, dict), ...],
    return a flattened feature vector in the order:
    [year, month, day, hour, minute,
     lat1, lat2, ..., lon1, lon2, ..., wp1, ..., size1, ..., distance1, ..., mask1, ...]
    
    This avoids name conflicts with global grid variables.
    """
    def normalise_lon(lon):
        """Ensure longitude is in the range [-180, 180]."""
        return ((lon + 180) % 360) - 180

    t0 = list(map(int, t0.values()))
    storm_lats = []
    storm_lons = []
    storm_wps = []
    storm_sizes = []
    storm_distances = []
    storm_masks = []

    for _, feature in X0_features:
        storm_lats.append(feature['lat'])
        storm_lons.append(normalise_lon(feature['lon']))
        storm_wps.append(feature['wp'])
        storm_sizes.append(feature['size'])
        storm_distances.append(feature['distance'])
        storm_masks.append(feature['mask'])

    return t0 + storm_lats + storm_lons + storm_wps + storm_sizes + storm_distances + storm_masks


def generate_fictional_storm(city_lat, city_lon,
                              context_lat_min, context_lat_max,
                              context_lon_min, context_lon_max,
                              min_km_buffer=500,
                              max_deg_buffer=4.5):
    """
    Generate a fictional storm at least `min_km_buffer` km from the context domain,
    but not farther than `max_deg_buffer` degrees away.

    Parameters:
        city_lat, city_lon: float
            Reference location (e.g. centre of target domain).
        context_lat_min, context_lat_max: float
            Latitude bounds of the context domain.
        context_lon_min, context_lon_max: float
            Longitude bounds of the context domain.
        min_km_buffer: float
            Minimum required distance from context domain edge.
        max_deg_buffer: float
            Maximum distance (in degrees) allowed from context edge for sampling.

    Returns:
        tuple: (0, dict with lat, lon, wp, size, distance, mask)
    """

    # Sampling range: context domain ± max_deg_buffer
    lat_range = (context_lat_min - max_deg_buffer, context_lat_max + max_deg_buffer)
    lon_range = (context_lon_min - max_deg_buffer, context_lon_max + max_deg_buffer)

    while True:
        lat = np.random.uniform(*lat_range)
        lon = np.random.uniform(*lon_range)

        # Reject if inside context domain
        if context_lat_min <= lat <= context_lat_max and context_lon_min <= lon <= context_lon_max:
            continue

        # Compute distance to each context edge
        d_north = haversine_distance(lat, lon, context_lat_max, lon)
        d_south = haversine_distance(lat, lon, context_lat_min, lon)
        d_east  = haversine_distance(lat, lon, lat, context_lon_max)
        d_west  = haversine_distance(lat, lon, lat, context_lon_min)

        min_edge_dist = min(d_north, d_south, d_east, d_west)

        if min_edge_dist < min_km_buffer:
            continue

        # Accept
        distance_to_city = haversine_distance(lat, lon, city_lat, city_lon)
        return (0, {
            'lat': lat,
            'lon': lon,
            'wp': 0.0,
            'size': 0,
            'distance': distance_to_city,
            'mask': 0
        })


    
def pad_closest_storms(sorted_database, nb_x0, location_lat, location_lon):
    if len(sorted_database) >= nb_x0:
        return sorted_database[:nb_x0]
    else:
        needed = nb_x0 - len(sorted_database)
        sorted_database.extend(
            [generate_fictional_storm(
                city_lat=location_lat, 
                city_lon=location_lon, 
                context_lat_min=CONTEXT_DOMAIN_LAT_MIN, 
                context_lat_max=CONTEXT_DOMAIN_LAT_MAX, 
                context_lon_min=CONTEXT_DOMAIN_LON_MIN,
                context_lon_max=CONTEXT_DOMAIN_LON_MAX
            ) for _ in range(needed)])
        return sorted_database


def resize_core(original_core, target_shape_y, target_shape_x):
    """
    Resize a 2D array using bilinear interpolation to the target shape.

    Parameters:
    - original_core: 2D np.ndarray
    - target_shape_y: int, desired number of rows
    - target_shape_x: int, desired number of columns

    Returns:
    - resized array of shape (target_shape_y, target_shape_x)
    """
    assert original_core.ndim == 2, "Input must be a 2D array"
    assert target_shape_y > 0 and target_shape_x > 0, "Target dimensions must be positive"

    zoom_factors = (
        target_shape_y / original_core.shape[0],
        target_shape_x / original_core.shape[1],
    )

    return zoom(original_core, zoom=zoom_factors, order=1)


YEAR = sys.argv[1]

# Data and output paths
LOCATION_NAME = "zambia"
DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
INPUT_LT0 = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Zambia/64x64/input-{LOCATION_NAME}-t0-{YEAR}.csv"
OUTPUT_PATHS = {
    f"LT{i}": f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Zambia/64x64/output-{LOCATION_NAME}-t{i}-{YEAR}.csv"
    for i in range(7)
}

# Months of interest
MONTHS_OF_INTEREST = {"12", "01", "02"}
all_files = [file for file in snflics.all_files_in(DATA_PATH) if snflics.get_time(file)["month"] in MONTHS_OF_INTEREST and snflics.get_time(file)["year"]==YEAR]
all_files.sort()

# Number of storms to consider (after analysing the whole dataset)
NB_X0 = 2
geodata = np.load("/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/data/geodata/lat_lon_2268_2080.npz")
lons = geodata["lon"][:]
lats = geodata["lat"][:]

# Coordinate of the center of the domain
center_lon = 28.0
center_lat = -13.0 
center_y, center_x = snflics.to_yx(center_lat, center_lon, lats, lons)

# Senegal domain
Domain_y_min, Domain_x_min = 547, 1436
Domain_y_max, Domain_x_max = 970, 1898

# Context domain
CONTEXT_DOMAIN_LAT_MIN = -25
CONTEXT_DOMAIN_LAT_MAX = -1

CONTEXT_DOMAIN_LON_MIN = 15
CONTEXT_DOMAIN_LON_MAX = 41


TARGET_SHAPE_Y, TARGET_SHAPE_X = 64, 64

with open(INPUT_LT0, "a") as feature_file:

    input_header = generate_storm_feature_header(NB_X0)
    feature_file.write(",".join(input_header) + "\n")

    for file_t0 in all_files[:]:

        time_t0 = snflics.get_time(file_t0)

        # file name for lead time from 0 to 6 hours
        files = [DATA_PATH + update_hour(time_t0, h)[1] for h in range(7)]          

        # if all the files exist
        if all(os.path.exists(f) for f in files):
            try:
                core_series = [prepare_core(f) for f in files]
            except OSError:
                continue

            with Dataset(file_t0, "r") as data_t0:
                x0_lat = data_t0["max_lat"][:]
                x0_lon = data_t0["max_lon"][:]

                in_region = (
                    (CONTEXT_DOMAIN_LON_MIN <= x0_lon) & (x0_lon <= CONTEXT_DOMAIN_LON_MAX) &
                    (CONTEXT_DOMAIN_LAT_MIN <= x0_lat) & (x0_lat <= CONTEXT_DOMAIN_LAT_MAX)
                )

                if in_region.any():
                    # database of all identified storms
                    storm_database = create_storm_database(center_lat, center_lon, x0_lat, x0_lon, in_region, data_t0, lats, lons)

                    # taking a certain number of closest storms
                    sorted_database = sorted(storm_database.items(), key=lambda item: item[1]['distance'])
                    X0_features = pad_closest_storms(sorted_database, NB_X0, center_lat, center_lon)
                    input_features = flatten_storm_features(time_t0, X0_features)
                    feature_file.write(",".join(map(str, input_features)) + "\n")

                    Cb_series = []
                    for core in core_series:
                        original_core = core[Domain_y_min:Domain_y_max+1, Domain_x_min:Domain_x_max+1]
                        resized_core = resize_core(original_core, TARGET_SHAPE_Y, TARGET_SHAPE_X)
                        Cb = (resized_core != 0).astype(int).flatten()
                        Cb_series.append(Cb)

                    # Write each Cb (flattened) to its corresponding output CSV
                    for i, Cb in enumerate(Cb_series):
                        output_file_path = OUTPUT_PATHS[f"LT{i}"]
                        with open(output_file_path, "a") as f:
                            f.write(",".join(map(str, Cb)) + "\n")





                



