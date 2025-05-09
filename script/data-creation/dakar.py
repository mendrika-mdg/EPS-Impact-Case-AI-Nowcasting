import sys
sys.path.insert(1, "../../module")
import copy, os
from datetime import datetime, timedelta
import snflics
import numpy as np      
from netCDF4 import Dataset                             
from scipy.ndimage import label


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
    Increases the hour value in the date dictionary and updates the
    dictionary to reflect any overflow into days, months, etc.
    The values in the dictionary are strings formatted as "01", "23", etc.

    Args:
        date_dict (dict): A dictionary with keys 'year', 'month', 'day', 'hour', 'minute'.
                          Values are strings formatted as "01", "23", etc.
        hours_to_add (int): The number of hours to add.

    Returns:
        dict: The updated dictionary with adjusted date and time, formatted as strings.
        str: A file path generated based on the updated date and time.
    """
    # Create a copy of the original dictionary
    new_date_dict = copy.deepcopy(date_dict)

    # Convert string values to integers from the original dictionary
    year = int(date_dict['year'])
    month = int(date_dict['month'])
    day = int(date_dict['day'])
    hour = int(date_dict['hour'])
    minute = int(date_dict['minute'])

    # Create a datetime object
    current_time = datetime(year, month, day, hour, minute)
    
    # Add the specified hours
    updated_time = current_time + timedelta(hours=hours_to_add)
    
    # Update the copied dictionary with the new values, formatted as two-digit strings
    new_date_dict['year'] = f"{updated_time.year:04d}"
    new_date_dict['month'] = f"{updated_time.month:02d}"
    new_date_dict['day'] = f"{updated_time.day:02d}"
    new_date_dict['hour'] = f"{updated_time.hour:02d}"
    new_date_dict['minute'] = f"{updated_time.minute:02d}"
    
    # Create the file path
    file_path = f"{updated_time.year:04d}/{updated_time.month:02d}/{updated_time.year:04d}{updated_time.month:02d}{updated_time.day:02d}{updated_time.hour:02d}{updated_time.minute:02d}.nc"
    
    return new_date_dict, file_path


def extract_box(data, y, x, box_size=5):
    """
    Extracts a square region around a central grid point

    Parameters:
    -----------
    data : np.ndarray
        2D array from which to extract the box (e.g., cores).
    y, x : int
        Central grid indices.
    box_size : int
        Size of the square box (must be odd), e.g., 5 for a 5x5 box.

    Returns:
    --------
    box : np.ndarray
        The extracted subregion.

    """
    assert box_size % 2 == 1, "box_size must be odd"
    
    half = box_size // 2
    y_min = max(0, y - half)
    y_max = min(data.shape[0], y + half + 1)
    x_min = max(0, x - half)
    x_max = min(data.shape[1], x + half + 1)
    
    box = data[y_min:y_max, x_min:x_max]  

    return box

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

def generate_storm_feature_header(n, location):
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
    
    headers.extend([f'Cb_{location}_t{i}' for i in range(7)])
    
    return headers

# core within the context domain    
def create_storm_database(location_lat, location_lon, x0_lat, x0_lon, mask_in, data_t0):

    x0_lat, x0_lon = x0_lat[mask_in], x0_lon[mask_in]

    # Extract storm data
    cores_t0 = data_t0["cores"][0, :, :]

    # Label the storm cores
    labeled_array, _ = label(cores_t0 != 0)
    core_labels = np.unique(labeled_array[labeled_array != 0])

    # Compute storm sizes and mean wavelet power
    dict_storm_size = {
        core_label: np.sum(labeled_array == core_label) * 9             # because the grid spacing is 3 x 3 km
        for core_label in core_labels
    }

    dict_storm_intensity = {
        core_label: np.mean(cores_t0[labeled_array == core_label])
        for core_label in core_labels
    }

    # Assign storm properties to each labeled storm using x0 coordinates
    storm_database = {}
    for lat, lon in zip(x0_lat, x0_lon):
        x0_y, x0_x = snflics.to_yx(lat, lon, lats, lons)
        lab = labeled_array[x0_y, x0_x]
        distance = haversine_distance(location_lat, location_lon, lat, lon)
        
        # Skip points not associated with a storm or already added
        if lab == 0 or lab in storm_database:
            continue

        storm_database[int(lab)] = {
            "lat": lat,
            "lon": lon,
            "wp": (dict_storm_intensity[lab]),
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
     lat1, lat2, ..., lon1, lon2, ..., wp1, wp2, ..., size1, ..., distance1, ..., mask1, ...]
    """
    t0 = list(map(int, t0.values()))
    lats = []
    lons = []
    wps = []
    sizes = []
    distances = []
    masks = []

    for _, feature in X0_features:
        lats.append(feature['lat'])
        lons.append(feature['lon'])
        wps.append(feature['wp'])
        sizes.append(feature['size'])
        distances.append(feature['distance'])
        masks.append(feature['mask'])

    return t0 + lats + lons + wps + sizes + distances + masks


def generate_fictional_storm(city_lat, city_lon):
    """
    Generate a fictional storm located more than 1000 km away from the given city.
    """
    min_dist_km = 1000
    # Extended context domain (centered in Dakar, min distance ~1000 km)
    EXTENDED_CONTEXT_DOMAIN_LAT_MIN, EXTENDED_CONTEXT_DOMAIN_LAT_MAX = 14.69 - 9, 14.69 + 9    # 5.69 to 23.69
    EXTENDED_CONTEXT_DOMAIN_LON_MIN, EXTENDED_CONTEXT_DOMAIN_LON_MAX = -17.45 - 9.3, -17.45 + 9.3  # -26.75 to -8.15

    while True:
        lat = np.random.uniform(EXTENDED_CONTEXT_DOMAIN_LAT_MIN, EXTENDED_CONTEXT_DOMAIN_LAT_MAX)
        lon = np.random.uniform(EXTENDED_CONTEXT_DOMAIN_LON_MIN, EXTENDED_CONTEXT_DOMAIN_LON_MAX)
        distance = haversine_distance(lat, lon, city_lat, city_lon)
        if distance > min_dist_km:
            return (0, {
                'lat': lat,
                'lon': lon,
                'wp': 0.0,
                'size': 0,
                'distance': distance,
                'mask': 0
            })
        
def pad_closest_storms(sorted_database, nb_x0, location_lat, location_lon):
    if len(sorted_database) >= nb_x0:
        return sorted_database[:nb_x0]
    else:
        needed = nb_x0 - len(sorted_database)
        sorted_database.extend([generate_fictional_storm(location_lat, location_lon) for _ in range(needed)])
        return sorted_database


YEAR = sys.argv[1]

# Data and output paths
LOCATION_NAME = "dakar"
DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
OUTPUT_PATH = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/output/data/dakar/data-eps-{LOCATION_NAME}-{YEAR}.csv"

# Months of interest
MONTHS_OF_INTEREST = {"06", "07", "08", "09"}
all_files = [file for file in snflics.all_files_in(DATA_PATH) if snflics.get_time(file)["month"] in MONTHS_OF_INTEREST and snflics.get_time(file)["year"]==YEAR]
all_files.sort()

# Number of storms to consider (after analysing the whole dataset)
NB_X0 = 3
geodata = np.load("/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/data/geodata/lat_lon_2268_2080.npz")
lons = geodata["lon"][:]
lats = geodata["lat"][:]

# Coordinate of interest
Dakar_lon = -17.467686
Dakar_lat = 14.716677
Dakar_y, Dakar_x = snflics.to_yx(Dakar_lat, Dakar_lon, lats, lons)

# Context domain
CONTEXT_DOMAIN_LAT_MIN, CONTEXT_DOMAIN_LAT_MAX = 8.69, 20.69
CONTEXT_DOMAIN_LON_MIN, CONTEXT_DOMAIN_LON_MAX = -23.45, -11.45

with open(OUTPUT_PATH, "a") as output_file:

    input_header = generate_storm_feature_header(NB_X0, LOCATION_NAME)
    output_file.write(",".join(input_header) + "\n")

    for file_t0 in all_files[:50]:

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
                    storm_database = create_storm_database(Dakar_lat, Dakar_lon, x0_lat, x0_lon, in_region, data_t0)

                    # taking a certain number of closest storms
                    sorted_database = sorted(storm_database.items(), key=lambda item: item[1]['distance'])
                    X0_features = pad_closest_storms(sorted_database, NB_X0, Dakar_lat, Dakar_lon)
                    input_features = flatten_storm_features(time_t0, X0_features)

                    Cb_series = []
                    for core in core_series:
                        binary_mask = (core != 0).astype(int)
                        Cb = extract_box(binary_mask, Dakar_y, Dakar_x, 5)
                        Cb_series.append(np.max(Cb))

                    instance = input_features + Cb_series
                    output_file.write(",".join(map(str, instance)) + "\n")


                



