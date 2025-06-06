import os
import sys
import torch
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import label, zoom
from multiprocessing import Pool, set_start_method
from datetime import datetime, timedelta

sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics

try:
    set_start_method("fork")
except RuntimeError:
    pass

def prepare_core(file):
    """
    Load core data from a NetCDF file.

    Args:
        file (str): Path to NetCDF file.

    Returns:
        np.ndarray: 2D array of core data.

    Raises:
        FileNotFoundError: If file is missing.
        OSError: If file cannot be opened.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    try:
        with Dataset(file, "r") as data:
            cores = data.variables["cores"][0, :, :]
    except OSError as e:
        raise OSError(f"Error opening NetCDF file: {file}. {e}")
    return cores

def update_hour(date_dict, hours_to_add):
    """
    Add hours to a datetime dictionary and return updated dict and file path.

    Args:
        date_dict (dict): {'year','month','day','hour','minute'} all as strings.
        hours_to_add (int): Number of hours to add.

    Returns:
        tuple: (updated_date_dict, file_path)
    """
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )
    updated = time_obj + timedelta(hours=hours_to_add)
    new_date_dict = {
        "year": f"{updated.year:04d}",
        "month": f"{updated.month:02d}",
        "day": f"{updated.day:02d}",
        "hour": f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}"
    }
    file_path = f"{new_date_dict['year']}/{new_date_dict['month']}/{new_date_dict['year']}{new_date_dict['month']}{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}.nc"
    return new_date_dict, file_path

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute Haversine distance between two points or arrays (in km).
    """
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def extract_box(matrix, y, x, box_size=3):
    half = box_size // 2
    y_min = max(y - half, 0)
    y_max = min(y + half + 1, matrix.shape[0])
    x_min = max(x - half, 0)
    x_max = min(x + half + 1, matrix.shape[1])
    return matrix[y_min:y_max, x_min:x_max]

def create_storm_database(data_t0, lats, lons):
    """
    Identify storm cores and extract features for each core.

    Args:
        data_t0 (Dataset): Dataset containing 'cores' and 'tir' variables.
        lats, lons (np.ndarray): 2D lat/lon arrays of the domain.

    Returns:
        dict: Storm database indexed by core label.
    """
    cores_t0 = data_t0["cores"][0, :, :]
    tir_t0   = data_t0['tir'][0, :, :]
    x0_lat, x0_lon = data_t0["max_lat"][:], data_t0["max_lon"][:]
    
    # label all cores
    labeled_array, _ = label(cores_t0 != 0)     
    core_labels = np.unique(labeled_array[labeled_array != 0])

    # creating database of sizes, intensities and ctts
    dict_storm_size = {lab: np.sum(labeled_array == lab) * 9 for lab in core_labels}
    dict_storm_intensity = {lab: np.mean(cores_t0[labeled_array == lab]) for lab in core_labels}

    # Compute min temperature of a core but based on 3x3 average around min TIR
    dict_storm_temperature = {}

    for lab in core_labels:
        mask = (labeled_array == lab)
        tir_core = tir_t0[mask]      
        # tir_core is a 1D array                   
        min_index = np.argmin(tir_core)                 
        # Get absolute indices of the min location
        yx_indices = np.argwhere(mask)[min_index]
        y, x = yx_indices
        box = extract_box(tir_t0, y, x)
        avg_tir = float(np.mean(box))
        dict_storm_temperature[lab] = avg_tir

    storm_database = {}
    for lat, lon in zip(x0_lat, x0_lon):
        try:
            y, x = snflics.to_yx(lat, lon, lats, lons)
        except IndexError:
            continue
        lab = labeled_array[y, x]
        if lab == 0 or lab in storm_database:
            continue
        storm_database[int(lab)] = {
            "lat": lat, 
            "lon": lon, 
            "wp": dict_storm_intensity[lab], 
            "tir": dict_storm_temperature[lab],
            "size": dict_storm_size[lab], 
            "mask": 1
        }
    return storm_database


def resize_core(original_core, target_shape_y, target_shape_x):
    """
    Resize a 2D array using bilinear interpolation.

    Args:
        original_core (np.ndarray): 2D core array.
        target_shape_y (int): Target rows.
        target_shape_x (int): Target columns.

    Returns:
        np.ndarray: Resized 2D array.
    """
    assert original_core.ndim == 2
    zoom_factors = (target_shape_y / original_core.shape[0], target_shape_x / original_core.shape[1])
    return zoom(original_core, zoom=zoom_factors, order=1)

def generate_fictional_storm(context_lat_min, context_lat_max, context_lon_min, context_lon_max, min_km_buffer=500, max_deg_buffer=4.5):
    """
    Generate a synthetic storm outside context domain but near enough.

    Returns:
        tuple: (storm_id, storm_dict)
    """
    lat_range = (context_lat_min - max_deg_buffer, context_lat_max + max_deg_buffer)
    lon_range = (context_lon_min - max_deg_buffer, context_lon_max + max_deg_buffer)
    while True:
        lat, lon = np.random.uniform(*lat_range), np.random.uniform(*lon_range)
        if context_lat_min <= lat <= context_lat_max and context_lon_min <= lon <= context_lon_max:
            continue
        d_north = haversine_distance(lat, lon, context_lat_max, lon)
        d_south = haversine_distance(lat, lon, context_lat_min, lon)
        d_east  = haversine_distance(lat, lon, lat, context_lon_max)
        d_west  = haversine_distance(lat, lon, lat, context_lon_min)
        if min(d_north, d_south, d_east, d_west) < min_km_buffer:
            continue

        # lat lon in the buffer zone
        # warm enough to be non-convective, realistic for Africa, covers both day and night
        return ('artificial', {'lat': lat, 'lon': lon, 'wp': 0.0, 'tir': float(np.random.uniform(20.0, 35.0)), 'size': 0, 'mask': 0})

def pad_observed_storms(storm_db, nb_x0, context_lat_min, context_lat_max, context_lon_min, context_lon_max):
    """
    Pad storm database to have exactly nb_x0 storms.

    Args:
        storm_db (dict): Storm database.
        nb_x0 (int): Required number of storms.

    Returns:
        list: Padded list of storms (id, dict).
    """
    storm_list = list(storm_db.items())
    if len(storm_list) >= nb_x0:
        sorted_db = sorted(storm_list, key=lambda item: item[1]['wp'], reverse=True)
        return sorted_db[:nb_x0]
    else:
        needed = nb_x0 - len(storm_list)
        storm_list.extend([
            generate_fictional_storm(context_lat_min, context_lat_max, context_lon_min, context_lon_max)
            for _ in range(needed)
        ])
        return storm_list

def transform_to_array(time_obs, data):
    """
    Transform list of storms into numpy array (tabular features).

    Args:
        time_obs (dict): Observation time.
        data (list): List of (id, storm dict).

    Returns:
        np.ndarray: Array shape (N, 11).
    """
    year = int(time_obs['year'])
    month = int(time_obs['month'])
    day = int(time_obs['day'])
    hour = int(time_obs['hour'])
    minute = int(time_obs['minute'])
    result = []
    for _, entry in data:
        lat, lon = float(entry['lat']), float(entry['lon'])
        wp, tir, size, mask = float(entry['wp']), float(entry['tir']), int(entry['size']), int(entry['mask'])
        result.append([year, month, day, hour, minute, lat, lon, wp, tir, size, mask])
    return np.array(result)


# Assume you load lats, lons, constants here once (global scope)
geodata = np.load("/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/data/geodata/lat_lon_2268_2080.npz")
lons, lats = geodata["lon"][:], geodata["lat"][:]

YEAR = sys.argv[1]

TARGET_DOMAIN_LAT_MIN, TARGET_DOMAIN_LAT_MAX = -40, 40
TARGET_DOMAIN_LON_MIN, TARGET_DOMAIN_LON_MAX = -25, 60
CONTEXT_DOMAIN_LAT_MIN, CONTEXT_DOMAIN_LAT_MAX = -46, 46
CONTEXT_DOMAIN_LON_MIN, CONTEXT_DOMAIN_LON_MAX = -31, 66
TARGET_SHAPE_Y, TARGET_SHAPE_X = 1024, 1024

DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
NB_X0 = 140


# Output folders
BASE_INPUT_DIR = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa/inputs_t0"
BASE_TARGET_DIR = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa/targets_t{}"
os.makedirs(BASE_INPUT_DIR, exist_ok=True)
for i in range(7):
    os.makedirs(BASE_TARGET_DIR.format(i), exist_ok=True)


all_files = [file for file in snflics.all_files_in(DATA_PATH) if snflics.get_time(file)["year"] == YEAR]
all_files.sort()


def process_file(file_t0):
    
    try:
        time_t0 = snflics.get_time(file_t0)
        files_info = [update_hour(time_t0, h) for h in range(7)]

        # index 1 to get the path
        files = [DATA_PATH + info[1] for info in files_info]

        if not all(os.path.exists(f) for f in files):
            return

        core_series = [prepare_core(f) for f in files]

        with Dataset(file_t0, "r") as data_t0:
            x0_lat, x0_lon = data_t0["max_lat"][:], data_t0["max_lon"][:]
            if x0_lat.size == 0 or x0_lon.size == 0:
                return

            storm_database = create_storm_database(data_t0, lats, lons)

            X0_features = pad_observed_storms(storm_database, NB_X0,
                                               CONTEXT_DOMAIN_LAT_MIN, CONTEXT_DOMAIN_LAT_MAX,
                                               CONTEXT_DOMAIN_LON_MIN, CONTEXT_DOMAIN_LON_MAX)
            
            input_features = transform_to_array(time_t0, X0_features)
            input_tensor = torch.tensor(input_features, dtype=torch.float32)

            input_filename = f"input-{time_t0['year']}{time_t0['month']}{time_t0['day']}_{time_t0['hour']}{time_t0['minute']}.pt"
            torch.save(input_tensor, os.path.join(BASE_INPUT_DIR, input_filename))

            for lead_time, core in enumerate(core_series):
                resized_core = resize_core(core, TARGET_SHAPE_Y, TARGET_SHAPE_X)
                cb_mask = (resized_core != 0).astype(np.uint8)
                target_tensor = torch.from_numpy(cb_mask)
                output_file = f"target-{time_t0['year']}{time_t0['month']}{time_t0['day']}_{time_t0['hour']}{time_t0['minute']}.pt"
                torch.save(target_tensor, os.path.join(BASE_TARGET_DIR.format(lead_time), output_file))

        print(f"Finished: {file_t0}")

    except Exception as e:
        print(f"Error on {file_t0}: {e}")

if __name__ == "__main__":

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {num_workers} workers")
    sys.stdout.flush()

    with Pool(num_workers) as p:
        p.map(process_file, all_files)
