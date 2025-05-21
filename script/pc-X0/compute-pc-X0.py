import os
import warnings
from netCDF4 import Dataset                                                             # type: ignore
import numpy as np                                                                      # type: ignore
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics                                                                          # type: ignore
from datetime import datetime, timedelta
from scipy.ndimage import maximum_filter                                                # type: ignore
import json                                                                             # type: ignore
import pandas as pd                                                                     # type: ignore

def prepare_core(file):
    """
    Prepares core data from a NetCDF file by applying binary thresholding
    and a 5x5 maximum filter to expand active pixels.

    Args:
        file (str): Path to the NetCDF file containing core data.

    Returns:
        np.ndarray: Binary array of processed core data after filtering.

    Raises:
        FileNotFoundError, OSError, RuntimeError, IndexError
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")

    try:
        with Dataset(file, "r") as data:
            cores = data.variables["cores"][0, :, :]

            if cores.ndim != 2:
                raise RuntimeError("Input data must be a 2D array.")

            cores = np.nan_to_num(cores, nan=0.0)
            binary = (cores > 0).astype(np.uint8)

            # Apply a 5x5 neighbourhood max filter
            filtered = maximum_filter(binary, size=(5, 5))

            return filtered

    except (FileNotFoundError, OSError, RuntimeError, IndexError) as e:
        raise e


def compute_pc(dataset):
    """
    Computes the probabilistic core (PC) from a list of pre-filtered file paths.

    Args:
        dataset (list): List of file paths.

    Returns:
        np.ndarray: 2D array of probabilistic core.

    Raises:
        ValueError: If no valid files were processed.
    """
    sum_cores = None
    valid_files = 0

    for file in dataset:
        try:
            cores = prepare_core(file)
            if sum_cores is None:
                sum_cores = np.zeros_like(cores, dtype=np.float32)

            sum_cores += cores
            valid_files += 1

        except (FileNotFoundError, OSError, RuntimeError, IndexError):
            continue

    if valid_files == 0:
        raise ValueError("No valid files were processed from the dataset.")

    return sum_cores / valid_files


# Geographical coordinates of the input location
LOCATION_LON =  -17.467686
LOCATION_LAT = 14.716677
GEODATA = np.load('/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/data/geodata/lat_lon_2268_2080.npz')
LONS = GEODATA["lon"][:]
LATS = GEODATA["lat"][:]
LOCATION_Y, LOCATION_X = snflics.to_yx(LOCATION_LAT, LOCATION_LON, LATS, LONS)


YEAR = sys.argv[1]

# Data and output paths
DATA_PATH = '/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/'
OUTPUT_PATH = '/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/output/data/dakar'

TEST_PATH = f'/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/output/data/dakar/test/data-eps-dakar-{YEAR}.csv'

test_data = pd.read_csv(TEST_PATH)

# Define file paths for each zone in a dictionary
file_paths = {
    f"zone_{i}": f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Senegal/H0/dakar/zone_{i}.json"
    for i in range(1, 17)
}

# Load data for each zone into a dictionary
data_zones = {}
for zone, file_path in file_paths.items():
    with open(file_path, "r") as file:
        data_zones[zone] = json.load(file)


zones = {
    'zone_1': {'lat_min': 8.69, 'lat_max': 11.69, 'lon_min': -23.45, 'lon_max': -20.45},
    'zone_2': {'lat_min': 8.69, 'lat_max': 11.69, 'lon_min': -20.45, 'lon_max': -17.45},
    'zone_3': {'lat_min': 8.69, 'lat_max': 11.69, 'lon_min': -17.45, 'lon_max': -14.45},
    'zone_4': {'lat_min': 8.69, 'lat_max': 11.69, 'lon_min': -14.45, 'lon_max': -11.45},
    'zone_5': {'lat_min': 11.69, 'lat_max': 14.69, 'lon_min': -23.45, 'lon_max': -20.45},
    'zone_6': {'lat_min': 11.69, 'lat_max': 14.69, 'lon_min': -20.45, 'lon_max': -17.45},
    'zone_7': {'lat_min': 11.69, 'lat_max': 14.69, 'lon_min': -17.45, 'lon_max': -14.45},
    'zone_8': {'lat_min': 11.69, 'lat_max': 14.69, 'lon_min': -14.45, 'lon_max': -11.45},
    'zone_9': {'lat_min': 14.69, 'lat_max': 17.69, 'lon_min': -23.45, 'lon_max': -20.45},
    'zone_10': {'lat_min': 14.69, 'lat_max': 17.69, 'lon_min': -20.45, 'lon_max': -17.45},
    'zone_11': {'lat_min': 14.69, 'lat_max': 17.69, 'lon_min': -17.45, 'lon_max': -14.45},
    'zone_12': {'lat_min': 14.69, 'lat_max': 17.69, 'lon_min': -14.45, 'lon_max': -11.45},
    'zone_13': {'lat_min': 17.69, 'lat_max': 20.69, 'lon_min': -23.45, 'lon_max': -20.45},
    'zone_14': {'lat_min': 17.69, 'lat_max': 20.69, 'lon_min': -20.45, 'lon_max': -17.45},
    'zone_15': {'lat_min': 17.69, 'lat_max': 20.69, 'lon_min': -17.45, 'lon_max': -14.45},
    'zone_16': {'lat_min': 17.69, 'lat_max': 20.69, 'lon_min': -14.45, 'lon_max': -11.45}
    }

def find_zone(lat, lon):
    for zone_name, boundaries in zones.items():
        if (boundaries["lat_min"] <= lat <= boundaries["lat_max"] and
                boundaries["lon_min"] <= lon <= boundaries["lon_max"]):
            return zone_name
    return "Out of bounds"


PC_X0_LT1 = []
PC_X0_LT3 = []
PC_X0_LT6 = []


for i in range(len(test_data)):

    instance = test_data.iloc[i]

    t0 = f"{int(instance['hour']):02d}{int(instance['minute']):02d}"
    
    selected_coords = []

    for j in range(1, 4):
        if instance[f'mask{j}'] == 1:
            selected_coords.append((instance[f'lat{j}'], instance[f'lon{j}']))

    print(len(selected_coords))
    
    pc_per_instance_lt1 = []
    pc_per_instance_lt3 = []
    pc_per_instance_lt6 = []
    
    for lat, lon in selected_coords:     
        region = find_zone(lat, lon)
    
        # taking t0
        # all the date that had storms in the region at time t0 and convert it into date time format so increasing it with the lead time would be easier
        H0 = [date + t0  for date in data_zones[region][region][t0]]
        H0_datetime = [datetime.strptime(date, "%Y%m%d%H%M") for date in H0]
    
        # defining lead time
        lead_time_t1 = timedelta(hours=1)
        lead_time_t3 = timedelta(hours=3)
        lead_time_t6 = timedelta(hours=6)
    
        # all the datetimes moved wrt to the lead time
        H1_datetime = [date + lead_time_t1 for date in H0_datetime]
        H3_datetime = [date + lead_time_t3 for date in H0_datetime]
        H6_datetime = [date + lead_time_t6 for date in H0_datetime]
    
        # Convert back to str and add path to the data for computation
        H1 = [DATA_PATH + date.strftime("%Y") + "/" + date.strftime("%m") + "/" + date.strftime("%Y%m%d%H%M") + ".nc" for date in H1_datetime]
        H3 = [DATA_PATH + date.strftime("%Y") + "/" + date.strftime("%m") + "/" + date.strftime("%Y%m%d%H%M") + ".nc" for date in H3_datetime]
        H6 = [DATA_PATH + date.strftime("%Y") + "/" + date.strftime("%m") + "/" + date.strftime("%Y%m%d%H%M") + ".nc" for date in H6_datetime]
            
        # compute pc|x0
        try:
            pc_x0_lt1 = compute_pc(H1)[LOCATION_Y, LOCATION_X]
        except ValueError:
            pc_x0_lt1 = 0
    
        try:
            pc_x0_lt3 = compute_pc(H3)[LOCATION_Y, LOCATION_X]
        except ValueError:
            pc_x0_lt3 = 0

        try:
            pc_x0_lt6 = compute_pc(H6)[LOCATION_Y, LOCATION_X]
        except ValueError:
            pc_x0_lt6 = 0

        print(pc_x0_lt1)
        print(pc_x0_lt3)
        print(pc_x0_lt6)
        
        pc_per_instance_lt1.append(pc_x0_lt1)
        pc_per_instance_lt3.append(pc_x0_lt3)
        pc_per_instance_lt6.append(pc_x0_lt6)
    
    PC_X0_LT1.append(np.max(pc_per_instance_lt1))
    PC_X0_LT3.append(np.max(pc_per_instance_lt3))
    PC_X0_LT6.append(np.max(pc_per_instance_lt6))

test_data["pc_nflics_t1"] = PC_X0_LT1
test_data["pc_nflics_t3"] = PC_X0_LT3
test_data["pc_nflics_t6"] = PC_X0_LT6

test_data.to_csv(f"{OUTPUT_PATH}/test-nflics-{YEAR}.csv", index=False)