from scipy import ndimage                                                               
import os
import warnings
from netCDF4 import Dataset                                                             
import numpy as np
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics                                                                         
from datetime import datetime, timedelta
import copy

# cmd arguments
YEAR = sys.argv[1]
LEAD_TIME = sys.argv[2]                                              

# for short lead time
CONTEXT_DOMAIN_LAT_MIN, CONTEXT_DOMAIN_LAT_MAX = 11.716677, 11.716677 + 6
CONTEXT_DOMAIN_LON_MIN, CONTEXT_DOMAIN_LON_MAX = -20.467686, -20.467686 + 6

# Geographical coordinates of the input location
LOCATION_NAME = "dakar"
LOCATION_LON = -17.467686
LOCATION_LAT = 14.716677

LONS = np.load("/localhome/home/mmmhr/EPS-Impact-Case-AI-Nowcasting/data/geodata/Senegal/Senegal-original-lons.npy")
LATS = np.load("/localhome/home/mmmhr/EPS-Impact-Case-AI-Nowcasting/data/geodata/Senegal/Senegal-original-lats.npy")

# this is Dakar location
LOCATION_Y, LOCATION_X = snflics.to_yx(LOCATION_LAT, LOCATION_LON, LATS, LONS)

# replace this with context domain
DOMAIN_Y_MIN, DOMAIN_Y_MAX, DOMAIN_X_MIN, DOMAIN_X_MAX = 1650, 1861, 96, 315

# Data and output paths
DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
OUTPUT_PATH = f".csv"

# Months of interest
MONTHS_OF_INTEREST = {"06", "07", "08", "09"}
all_files = [file for file in snflics.all_files_in(DATA_PATH) if snflics.get_time(file)["month"] in MONTHS_OF_INTEREST and snflics.get_time(file)["year"]==YEAR]
all_files.sort()

# Number of storms to consider
NB_X0 = 1

def format_coord(nb_x0, x0, lat, lon):
    """Formats the coordinates for the top `nb_x0` power-maxima as a CSV string."""
    top_power = snflics.top(nb_x0, x0, reverse=True)
    top_indices = [x0.index(power) for power in top_power]
    return ",".join(f"{lat[i]:.4f},{lon[i]:.4f}" for i in top_indices)

def construct_field_names(nb_x0, location_name):
    """Constructs CSV field names dynamically."""
    t0_fields = "year,month,day,hour,minute"
    loc_fields = ",".join(f"lat{i},lon{i}" for i in range(1, nb_x0 + 1))
    wp_fields = ",".join(f"wp{i}" for i in range(1, nb_x0 + 1))
    ds_fields = ",".join(f"ds{i}" for i in range(1, nb_x0 + 1))
    sz_fields = ",".join(f"size{i}" for i in range(1, nb_x0 + 1))
    return f"{t0_fields},{loc_fields},{wp_fields},{sz_fields},{ds_fields},Cb_{location_name}_t0,Cb_{location_name}_t1,Cb_{location_name}_t3\n"

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

def extract_box(data, y, x, box_size=5, spacing_km=3):
    """
    Extracts a square region around a central grid point and calculates area coverage.

    Parameters:
    -----------
    data : np.ndarray
        2D array from which to extract the box (e.g., cores).
    y, x : int
        Central grid indices.
    box_size : int
        Size of the square box (must be odd), e.g., 5 for a 5x5 box.
    spacing_km : float
        Grid spacing in kilometres.

    Returns:
    --------
    box : np.ndarray
        The extracted subregion.
    extent_km : float
        The physical width/height (in km) of the area covered.
    """
    assert box_size % 2 == 1, "box_size must be odd"
    
    half = box_size // 2
    y_min = max(0, y - half)
    y_max = min(data.shape[0], y + half + 1)
    x_min = max(0, x - half)
    x_max = min(data.shape[1], x + half + 1)
    
    box = data[y_min:y_max, x_min:x_max]  

    return box


print(len(all_files))

with open(OUTPUT_PATH, "+a") as output_file:

    output_file.write(construct_field_names(NB_X0, LOCATION_NAME))

    for file_t0 in all_files[:]:
        # Check if lead time file exists
        time_t0 = snflics.get_time(file_t0)        
        file_t1 = DATA_PATH + update_hour(time_t0, 1)[1]
        file_t3 = DATA_PATH + update_hour(time_t0, 3)[1]

        if os.path.exists(file_t0) and os.path.exists(file_t1) and os.path.exists(file_t3):
        
            try:
                # Open datasets
                data_t0 = Dataset(file_t0, "r")
                data_t1 = Dataset(file_t1, "r")
                data_t3 = Dataset(file_t3, "r")
            except OSError:
                continue
            
            lat = data_t0["max_lat"][:]
            lon = data_t0["max_lon"][:]        

            # within the region of interest
            in_region = (lon >= CONTEXT_DOMAIN_LON_MIN) & (lon <= CONTEXT_DOMAIN_LON_MAX) & (lat >= CONTEXT_DOMAIN_LAT_MIN) & (lat <= CONTEXT_DOMAIN_LAT_MAX)
            lat, lon = lat[in_region], lon[in_region]

            # if there are storms within the context domain, we include them as input if close to Dakar at time t0

            # Extract storm data
            cores_t0 = data_t0["cores"][0, DOMAIN_Y_MIN:DOMAIN_Y_MAX+1, DOMAIN_X_MIN:DOMAIN_X_MAX+1]
            binary_cores_t0 = snflics.prepare_core(file_t0, 1, DOMAIN_Y_MIN, DOMAIN_Y_MAX, DOMAIN_X_MIN, DOMAIN_X_MAX)

            # Transform lat/lon to y/x coordinates
            if lat.size > 0 and lon.size > 0:
                Y, X = zip(*(snflics.to_yx(lt, ln, LATS, LONS) for lt, ln in zip(lat, lon) if snflics.to_yx(lt, ln, LATS, LONS)))
                Y, X = np.array(Y), np.array(X)

                distances = np.sqrt((X - LOCATION_X)**2 + (Y - LOCATION_Y)**2)
                
                top_dist = snflics.top(NB_X0, distances.tolist(), reverse=False)
                print(top_dist)

                # Process the nearest storms
                if top_dist:
                    top_indices = [distances.tolist().index(dist) for dist in top_dist]
                    lat, lon, distances = lat[top_indices], lon[top_indices], distances[top_indices]
                    x0 = snflics.x0_from(lat, lon, LATS, LONS, cores_t0)

                    if len(x0) > 0:
                        # Gather data for output
                        wavelet_power = ",".join(map(str, x0))
                        distance_data = ",".join(f"{dist:.2f}" for dist in distances)
                        Storm_t0 = snflics.get_storm(binary_cores_t0)
                        storm_sizes = [Storm_t0["size"].get(str(snflics.get_x0_label(power, cores_t0, Storm_t0["labels"])[0]), "") for power in x0]
                        storm_size_data = ",".join(str(size) for size in storm_sizes if size)

                        # Core value at location and time t+0, +1 and +3
                        try:
                            binary_cores_t0 = snflics.prepare_core(file_t0, 1, DOMAIN_Y_MIN, DOMAIN_Y_MAX, DOMAIN_X_MIN, DOMAIN_X_MAX)
                            Cb_location_t0 = np.max(extract_box(binary_cores_t0, LOCATION_Y, LOCATION_X, box_size=5))

                            binary_cores_t1 = snflics.prepare_core(file_t1, 1, DOMAIN_Y_MIN, DOMAIN_Y_MAX, DOMAIN_X_MIN, DOMAIN_X_MAX)
                            Cb_location_t1 = np.max(extract_box(binary_cores_t1, LOCATION_Y, LOCATION_X, box_size=5))

                            binary_cores_t3 = snflics.prepare_core(file_t3, 1, DOMAIN_Y_MIN, DOMAIN_Y_MAX, DOMAIN_X_MIN, DOMAIN_X_MAX)
                            Cb_location_t3 = np.max(extract_box(binary_cores_t3, LOCATION_Y, LOCATION_X, box_size=5))

                            # Construct row and write to file
                            time_data = f"{int(time_t0['year'])},{int(time_t0['month'])},{int(time_t0['day'])},{int(time_t0['hour'])},{int(time_t0['minute'])}"
                            coord_data = format_coord(NB_X0, x0, lat, lon)
                            row_data = f"{time_data},{coord_data},{wavelet_power},{storm_size_data},{distance_data},{Cb_location_t0},{Cb_location_t1},{Cb_location_t3}\n"
                            output_file.write(row_data)
                        except OSError:
                            continue
