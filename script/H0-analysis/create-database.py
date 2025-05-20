from scipy import ndimage                                                               # type: ignore
import os
import warnings
from netCDF4 import Dataset                                                             # type: ignore
import numpy as np                                                                      # type: ignore
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics                                                                          # type: ignore
from datetime import datetime, timedelta
import copy
import json                                                                             # type: ignore

# Data and output paths
DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
OUTPUT_PATH = f"/home/users/mendrika/SSA/SA/output/H0-analysis/"

zone_name = sys.argv[1]

zone = {
    "zone_1": {"lat_min": -13.75, "lat_max": -12.5, "lon_min": 47.5, "lon_max": 49},
    "zone_2": {"lat_min": -13.75, "lat_max": -12.5, "lon_min": 49, "lon_max": 50.5},
    "zone_3": {"lat_min": -15, "lat_max": -13.75, "lon_min": 47.5, "lon_max": 49},
    "zone_4": {"lat_min": -15, "lat_max": -13.75, "lon_min": 49, "lon_max": 50.5},
}

all_t0 = []
for hour in range(24):
    for minute in ["00", "15", "30", "45"]:
        all_t0.append(f"{hour:02d}{minute}")


MONTHS_OF_INTEREST = {'06', '07', '08', '09'}

all_files = [file for file in snflics.all_files_in(DATA_PATH) if snflics.get_time(file)["month"] in MONTHS_OF_INTEREST and int(snflics.get_time(file)["year"]) <= 2019]
S_0 = zone[zone_name]
db = {zone_name: {}}

for t0 in all_t0:
    
    all_files_t0 = [file for file in all_files if (snflics.get_time(file)["hour"] + snflics.get_time(file)["minute"]) == t0]
    all_files_t0.sort()

    dates_wx0_in_S0_at_t0 = []

    for file_t0 in all_files_t0[:]:

        time_t0 = snflics.get_time(file_t0)        

        if os.path.exists(file_t0):
        
            try:
                data_t0 = Dataset(file_t0, "r")
            except OSError:
                continue
                    
            lat = data_t0["max_lat"][:]
            lon = data_t0["max_lon"][:]        

            in_region = (lon >= S_0["lon_min"]) & (lon < S_0["lon_max"]) & (lat >= S_0["lat_min"]) & (lat < S_0["lat_max"])
            lat, lon = lat[in_region], lon[in_region]

            if lat.size > 0 and lon.size > 0:
                dates_wx0_in_S0_at_t0.append(f"{time_t0['year']}{time_t0['month']}{time_t0['day']}")
        
    data_zone = db[zone_name]
    data_zone[t0] = dates_wx0_in_S0_at_t0

with open(f"{OUTPUT_PATH}{zone_name}_sofia.json", "w") as json_file:
    json.dump(db, json_file, indent=4)