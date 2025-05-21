#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path

import numpy as np                      # type: ignore
from netCDF4 import Dataset             # type: ignore

# Insert local module path
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics  # type: ignore

# ------------------------ CONFIGURATION ------------------------

zone_name = sys.argv[1]

DATA_PATH = Path("/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/")
OUTPUT_BASE = Path("/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Senegal/H0/")
ZONE_SIZE = 3.0  # in degrees
MONTHS_OF_INTEREST = {6, 7, 8, 9}

CONTEXT_DOMAIN_LAT_MIN, CONTEXT_DOMAIN_LAT_MAX = 8.69, 20.69
CONTEXT_DOMAIN_LON_MIN, CONTEXT_DOMAIN_LON_MAX = -23.45, -11.45

# Time slots: every 15 minutes
all_t0 = [f"{hour:02d}{minute}" for hour in range(24) for minute in ["00", "15", "30", "45"]]

# ------------------------ MAIN FUNCTION ------------------------

def main():
    # Define zones
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
        'zone_16': {'lat_min': 17.69, 'lat_max': 20.69, 'lon_min': -14.45, 'lon_max': -11.45},
        'zone_17': {'lat_min': 20.69, 'lat_max': 20.69, 'lon_min': -23.45, 'lon_max': -20.45},
        'zone_18': {'lat_min': 20.69, 'lat_max': 20.69, 'lon_min': -20.45, 'lon_max': -17.45},
        'zone_19': {'lat_min': 20.69, 'lat_max': 20.69, 'lon_min': -17.45, 'lon_max': -14.45},
        'zone_20': {'lat_min': 20.69, 'lat_max': 20.69, 'lon_min': -14.45, 'lon_max': -11.45},
    }

    if zone_name not in zones:
        print(f"Error: zone_name '{zone_name}' not found.")
        sys.exit(1)

    def is_valid_file(file):
        time = snflics.get_time(file)
        return int(time["year"]) <= 2019 and int(time["month"]) in MONTHS_OF_INTEREST

    all_files = [file for file in snflics.all_files_in(DATA_PATH) if is_valid_file(file)]
    S_0 = zones[zone_name]
    db = {zone_name: {}}

    for t0 in all_t0:
        all_files_t0 = [
            file for file in all_files
            if (snflics.get_time(file)["hour"] + snflics.get_time(file)["minute"]) == t0
        ]
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

                in_region = (
                    (lon >= S_0["lon_min"]) & (lon < S_0["lon_max"]) &
                    (lat >= S_0["lat_min"]) & (lat < S_0["lat_max"])
                )
                lat, lon = lat[in_region], lon[in_region]

                if lat.size > 0 and lon.size > 0:
                    date_str = f"{time_t0['year']}{time_t0['month']}{time_t0['day']}"
                    dates_wx0_in_S0_at_t0.append(date_str)

        db[zone_name][t0] = dates_wx0_in_S0_at_t0

    output_path = OUTPUT_BASE / f"{zone_name}.json"
    with open(output_path, "w") as json_file:
        json.dump(db, json_file, indent=4)

# ------------------------ ENTRY POINT ------------------------

if __name__ == "__main__":
    main()
