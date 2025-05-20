import os
import sys
import json
from pathlib import Path

import numpy as np              # type: ignore
from netCDF4 import Dataset     # type: ignore

# Insert local module path
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics                  # type: ignore


# ------------------------ CONFIGURATION ------------------------

DATA_PATH = Path("/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/")
OUTPUT_PATH = Path("/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Senegal/H0/dakar")

CONTEXT_DOMAIN_LAT_MIN, CONTEXT_DOMAIN_LAT_MAX = 8.69, 20.69
CONTEXT_DOMAIN_LON_MIN, CONTEXT_DOMAIN_LON_MAX = -23.45, -11.45

ZONE_SIZE = 1.0                  # in degrees
MONTHS_OF_INTEREST = {6, 7, 8, 9}

# Time slots
all_t0 = [f"{hour:02d}{minute}" for hour in range(24) for minute in ["00", "15", "30", "45"]]


# ------------------------ BUILD ZONES ------------------------

def generate_zones(lat_min, lat_max, lon_min, lon_max, step=1.0):
    zones = {}
    zone_id = 1
    lat_start = int(np.floor(lat_min))
    lat_end = int(np.ceil(lat_max))
    lon_start = int(np.floor(lon_min))
    lon_end = int(np.ceil(lon_max))
    for lat in range(lat_start, lat_end):
        for lon in range(lon_start, lon_end):
            zones[f"zone_{zone_id}"] = {
                "lat_min": lat,
                "lat_max": lat + step,
                "lon_min": lon,
                "lon_max": lon + step
            }
            zone_id += 1
    return zones


# ------------------------ MAIN PROCESSING ------------------------
def main():

    zones = generate_zones(CONTEXT_DOMAIN_LAT_MIN, CONTEXT_DOMAIN_LAT_MAX,
                           CONTEXT_DOMAIN_LON_MIN, CONTEXT_DOMAIN_LON_MAX)

    # Filter files only once
    def is_valid_file(file):
        time = snflics.get_time(file)
        return int(time["year"]) <= 2019 and int(time["month"]) in MONTHS_OF_INTEREST

    all_files = [file for file in snflics.all_files_in(DATA_PATH) if is_valid_file(file)]

    for zone_name, zone_bounds in zones.items():

        print(f"Processing {zone_name} ...")
        db = {zone_name: {}}
        
        for t0 in all_t0:
            files_t0 = []
            for file in all_files:
                time = snflics.get_time(file)
                if (time["hour"] + time["minute"]) == t0:
                    files_t0.append((file, time))

            files_t0.sort()
            dates_within_zone = []

            for file_t0, time_t0 in files_t0:
                if not os.path.exists(file_t0):
                    continue
                try:
                    with Dataset(file_t0, "r") as data:
                        lat = data["max_lat"][:]
                        lon = data["max_lon"][:]
                except Exception:
                    continue

                in_region = (
                    (lon >= zone_bounds["lon_min"]) & (lon < zone_bounds["lon_max"]) &
                    (lat >= zone_bounds["lat_min"]) & (lat < zone_bounds["lat_max"])
                )

                if np.any(in_region):
                    date_str = f"{time_t0['year']}{time_t0['month']}{time_t0['day']}"
                    dates_within_zone.append(date_str)

            db[zone_name][t0] = dates_within_zone

        output_file = OUTPUT_PATH / f"{zone_name}_dakar.json"
        with open(output_file, "w") as f:
            json.dump(db, f, indent=4)

        print(f"Saved results to {output_file}")

    # ------------------------ ENTRY POINT ------------------------

if __name__ == "__main__":
    main()
