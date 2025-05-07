import sys
import os
import numpy as np                                                                      # type: ignore
import matplotlib.pyplot as plt                                                         # type: ignore
from netCDF4 import Dataset                                                             # type: ignore

# Local module
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics                                                                          # type: ignore

# Read command-line arguments
domain_lat_min = float(sys.argv[1])
domain_lat_max = float(sys.argv[2])
domain_lon_min = float(sys.argv[3])
domain_lon_max = float(sys.argv[4])

# Data path
data_path = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
all_files = sorted(snflics.all_files_in(data_path))                                     # Sorted for reproducibility

# Output path
output_dir = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/output/x0-analysis"
os.makedirs(output_dir, exist_ok=True)

# List to store storm counts
all_valid_lons = []
all_valid_lats = []

# Process each file
for file_t0 in all_files:
    try:
        if os.path.exists(file_t0):
            time_t0 = snflics.get_time(file_t0)
            if time_t0["month"] in ["06", "07", "08", "09"]:
                with Dataset(file_t0, "r") as data_t0:
                    latitudes = data_t0["max_lat"][:].compressed()
                    longitudes = data_t0["max_lon"][:].compressed()

                    # Filter based on geographic boundaries
                    valid_indices = (
                        (longitudes >= domain_lon_min) & (longitudes <= domain_lon_max) &
                        (latitudes  >= domain_lat_min) & (latitudes  <= domain_lat_max)
                    )
                    if np.any(valid_indices):
                        all_valid_lons.extend(longitudes[valid_indices])
                        all_valid_lats.extend(latitudes[valid_indices])
        else:
            print(f"File not found: {file_t0}")

    except OSError:
        print(f"Corrupted or unreadable file skipped: {file_t0}")
        continue
    except Exception as e:
        print(f"Error processing file {file_t0}: {e}")
        continue

# Save and plot results
np.save(f"{output_dir}/all_valid_lats.npy", all_valid_lats)
np.save(f"{output_dir}/all_valid_lons.npy", all_valid_lons)