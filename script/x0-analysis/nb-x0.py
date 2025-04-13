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
region_name = sys.argv[5]
lead_time = sys.argv[6]

# Data path
data_path = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
all_files = sorted(snflics.all_files_in(data_path))                                     # Sorted for reproducibility

# Output path
output_dir = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/output/x0-analysis"
os.makedirs(output_dir, exist_ok=True)

# List to store storm counts
storm_counts = []

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
                    count = np.count_nonzero(valid_indices)
                    if count > 0:
                        storm_counts.append(count)
        else:
            print(f"File not found: {file_t0}")

    except OSError:
        print(f"Corrupted or unreadable file skipped: {file_t0}")
        continue
    except Exception as e:
        print(f"Error processing file {file_t0}: {e}")
        continue

# Save and plot results
if storm_counts:
    storm_counts = np.array(storm_counts)
    np.save(f"{output_dir}/nbx0-{region_name}-{lead_time}.npy", storm_counts)
    plt.hist(storm_counts, bins=20, color="blue", alpha=0.7)
    plt.xlabel(f"Number of storms per image in {region_name.capitalize()}")
    plt.ylabel("Frequency")
    plt.title(f"Storm counts over {region_name.capitalize()} (JJA+S)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nbx0-{region_name}-{lead_time}.png")
else:
    print("No valid storm data was found.")