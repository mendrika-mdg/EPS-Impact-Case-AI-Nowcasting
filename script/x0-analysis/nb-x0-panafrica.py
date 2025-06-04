import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Add local module path
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics

# Data path
data_path = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
all_files = sorted(snflics.all_files_in(data_path))

# Output path
output_dir = "/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/output/x0-analysis/pan-africa"
os.makedirs(output_dir, exist_ok=True)

# List to store storm counts
storm_counts = []

# Process each file
for file_path in all_files:
    try:
        time_t0 = snflics.get_time(file_path)
        with Dataset(file_path, "r") as nc_file:
            latitudes = nc_file["max_lat"][:].compressed()
            longitudes = nc_file["max_lon"][:].compressed()
            count = latitudes.size
            if count > 0:
                storm_counts.append(count)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except OSError:
        print(f"Corrupted or unreadable file skipped: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Save and plot results
if storm_counts:
    storm_counts = np.array(storm_counts)
    np.save(f"{output_dir}/nbx0-pan-africa.npy", storm_counts)

    plt.figure(figsize=(8, 6))
    plt.hist(storm_counts, bins=20, color="blue", alpha=0.7)
    plt.xlabel("Number of storms per image in Africa")
    plt.ylabel("Frequency")
    plt.title("Storm counts over Africa")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nbx0-pan-africa.png")
    plt.close()

else:
    print("No valid storm data was found.")
