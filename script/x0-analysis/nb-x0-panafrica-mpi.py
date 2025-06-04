import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from multiprocessing import Pool

# Add local module path
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics

# Data path
data_path = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
all_files = sorted(snflics.all_files_in(data_path))

# Output path
output_dir = "/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/output/x0-analysis/pan-africa"
os.makedirs(output_dir, exist_ok=True)

# === Worker function ===
def process_file(file_path):
    try:
        with Dataset(file_path, "r") as nc_file:
            latitudes = nc_file["max_lat"][:].compressed()
            count = latitudes.size
            print(f"Processed {file_path} -> count: {count}")
            sys.stdout.flush()
            return count
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        sys.stdout.flush()
        return None

if __name__ == "__main__":
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {num_workers} workers")
    sys.stdout.flush()

    with Pool(num_workers) as p:
        results = p.map(process_file, all_files)

    # Filter out failed files
    storm_counts = np.array([r for r in results if r is not None])

    if storm_counts.size > 0:
        np.save(f"{output_dir}/nbx0-pan-africa-mpi.npy", storm_counts)

        plt.figure(figsize=(8, 6))
        plt.hist(storm_counts, bins=20, color="blue", alpha=0.7)
        plt.xlabel("Number of storms per image in Africa")
        plt.ylabel("Frequency")
        plt.title("Storm counts over Africa")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nbx0-pan-africa-mpi.png")
        plt.close()

        print("Finished histogram successfully.")
    else:
        print("No valid storm data was found.")
