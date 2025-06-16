import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool

def process_file(args):

    input_dir, fname = args

    full_path = os.path.join(input_dir, fname)
    data = torch.load(full_path).float()

    mask = (data[:, 10] == 1)

    if mask.sum() == 0:
        return None  # skip files without valid cores

    real = data[mask]

    return {
        'lats': real[:, 5].numpy(),
        'lons': real[:, 6].numpy(),
        'wps': real[:, 7].numpy(),
        'tirs': real[:, 8].numpy(),
        'sizes': real[:, 9].numpy()
    }


def compute_normalisation(input_dir, train_file_list, output_path):
    """
    Parallel version using SLURM environment variables
    """

    train_files = pd.read_csv(train_file_list, header=None)[0].tolist()
    args_list = [(input_dir, fname) for fname in train_files]

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {num_workers} workers", flush=True)

    lats, lons, wps, tirs, sizes = [], [], [], [], []

    with Pool(num_workers) as p:
        for result in tqdm(p.imap_unordered(process_file, args_list), total=len(args_list)):
            if result is None:
                continue
            lats.append(result['lats'])
            lons.append(result['lons'])
            wps.append(result['wps'])
            tirs.append(result['tirs'])
            sizes.append(result['sizes'])

    if len(lats) == 0:
        print("No valid cores found — exiting")
        return

    lats = np.concatenate(lats)
    lons = np.concatenate(lons)
    wps = np.concatenate(wps)
    tirs = np.concatenate(tirs)
    sizes = np.concatenate(sizes)

    normalisation = {
        "lat_min": float(lats.min()),
        "lat_max": float(lats.max()),
        "lon_min": float(lons.min()),
        "lon_max": float(lons.max()),
        "wp_max": float(np.log1p(wps).max()),  # log scale for wp
        "tir_min": float(tirs.min()),
        "tir_max": float(tirs.max()),
        "size_max": float(sizes.max())
    }

    with open(output_path, "w") as f:
        json.dump(normalisation, f, indent=4)

    print(f"Saved normalisation to {output_path}")

# Example usage:
if __name__ == "__main__":
    input_dir = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa/inputs_t0"
    train_file_list = "/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/splits/train_files.csv"
    output_path = "/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/scaling/normalisation.json"

    compute_normalisation(input_dir, train_file_list, output_path)
