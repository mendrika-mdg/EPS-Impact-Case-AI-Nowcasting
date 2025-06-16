import os
import torch
import numpy as np
import json
from tqdm import tqdm
import glob

PARTITION = "train"

# Directory containing all your .pt shards
shard_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa_sharded/t1/{PARTITION}"

shard_files = sorted(glob.glob(os.path.join(shard_dir, f"shard-{PARTITION}-*.pt")))

# Initialize containers
lats, lons, tirs, sizes = [], [], [], []

for fname in tqdm(shard_files, desc=f"Processing shards"):
    path = os.path.join(shard_dir, fname)
    try:
        data = torch.load(path)
        input = data["inputs"].float()               # shape: [T, N, F]
        mask = input[:, :, 10] == 1                  # real cores only
        core = input[mask]                           # shape: [num_cores, F]
        lats.append(core[:, 5].numpy())
        lons.append(core[:, 6].numpy())
        tirs.append(core[:, 8].numpy())
        sizes.append(core[:, 9].numpy())
    except Exception as e:
        print(f"[WARN] Skipping {fname}: {e}")

# Check if any data was collected
if not lats:
    print("No valid core data found in any shard.")
    exit()

# Concatenate all arrays
lats = np.concatenate(lats)
lons = np.concatenate(lons)
tirs = np.concatenate(tirs)
sizes = np.concatenate(sizes)

# Compute stats
scaling = {
    "lat_min":  float(lats.min()),
    "lat_max":  float(lats.max()),
    "lon_min":  float(lons.min()),
    "lon_max":  float(lons.max()),
    "tir_min":  float(tirs.min()),
    "tir_max":  float(tirs.max()),
    "size_max": float(np.log1p(sizes).max())
}

# Save to JSON
output_path = "/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/normalisation/parameters/normalisation.json"

with open(output_path, "w") as f:
    json.dump(scaling, f, indent=4)

print(f"Scaling parameters saved to: {output_path}")
