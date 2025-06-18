import os
import torch
import json
import glob
from tqdm import tqdm
import numpy as np

# === Config ===
PARTITION = "train"
shard_dir = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard/t1/{PARTITION}"
shard_files = sorted(glob.glob(os.path.join(shard_dir, f"shard-{PARTITION}-*.pt")))
output_path = "/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/normalisation/parameters/normalisation.json"

# === Init stats ===
lat_min = 1e9; lat_max = -1e9
lon_min = 1e9; lon_max = -1e9
tir_min = 1e9; tir_max = -1e9
size_min = 1e9; size_max = -1e9
valid_count = 0

# === Process shards ===
for path in tqdm(shard_files, desc="Processing shards"):
    try:
        data = torch.load(path, map_location="cpu")
        x = data["inputs"].float()  # shape: (T, N, F)

        # Combined mask: real core & cold core (TIR < 0°C)
        combined_mask = (x[:, :, 10] == 1) & (x[:, :, 8] < 0)
        core = x[combined_mask]  # shape: (M, F)

        if core.numel() == 0:
            continue

        lat = core[:, 5]
        lon = core[:, 6]
        tir = core[:, 8]
        size = core[:, 9]

        # Update stats
        lat_min = min(lat_min, lat.min().item())
        lat_max = max(lat_max, lat.max().item())
        lon_min = min(lon_min, lon.min().item())
        lon_max = max(lon_max, lon.max().item())
        tir_min = min(tir_min, tir.min().item())
        tir_max = max(tir_max, tir.max().item())

        size_log = np.log1p(size.numpy())
        size_min = min(size_min, float(size_log.min()))
        size_max = max(size_max, float(size_log.max()))

        valid_count += core.shape[0]

    except Exception as e:
        print(f"[WARN] Skipping {path}: {e}")

# === Save ===
if valid_count == 0:
    raise ValueError("No valid cold core data found!")

scaling = {
    "lat_min": float(lat_min),
    "lat_max": float(lat_max),
    "lon_min": float(lon_min),
    "lon_max": float(lon_max),
    "tir_min": float(tir_min),
    "tir_max": float(tir_max),
    "size_min": float(size_min),
    "size_max": float(size_max),
}

with open(output_path, "w") as f:
    json.dump(scaling, f, indent=4)

print(f"Saved {valid_count} valid cold cores to: {output_path}")
