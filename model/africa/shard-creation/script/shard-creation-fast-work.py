import os
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

# CONFIG
LEAD_TIME = 1
PARTITION = 'val'


TMPDIR = "/work/scratch-nopw2/mrakotomanga/eps/pancast"
INPUT_DIR = os.path.join(TMPDIR, "inputs_t0")
TARGET_DIR = os.path.join(TMPDIR, f"targets_t{LEAD_TIME}")
OUTPUT_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa_sharded/t{LEAD_TIME}/{PARTITION}"

SAMPLES_PER_SHARD = 1000

# Load file list
file_list = pd.read_csv(
    f'/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/splits/{PARTITION}_files.csv',
    header=None
)[0].tolist()

def process_one(idx):
    try:
        fname = file_list[idx]
        input_path = os.path.join(INPUT_DIR, fname)
        target_fname = "target-" + fname.replace("input-", "")
        target_path = os.path.join(TARGET_DIR, target_fname)

        inputs = torch.load(input_path, map_location="cpu").float()
        targets = torch.load(target_path, map_location="cpu").float()
        return (inputs, targets)

    except Exception as e:
        print(f"[Error] Index {idx}, File {fname}: {e}")
        return None

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sharding
num_files = len(file_list)
num_shards = (num_files + SAMPLES_PER_SHARD - 1) // SAMPLES_PER_SHARD

for shard_idx in range(num_shards):
    shard_path = os.path.join(OUTPUT_DIR, f"shard-{PARTITION}-{shard_idx:03d}.pt")
    if os.path.exists(shard_path):
        print(f"[Skip] Shard {shard_idx} already exists")
        continue

    start = shard_idx * SAMPLES_PER_SHARD
    end = min((shard_idx + 1) * SAMPLES_PER_SHARD, num_files)
    indices = list(range(start, end))
    # num_workers = min(48, len(indices))
    
    num_workers = min(4, len(indices))

    print(f"\n[Shard {shard_idx+1}/{num_shards}] Processing {len(indices)} samples with {num_workers} workers")

    with Pool(num_workers) as p:
        data = list(tqdm(p.imap(process_one, indices), total=len(indices)))

    data = [d for d in data if d is not None]

    if not data:
        print(f"[Warning] No valid data in shard {shard_idx}")
        continue

    shard_inputs, shard_targets = zip(*data)
    shard_dict = {
        "inputs": torch.stack(shard_inputs),     # (N, 140, 11)
        "targets": torch.stack(shard_targets)    # (N, 1024, 1024)
    }

    torch.save(shard_dict, shard_path)
    print(f"[Saved] Shard {shard_idx+1}/{num_shards} at {shard_path}")
