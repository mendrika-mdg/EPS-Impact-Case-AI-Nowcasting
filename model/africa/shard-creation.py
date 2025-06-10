import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time, random
from multiprocessing import Pool

# CONFIG
LEAD_TIME = 1
INPUT_DIR = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa"
OUTPUT_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa_sharded/t{LEAD_TIME}"

# Shard size: you can adjust this
SAMPLES_PER_SHARD = 10000

# Create output dir
#os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load file list (you can use your train/test/val csv)
file_list = pd.read_csv('/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/splits/train_files.csv', header=None)[0].tolist()

def process_one(idx):
    """
    Load one sample and return (input_tensor, target_tensor)
    """
    time.sleep(random.uniform(0, 1))
    fname = file_list[idx]
    
    input_path = os.path.join(INPUT_DIR, "inputs_t0", fname)
    timestamp = fname.replace("input-", "")
    target_fname = f"target-{timestamp}"
    target_path = os.path.join(INPUT_DIR, f"targets_t{LEAD_TIME}", target_fname)

    inputs = torch.load(input_path).float()
    targets = torch.load(target_path).float()  # (1024, 1024)
    return (inputs, targets)

# Loop through shards
num_files = len(file_list)
num_shards = (num_files + SAMPLES_PER_SHARD - 1) // SAMPLES_PER_SHARD

for shard_idx in range(num_shards):

    start = shard_idx * SAMPLES_PER_SHARD
    end = min((shard_idx + 1) * SAMPLES_PER_SHARD, num_files)
    indices = list(range(start, end))

    # Optional: parallel read inside shard (safe here since it's small batch)
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    with Pool(num_workers) as p: 
        data = list(tqdm(p.imap(process_one, indices), total=len(indices)))

    shard_inputs, shard_targets = zip(*data)

    # Save shard
    shard_dict = {
        "inputs": torch.stack(shard_inputs),     # (N, 140, 11)
        "targets": torch.stack(shard_targets)    # (N, 1024, 1024)
    }

    shard_path = os.path.join(OUTPUT_DIR, f"shard-{shard_idx:05d}.pt")
    torch.save(shard_dict, shard_path)

    print(f"Saved shard {shard_idx+1}/{num_shards}: {shard_path}")
