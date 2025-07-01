import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

# === SHARD CONFIG ===
PARTITION = sys.argv[1]
LEAD_TIME = int(sys.argv[2])

SAMPLES_PER_SHARD = 1000

# TEMPORARY DIRECTORIES FOR FASTER READ
TMPDIR = "/work/scratch-nopw2/mrakotomanga/eps/pancast"
INPUT_DIR = os.path.join(TMPDIR, "inputs_t0")
TARGET_DIR = os.path.join(TMPDIR, f"targets_t{LEAD_TIME}")
OUTPUT_DIR = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard/t{LEAD_TIME}/{PARTITION}"

# === SLURM Array Index ===
try:
    shard_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
except KeyError:
    print("SLURM_ARRAY_TASK_ID not set. Run with --array=0-N.")
    sys.exit(1)

# === Load full file list ===
file_list_path = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/splits/{PARTITION}_files.csv"
file_list = pd.read_csv(file_list_path, header=None)[0].tolist()

start = shard_idx * SAMPLES_PER_SHARD
end = min((shard_idx + 1) * SAMPLES_PER_SHARD, len(file_list))
shard_files = file_list[start:end]

shard_path = os.path.join(OUTPUT_DIR, f"shard-{PARTITION}-{shard_idx:03d}.pt")
if os.path.exists(shard_path):
    print(f"[Skip] Shard {shard_idx} already exists.")
    sys.exit(0)

print(f"[Shard {shard_idx}] Processing samples {start} to {end}...")

xs, ys = [], []
for fname in tqdm(shard_files, desc=f"Shard {shard_idx}"):
    try:
        input_path = os.path.join(INPUT_DIR, fname)
        target_fname = "target-" + fname.replace("input-", "")
        target_path = os.path.join(TARGET_DIR, target_fname)

        x = torch.load(input_path, map_location="cpu").float()
        y = torch.load(target_path, map_location="cpu").float()

        xs.append(x)
        ys.append(y)
    except Exception as e:
        print(f"[Error] {fname}: {e}")

if not xs:
    print(f"[Warning] No valid samples in shard {shard_idx}")
    sys.exit(1)

torch.save({
    "inputs": torch.stack(xs),
    "targets": torch.stack(ys)
}, shard_path)

print(f"[Saved] {shard_path}")