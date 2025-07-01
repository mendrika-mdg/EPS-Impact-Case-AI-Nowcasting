import os
import sys
import glob
import logging
import torch
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@torch.no_grad()
def downsample_single_shard(shards_dir: str, output_dir: str, target_size=(512, 512), glob_pattern: str = "*_proc.pt"):
    try:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        raise RuntimeError("SLURM_ARRAY_TASK_ID not set. This script is intended for SLURM array jobs.")

    shard_files = sorted(glob.glob(os.path.join(shards_dir, glob_pattern)))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shards_dir}")

    if idx >= len(shard_files):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} out of range. Found only {len(shard_files)} files.")

    os.makedirs(output_dir, exist_ok=True)

    fp = shard_files[idx]
    logging.info(f"Downsampling shard [{idx+1}/{len(shard_files)}]: {fp}")

    data = torch.load(fp, map_location="cpu")
    if "inputs" not in data or "targets" not in data:
        raise KeyError(f"Missing 'inputs' or 'targets' in {fp}")

    inputs = data["inputs"]
    targets = data["targets"]  # shape: (N, 1024, 1024)

    # Downsample: (N, 1024, 1024) → (N, 512, 512)
    targets = targets.unsqueeze(1)  # Add channel: (N, 1, 1024, 1024)
    targets_down = F.interpolate(targets, size=target_size, mode="bilinear", align_corners=False)
    targets_down = targets_down.squeeze(1)  # Remove channel: (N, 512, 512)

    fname = os.path.basename(fp)
    out_path = os.path.join(output_dir, fname)

    torch.save({"inputs": inputs, "targets": targets_down}, out_path)
    logging.info(f"Saved downsampled shard to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: preprocess-shard-array.py <partition> <lead_time>")

    PARTITION = sys.argv[1]
    LEAD_TIME = int(sys.argv[2])

    root_dir = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed/t{LEAD_TIME}/{PARTITION}"
    output_path = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed/t{LEAD_TIME}/{PARTITION}_512"

    downsample_single_shard(
        shards_dir=root_dir,
        output_dir=output_path
    )
