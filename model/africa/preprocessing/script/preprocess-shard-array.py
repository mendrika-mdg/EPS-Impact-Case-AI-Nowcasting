# Standard libraries
import os
import sys
import json
import math
import glob
import logging

# Third-party libraries
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class _PreProcessor:
    """Batch-compatible preprocessor for storm nowcasting features (with `year`, `day` and `wp` removed)."""
    def __init__(self, norm_json: str):
        with open(norm_json, "r") as f:
            self.norm = json.load(f)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        This works for a batch/shard or a single file
        x: Tensor of shape (B, 140, 11) or (140, 11)
        returns Tensor of shape (B, 140, 9) or (140, 9)
        """

        # original header: [year, month, day, hour, minute, lat, lon, wp, tir, size, mask]

        # removing: year (0), day (2), wp (7)
        is_batch = x.dim() == 3  # (B, 140, 11)

        if not is_batch:
            x = x.unsqueeze(0)  # convert to batch shape

        # Keep [month, hour, minute, lat, lon, tir, size, mask]
        x = x[:, :, [1, 3, 4, 5, 6, 8, 9, 10]]  # shape: (B, 140, 8)

        month  = x[:, :, 0]
        hour   = x[:, :, 1]
        minute = x[:, :, 2]
        lat    = x[:, :, 3]
        lon    = x[:, :, 4]
        tir    = x[:, :, 5]
        size   = x[:, :, 6]
        mask   = x[:, :, 7]

        B, N = x.shape[:2]
        out = torch.empty((B, N, 9), dtype=torch.float32)

        # month encoding
        out[:, :, 0] = torch.sin(2 * math.pi * (month - 1) / 12.0)
        out[:, :, 1] = torch.cos(2 * math.pi * (month - 1) / 12.0)

        # time of day encoding
        tod = hour + minute / 60.0
        out[:, :, 2] = torch.sin(2 * math.pi * tod / 24.0)
        out[:, :, 3] = torch.cos(2 * math.pi * tod / 24.0)

        # lat, lon, tir, size scaling
        out[:, :, 4] = (lat - self.norm["lat_min"]) / (self.norm["lat_max"] - self.norm["lat_min"])
        out[:, :, 5] = (lon - self.norm["lon_min"]) / (self.norm["lon_max"] - self.norm["lon_min"])
        out[:, :, 6] = (tir - self.norm["tir_min"]) / (self.norm["tir_max"] - self.norm["tir_min"])
        out[:, :, 7] = torch.log1p(size) / self.norm["size_max"]

        # mask kept as is
        out[:, :, 8] = mask

        return out if is_batch else out.squeeze(0)


@torch.no_grad()
def process_single_shard(shards_dir: str, norm_path: str, output_dir: str, glob_pattern: str = "*.pt"):
    processor = _PreProcessor(norm_path)

    # Get SLURM task index
    try:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        raise RuntimeError("SLURM_ARRAY_TASK_ID not set. This script is meant to run inside an array job.")

    shard_files = sorted(glob.glob(os.path.join(shards_dir, glob_pattern)))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in: {shards_dir}")

    # idx is from the SLURM scheduler, I just need to make sure this has the same length as the iterable
    if idx >= len(shard_files):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} is out of range. Found only {len(shard_files)} files.")

    fp = shard_files[idx]
    logging.info(f"Processing shard [{idx}/{len(shard_files)}]: {fp}")

    data = torch.load(fp, map_location="cpu")
    if "inputs" not in data or "targets" not in data:
        raise KeyError(f"Missing 'inputs' or 'targets' in: {fp}")

    x_batch, y_batch = data["inputs"], data["targets"]
    x_batch_proc = processor(x_batch.float())

    fname = os.path.basename(fp).replace(".pt", "_proc.pt")
    out_path = os.path.join(output_dir, f"t{LEAD_TIME}-{fname}")

    torch.save({"inputs": x_batch_proc, "targets": y_batch}, out_path)
    logging.info(f"Saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: preprocess-shard-array.py <partition> <lead_time>")

    PARTITION = sys.argv[1]
    LEAD_TIME = int(sys.argv[2])  # ensure integer


    norm_path = "/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/normalisation/parameters/normalisation.json"
    root_dir = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard/t{LEAD_TIME}/{PARTITION}"
    output_path = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed/t{LEAD_TIME}/{PARTITION}"

    process_single_shard(
        shards_dir=root_dir,
        norm_path=norm_path,
        output_dir=output_path
    )
