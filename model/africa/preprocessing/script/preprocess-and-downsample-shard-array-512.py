# Standard libraries
import os
import sys
import json
import math
import glob
import logging

# Third-party
import torch
import torch.nn.functional as F

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === Preprocessor Class ===
class _PreProcessor:
    def __init__(self, norm_json: str):
        with open(norm_json, "r") as f:
            self.norm = json.load(f)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (B, 140, 11) or (140, 11)
        Output shape: (B, 140, 9) or (140, 9)
        """
        is_batch = x.dim() == 3
        if not is_batch:
            x = x.unsqueeze(0)

        # keep: month (1), hour (3), minute (4), lat (5), lon (6), tir (8), size (9), mask (10)
        x = x[:, :, [1, 3, 4, 5, 6, 8, 9, 10]]

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

        # Month cyclic encoding
        out[:, :, 0] = torch.sin(2 * math.pi * (month - 1) / 12.0)
        out[:, :, 1] = torch.cos(2 * math.pi * (month - 1) / 12.0)

        # Time of day cyclic encoding
        tod = hour + minute / 60.0
        out[:, :, 2] = torch.sin(2 * math.pi * tod / 24.0)
        out[:, :, 3] = torch.cos(2 * math.pi * tod / 24.0)

        # Normalisation
        out[:, :, 4] = (lat - self.norm["lat_min"]) / (self.norm["lat_max"] - self.norm["lat_min"])
        out[:, :, 5] = (lon - self.norm["lon_min"]) / (self.norm["lon_max"] - self.norm["lon_min"])
        out[:, :, 6] = (tir - self.norm["tir_min"]) / (self.norm["tir_max"] - self.norm["tir_min"])
        out[:, :, 7] = torch.log1p(size) / self.norm["size_max"]
        out[:, :, 8] = mask

        return out if is_batch else out.squeeze(0)

# === Combined Processing ===
@torch.no_grad()
def process_and_downsample(shards_dir: str, norm_path: str, output_dir: str, lead_time: int, target_size=(512, 512), glob_pattern="*.pt"):
    try:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        raise RuntimeError("SLURM_ARRAY_TASK_ID not set. This script is intended for SLURM array jobs.")

    shard_files = sorted(glob.glob(os.path.join(shards_dir, glob_pattern)))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shards_dir}")
    if idx >= len(shard_files):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} is out of range. Found {len(shard_files)} files.")

    os.makedirs(output_dir, exist_ok=True)

    fp = shard_files[idx]
    logging.info(f"[{idx+1}/{len(shard_files)}] Processing: {fp}")

    data = torch.load(fp, map_location="cpu")
    if "inputs" not in data or "targets" not in data:
        raise KeyError(f"Missing 'inputs' or 'targets' in {fp}")

    # Process
    x_raw = data["inputs"].float()                          # (B, 140, 11)
    y_raw = data["targets"]                                 # (B, 1024, 1024)
    x_proc = _PreProcessor(norm_path)(x_raw)                # (B, 140, 9)

    # Downsample targets
    y_raw = y_raw.unsqueeze(1)                              # (B, 1, 1024, 1024)
    y_down = F.interpolate(y_raw, size=target_size, mode="bilinear", align_corners=False)
    y_down = y_down.squeeze(1)                              # (B, 512, 512)

    fname = os.path.basename(fp).replace(".pt", "_proc.pt")
    out_path = os.path.join(output_dir, f"t{lead_time}-{fname}")
    torch.save({"inputs": x_proc, "targets": y_down}, out_path)

    logging.info(f"Saved to: {out_path}")

# === Entrypoint ===
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: preprocess_and_downsample.py <partition> <lead_time>")

    PARTITION = sys.argv[1]
    LEAD_TIME = int(sys.argv[2])

    norm_path = "/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/normalisation/parameters/normalisation.json"
    shards_dir = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard/t{LEAD_TIME}/{PARTITION}"
    output_dir = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed-ds/t{LEAD_TIME}/{PARTITION}_512"

    process_and_downsample(
        shards_dir=shards_dir,
        norm_path=norm_path,
        output_dir=output_dir,
        lead_time=LEAD_TIME
    )
