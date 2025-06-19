# Standard libraries
import os
import math
import json
import bisect
import numpy as np
from typing import Optional

# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

# PyTorch metrics
from torchmetrics.classification import BinaryAUROC


# PyTorch lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


LEAD_TIME = 1
torch.set_float32_matmul_precision("medium")  # or "high"


class FastVariableShardDataset(Dataset):
    """
    Load sharded (inputs, targets) pairs efficiently.

    Each shard is a .pt file containing:
        {
            "inputs":  Tensor of shape (B, N, F),
            "targets": Tensor of shape (B, H, W)
        }

    - Inputs are features per convective core.
    - Targets are preprocessed binary maps (already resized if needed).
    - Only one shard is cached in memory at a time for efficiency.
    """

    def __init__(self, 
                 shards_dir: str, 
                 sizes_file: str,
                 output_size: Optional[tuple[int, int]] = (512, 512)):

        super().__init__()
        self.shard_dir = shards_dir
        self.output_size = output_size  # Not used in this version (preprocessed only)

        # Gather all shard files in the directory (must end with _proc.pt)
        self.shard_files = sorted(
            f for f in os.listdir(shards_dir) if f.endswith("_proc.pt")
        )
        if not self.shard_files:
            raise FileNotFoundError(f"No *_proc.pt files found in {shards_dir}")

        # Load sample counts per shard from sizes JSON
        with open(sizes_file) as f:
            sizes_dict = json.load(f)

        # Store shard sizes and offsets (for global-to-local index lookup)
        self.shard_sizes = [sizes_dict[f] for f in self.shard_files]
        self.shard_offsets = [0]
        for sz in self.shard_sizes[:-1]:
            self.shard_offsets.append(self.shard_offsets[-1] + sz)
        self.total_samples = sum(self.shard_sizes)

        # One-shard RAM cache to avoid repeatedly loading from disk
        self._cache_path: Optional[str] = None
        self._cache_data: Optional[dict] = None

    def __len__(self) -> int:
        """Return total number of samples across all shards."""
        return self.total_samples

    def _load_shard(self, fname: str) -> dict:
        """
        Load a single shard file and cache it in memory.
        If the requested shard is already cached, reuse it.
        """
        if fname != self._cache_path:
            path = os.path.join(self.shard_dir, fname)
            data = torch.load(path, map_location="cpu")
            self._cache_data = data
            self._cache_path = fname
        return self._cache_data

    def __getitem__(self, idx: int):
        """
        Fetch a single sample by global index.
        Automatically finds the correct shard and index within it.
        Returns:
            x: Tensor of shape (N, F)
            y: Tensor of shape (H, W)
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of bounds")

        # Binary search to find the correct shard for this global index
        shard_idx = bisect.bisect_right(self.shard_offsets, idx) - 1
        local_idx = idx - self.shard_offsets[shard_idx]

        # Load the shard (from disk or cache)
        shard_data = self._load_shard(self.shard_files[shard_idx])

        # Extract the sample
        x = shard_data["inputs"][local_idx].float()    # shape: (T, F)
        y = shard_data["targets"][local_idx].float()   # shape: (H, W)

        return x, y


# === Validation ===
VAL_PARTITION = "val"
VAL_SHARDS_DIR = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed/t{LEAD_TIME}/{VAL_PARTITION}"
VAL_SIZES_FILE = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/shard-analysis/t{LEAD_TIME}/{VAL_PARTITION}/{VAL_PARTITION}-t{LEAD_TIME}-shards-proc-sizes.json"

val_ds = FastVariableShardDataset(
    shards_dir=VAL_SHARDS_DIR,
    sizes_file=VAL_SIZES_FILE,
    output_size=None
)

val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True
)


def sanity_check_loader(loader, n_batches_to_check=2, binary_targets=True):
    """
    Perform a basic sanity check on a DataLoader.

    Args:
        loader: torch.utils.data.DataLoader
        n_batches_to_check: number of batches to check (default: 2)
        binary_targets: if True, checks that y is binary (0 or 1)
    """
    for batch_idx, (x, y) in enumerate(loader):
        print(f"[Batch {batch_idx}]")
        print(f"   x shape: {x.shape}")  # e.g., (B, N, F)
        print(f"   y shape: {y.shape}")  # e.g., (B, H, W)

        if torch.isnan(x).any():
            print(f"NaNs in input")
        if torch.isnan(y).any():
            print(f"NaNs in target")
        if binary_targets and not ((y == 0) | (y == 1)).all():
            print(f"Targets not binary — min={y.min().item()}, max={y.max().item()}")

        if batch_idx + 1 >= n_batches_to_check:
            break

    print("✅ Sanity check complete.")


class ShardedSampler(Sampler):
    """
    Custom sampler for sharded datasets.

    Instead of shuffling the entire dataset globally (which would require random disk access),
    this sampler:
      - Shuffles samples **within** each shard
      - Preserves **sequential shard-level access**
      - Optionally applies a per-epoch seed for reproducible shuffling

    This greatly improves performance for datasets stored in separate `.pt` shard files.

    Args:
        shard_sizes: List[int] — number of samples in each shard
        batch_size:  Optional — not used directly, but can inform batching externally
        shuffle:     Whether to shuffle within each shard
        seed:        Optional random seed to control shuffling behaviour (e.g. per epoch)
    """

    def __init__(self, shard_sizes, batch_size=16, shuffle=True, seed=None):
        self.shard_sizes = shard_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Compute cumulative offsets to map global indices to shards
        self.shard_offsets = [0]
        for sz in shard_sizes[:-1]:
            self.shard_offsets.append(self.shard_offsets[-1] + sz)

    def __iter__(self):
        rng = np.random.default_rng(self.seed) if self.shuffle else None
        indices = []

        for shard_idx, size in enumerate(self.shard_sizes):
            offset = self.shard_offsets[shard_idx]
            local_indices = np.arange(offset, offset + size)

            if self.shuffle:
                rng.shuffle(local_indices)

            indices.extend(local_indices.tolist())

        return iter(indices)


    def __len__(self):
        """Total number of samples across all shards."""
        return sum(self.shard_sizes)


TRAIN_PARTITION = "train"
TRAIN_SHARDS_DIR = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed/t{LEAD_TIME}/{TRAIN_PARTITION}"
TRAIN_SIZES_FILE = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/shard-analysis/t{LEAD_TIME}/{TRAIN_PARTITION}/{TRAIN_PARTITION}-t{LEAD_TIME}-shards-proc-sizes.json"

train_ds = FastVariableShardDataset(
    shards_dir=TRAIN_SHARDS_DIR,
    sizes_file=TRAIN_SIZES_FILE,
    output_size=None
)

train_sampler = ShardedSampler(
    shard_sizes=train_ds.shard_sizes,
    batch_size=16,
    shuffle=False,
    seed=None  # set per epoch if needed
)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=16, sampler=train_sampler, num_workers=8, pin_memory=True
)


sanity_check_loader(train_loader, n_batches_to_check=4)


# ──────────────────────── Basic Conv Block ────────────────────────
def conv_block(in_ch, out_ch):
    """3×3 conv → BN → ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

# ──────────────────────── Decoder ────────────────────────
class SimpleDecoder(nn.Module):
    """
    Decoder that upsamples from (B, C, 16, 16) to (B, 1, 1024, 1024)
    using 6× upsampling + 6× ConvBlock.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        channels = [embed_dim, 256, 128, 64, 32, 16, 8]

        self.blocks = nn.ModuleList([
            conv_block(in_ch, out_ch)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])
        self.final = nn.Conv2d(channels[-1], 1, 1)

    def forward(self, x):
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(x)
        return self.final(x)  # (B, 1, 1024, 1024)

# ──────────────────────── Lightning Module ────────────────────────
class Core2MapModel(pl.LightningModule):
    """
    Maps (B, 140, 9) → (B, 1, 1024, 1024) using:
      - Transformer encoder (order-agnostic)
      - Learnable projection to 16×16 grid
      - CNN decoder
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Project input features (9 → embed_dim)
        self.in_proj = nn.Linear(9, embed_dim)

        # 2) Transformer encoder (order-agnostic)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 3) Project transformer output from 140 → 256 tokens (learnable)
        self.token_proj = nn.Linear(140, 256)

        # 4) CNN decoder (from 16×16 to 1024×1024)
        self.decoder = SimpleDecoder(embed_dim)

        # 5) Loss and metrics
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_auc = BinaryAUROC()

    def forward(self, x):
        """
        x: (B, 140, 9)
        return: (B, 1, 1024, 1024)
        """
        b, _, _ = x.shape

        x = self.in_proj(x)              # (B, 140, D)
        x = self.transformer(x)          # (B, 140, D)

        x = x.transpose(1, 2)            # (B, D, 140)
        x = self.token_proj(x)           # (B, D, 256)
        x = x.view(b, -1, 16, 16)        # → (B, D, 16, 16)

        return self.decoder(x)

    def training_step(self, batch, _):
        x, y = batch                     # y: (B, 1024, 1024)
        logits = self(x)                # (B, 1, 1024, 1024)
        loss = self.criterion(logits.squeeze(1), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        y_true = y.int().flatten()
        y_pred = torch.sigmoid(logits).squeeze(1).flatten()
        self.val_auc.update(y_pred, y_true)

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log("val_auc", auc, prog_bar=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# 1. Instantiate the model
model = Core2MapModel(
    embed_dim=128,      # embedding dimension for each core
    num_heads=4,        # number of attention heads
    num_layers=4,       # number of transformer encoder layers
    lr=1e-4             # learning rate
)


# Save best model based on AUC (maximise)
checkpoint_callback = ModelCheckpoint(
    monitor="val_auc",
    mode="max",
    save_top_k=1,
    filename="best-core2map",
    verbose=True
)

# Stop training early if AUC doesn't improve for 5 epochs
early_stop_callback = EarlyStopping(
    monitor="val_auc",     # AUC must improve
    patience=5,            # Stop if no improvement in 5 val epochs
    mode="max",            # Looking for max AUC
    min_delta=0.001,       # AUC must increase by at least 0.001 to count
    verbose=True
)

wandb_logger = WandbLogger(
    project="core2map",
    name="run-v1",   # Optional: give a name to this run
    log_model=True   # Logs checkpoints as artifacts
)


logger = CSVLogger("lightning_logs/", name="core2map")

trainer = Trainer(
    max_epochs=100,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    log_every_n_steps=10,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, early_stop_callback]
)

trainer.fit(model, train_loader, val_loader)

