import os, json, random
from typing import Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

LEAD_TIME   = 1
BATCH_SIZE  = 96
WORLD_SIZE  = int(os.environ.get("WORLD_SIZE", 1))                  # number of processes
LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", 0))                  # GPU idx from the local rank environment variable


# === Iterable Shard Dataset =================================================
class IterableShardDataset(IterableDataset):
    def __init__(self, shard_dir: str, split_by_rank: bool = True):
        super().__init__()
        self.epoch         = 0
        self.split_by_rank = split_by_rank

        all_files = sorted(f for f in os.listdir(shard_dir) if f.endswith("_proc.pt"))[:-1]                 # do not consider the last shard
        if split_by_rank:
            self.shard_files = [f for i, f in enumerate(all_files) if i % WORLD_SIZE == LOCAL_RANK]         # assigns a unique subset of shards to this process (GPU)
        else:
            self.shard_files = all_files

        if not self.shard_files:
            raise RuntimeError(f"[Rank {LOCAL_RANK}] No shard assigned. Total shards: {len(all_files)}")

        self.shard_paths = [os.path.join(shard_dir, f) for f in self.shard_files]

    def set_epoch(self, epoch: int): self.epoch = epoch

    def __iter__(self) -> Iterator:
        rng = random.Random(42 + self.epoch * 1000 + LOCAL_RANK)                # random seed
        shard_paths = self.shard_paths.copy()
        rng.shuffle(shard_paths)

        for path in shard_paths:
            shard   = torch.load(path, map_location="cpu")
            inputs  = shard["inputs"].float()
            targets = shard["targets"].float()  # now (N, 512, 512)

            idx = list(range(len(inputs)))
            rng.shuffle(idx)
            for i in idx:
                yield inputs[i], targets[i]


# === Model Components ========================================================
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        ch = [embed_dim, 512, 256, 128, 64, 32, 16]  # 6 steps for 8→512
        self.blocks = nn.ModuleList([conv_block(c1, c2) for c1, c2 in zip(ch[:-1], ch[1:])])
        self.final = nn.Conv2d(ch[-1], 1, kernel_size=1)

    def forward(self, x):
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(x)
        return self.final(x)



# === Lightning Module ========================================================
class Core2MapModel(pl.LightningModule):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4, lr=1e-4, pos_weight: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters()

        self.in_proj     = nn.Linear(9, embed_dim)
        enc_layer        = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                      dim_feedforward=4*embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)

        # New: permutation-invariant set pooling and projection to 4x4 seed map
        self.set_pool = nn.AdaptiveAvgPool1d(1)
        self.map_proj = nn.Linear(embed_dim, embed_dim * 8 * 8)

        self.decoder = SimpleDecoder(embed_dim)

        if pos_weight:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.val_auc = BinaryAUROC()

    def forward(self, x):
        b, _, _ = x.shape
        x = self.in_proj(x)                        # (B, 140, embed_dim)
        x = self.transformer(x)                    # (B, 140, embed_dim)
        x = x.transpose(1, 2)                      # (B, embed_dim, 140)
        x = self.set_pool(x).squeeze(-1)           # (B, embed_dim)
        x = self.map_proj(x)                       # (B, embed_dim * 4 * 4)
        x = x.view(b, -1, 8, 8)                    # (B, embed_dim, 4, 4)
        return self.decoder(x)                     # (B, 1, 512, 512)

    def training_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        logits = self(x)
        loss = self.criterion(logits, y.unsqueeze(1))  # (B, 1, 512, 512)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        preds = torch.sigmoid(self(x))
        preds_ds = F.max_pool2d(preds, 16).flatten()
        targets_ds = F.max_pool2d(y.unsqueeze(1), 16).flatten()
        self.val_auc.update(preds_ds, targets_ds.int())

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        self.val_auc.reset()

    def on_train_epoch_start(self):
        ds = self.trainer.train_dataloader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# === DataLoader Factory ======================================================
def make_loader(partition: str, batch_size: int):
    shard_dir  = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed/t{LEAD_TIME}/{partition}_512"
    ds         = IterableShardDataset(shard_dir, split_by_rank=True)
    return DataLoader(
        ds, batch_size=batch_size,
        num_workers=3, pin_memory=True, persistent_workers=True
    )


# === Main ====================================================================
def main():
    torch.set_float32_matmul_precision("high")

    train_loader = make_loader("train", BATCH_SIZE)
    val_loader   = make_loader("val",   BATCH_SIZE)

    logger = WandbLogger(project="pancast", name="pancast-10km-smooth", log_model=True) if LOCAL_RANK == 0 else None

    model = Core2MapModel(embed_dim=128, num_heads=4, num_layers=4, lr=1e-4, pos_weight=2.0)

    trainer = pl.Trainer(
        max_epochs=50,
        precision="bf16-mixed",
        strategy="ddp", accelerator="gpu", devices=4,
        logger=logger,
        enable_progress_bar=(LOCAL_RANK == 0),
        log_every_n_steps=1,
        limit_train_batches=1210,
        limit_val_batches=100,
        callbacks=[
            ModelCheckpoint(monitor="val_auc", mode="max",
                            filename="best-core2map", save_top_k=1, verbose=True),
            EarlyStopping(monitor="val_auc", mode="max",
                          patience=10, min_delta=0.001, verbose=True)
        ]
    )

    trainer.fit(model, train_loader, val_loader)
    print("Done!")


if __name__ == "__main__":
    main()
