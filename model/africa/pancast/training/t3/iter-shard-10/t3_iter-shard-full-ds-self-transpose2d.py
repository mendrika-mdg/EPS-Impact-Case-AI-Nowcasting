import os, random
from typing import Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

LEAD_TIME   = 3
BATCH_SIZE  = 96
WORLD_SIZE  = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", 0))


def compute_fss(preds, targets, window: int = 16):
    """
    Compute Fractions Skill Score between predicted and target maps.
    Inputs are assumed to be (B, 1, H, W) tensors with values in [0, 1].
    """
    pool = nn.AvgPool2d(kernel_size=window, stride=1, padding=window // 2)
    preds_bin = (preds > 0.5).float()  # Optional: or keep raw probs

    p = pool(preds_bin)
    t = pool(targets)

    mse = F.mse_loss(p, t, reduction='mean')
    ref = F.mse_loss(p, p, reduction='mean') + F.mse_loss(t, t, reduction='mean')

    return 1.0 - (mse / (ref + 1e-6))  # add epsilon to avoid div-by-zero


# === SELF FSS LOSS ==========================================================
class SelfFSSLoss(nn.Module):
    def __init__(self, window_size: int = 16, pos_weight: Optional[float] = None, alpha: float = 0.3):
        """
        Spatially Enhanced Loss: alpha * BCE + (1 - alpha) * FSS-style MSE
        """
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=window_size, stride=1, padding=window_size // 2)
        self.alpha = alpha
        if pos_weight:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = self.bce(logits, targets)
        fss_loss = F.mse_loss(self.pool(probs), self.pool(targets))
        return self.alpha * bce_loss + (1 - self.alpha) * fss_loss


# === Iterable Shard Dataset =================================================
class IterableShardDataset(IterableDataset):
    def __init__(self, shard_dir: str, split_by_rank: bool = True):
        super().__init__()
        self.epoch = 0
        self.split_by_rank = split_by_rank

        all_files = sorted(f for f in os.listdir(shard_dir) if f.endswith("_proc.pt"))
        if split_by_rank:
            self.shard_files = [f for i, f in enumerate(all_files) if i % WORLD_SIZE == LOCAL_RANK]
        else:
            self.shard_files = all_files

        if not self.shard_files:
            raise RuntimeError(f"[Rank {LOCAL_RANK}] No shard assigned. Total shards: {len(all_files)}")

        self.shard_paths = [os.path.join(shard_dir, f) for f in self.shard_files]

    def set_epoch(self, epoch: int): self.epoch = epoch

    def __iter__(self) -> Iterator:
        rng = random.Random(42 + self.epoch * 1000 + LOCAL_RANK)
        shard_paths = self.shard_paths.copy()
        rng.shuffle(shard_paths)

        for path in shard_paths:
            shard = torch.load(path, map_location="cpu")
            inputs = shard["inputs"].float()
            targets = shard["targets"].float()

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
        ch = [embed_dim, 512, 256, 128, 64, 32]

        self.blocks = nn.ModuleList()
        for c1, c2 in zip(ch[:-1], ch[1:]):
            block = nn.Sequential(
                nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2, c2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True)
            )
            self.blocks.append(block)

        self.final = nn.Conv2d(ch[-1], 1, kernel_size=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x)




# === Lightning Module ========================================================
class Core2MapModel(pl.LightningModule):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4, lr=1e-4, pos_weight: Optional[float] = None):
        super().__init__()
        self.save_hyperparameters()

        self.in_proj = nn.Linear(9, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                               dim_feedforward=4 * embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)

        self.set_pool = nn.AdaptiveAvgPool1d(1)
        self.map_proj = nn.Linear(embed_dim, embed_dim * 16 * 16)

        self.decoder = SimpleDecoder(embed_dim)
        self.criterion = SelfFSSLoss(window_size=4, pos_weight=pos_weight, alpha=0.3)

        self.val_auc = BinaryAUROC()

    def forward(self, x):
        b, _, _ = x.shape
        x = self.in_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.set_pool(x).squeeze(-1)
        x = self.map_proj(x)
        x = x.view(b, -1, 16, 16)
        return self.decoder(x)

    def training_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        logits = self(x)
        loss = self.criterion(logits, y.unsqueeze(1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        preds = torch.sigmoid(self(x))
        preds_ds = F.max_pool2d(preds, 4).flatten()
        targets_ds = F.max_pool2d(y.unsqueeze(1), 4).flatten()
        self.val_auc.update(preds_ds, targets_ds.int())
        fss4 = compute_fss(preds, y.unsqueeze(1), window=4)
        self.log("val", fss4, prog_bar=True, sync_dist=True)

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
    shard_dir = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed-ds/t{LEAD_TIME}/{partition}_512"
    ds = IterableShardDataset(shard_dir, split_by_rank=True)
    return DataLoader(
        ds, batch_size=batch_size,
        num_workers=3, pin_memory=True, persistent_workers=True
    )


# === Main ====================================================================
def main():
    torch.set_float32_matmul_precision("high")

    train_loader = make_loader("train", BATCH_SIZE)
    val_loader = make_loader("val", BATCH_SIZE)

    logger = WandbLogger(project="pancast-t3", name="pancast-10km-deconv", log_model=True) if LOCAL_RANK == 0 else None

    model = Core2MapModel(embed_dim=128, num_heads=4, num_layers=4, lr=1e-4, pos_weight=None)

    trainer = pl.Trainer(
        max_epochs=20,
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
                          patience=5, min_delta=0.01, verbose=True)
        ]
    )

    trainer.fit(model, train_loader, val_loader)
    print("Done!")


if __name__ == "__main__":
    main()
