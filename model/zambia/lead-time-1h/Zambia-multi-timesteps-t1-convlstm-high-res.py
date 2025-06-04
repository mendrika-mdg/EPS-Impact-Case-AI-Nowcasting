# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
from joblib import dump

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
)
import wandb


# === ConvLSTM Cell ===
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# === Model Class ===
class MultiStepTabularConvLSTM(nn.Module):
    def __init__(self, input_size_per_step=17, lstm_hidden=[256, 128], projection_size=2, seq_len=3):
        super().__init__()
        self.seq_len = seq_len
        self.projection_size = projection_size

        self.lstm1 = nn.LSTM(input_size=input_size_per_step, hidden_size=lstm_hidden[0], batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden[0], hidden_size=lstm_hidden[1], batch_first=True)

        self.fc = nn.Linear(lstm_hidden[1], projection_size * projection_size * 32)
        self.relu = nn.ReLU()

        self.convlstm = ConvLSTMCell(input_dim=32, hidden_dim=32, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(8, 1, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(8)

    def forward(self, x_seq):
        batch_size = x_seq.size(0)
        latent_maps = []
        for t in range(self.seq_len):
            x_t = x_seq[:, t, :]
            x_t = x_t.unsqueeze(1)
            x_t, _ = self.lstm1(x_t)
            x_t, _ = self.lstm2(x_t)
            x_t = self.relu(self.fc(x_t[:, -1]))
            x_t = x_t.view(-1, 32, self.projection_size, self.projection_size)
            latent_maps.append(x_t)

        h, c = torch.zeros_like(latent_maps[0]), torch.zeros_like(latent_maps[0])
        for t in range(self.seq_len):
            h, c = self.convlstm(latent_maps[t], h, c)

        x = h
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn4(self.conv4(x)))
        x = F.interpolate(x, scale_factor=2)
        return torch.sigmoid(self.out_conv(x))

# === Lightning Module ===
class LightningMultiStepConvLSTM(pl.LightningModule):
    def __init__(self, lr=1e-3, threshold=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiStepTabularConvLSTM()
        self.criterion = nn.BCELoss()
        self.precision = BinaryPrecision(threshold=threshold)
        self.recall = BinaryRecall(threshold=threshold)
        self.f1 = BinaryF1Score(threshold=threshold)
        self.auc = BinaryAUROC()
        self.prc = BinaryAveragePrecision()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.precision.update(y_hat.view(-1), y.view(-1))
        self.recall.update(y_hat.view(-1), y.view(-1))
        self.f1.update(y_hat.view(-1), y.view(-1))
        self.auc.update(y_hat.view(-1), y.view(-1))
        self.prc.update(y_hat.view(-1), y.view(-1).long())
        return loss

    def on_validation_epoch_end(self):
        self.log('val_precision', self.precision.compute(), prog_bar=True, sync_dist=True)
        self.log('val_recall', self.recall.compute(), prog_bar=True, sync_dist=True)
        self.log('val_f1', self.f1.compute(), prog_bar=True, sync_dist=True)
        self.log('val_auc', self.auc.compute(), prog_bar=True, sync_dist=True)
        self.log('val_prc', self.prc.compute(), prog_bar=True, sync_dist=True)
        for metric in [self.precision, self.recall, self.f1, self.auc, self.prc]:
            metric.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


# === Utility ===
def batched_predict(model, x, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=64)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch[0].to(device)).cpu())
    return torch.cat(preds).numpy().flatten()


def plot_roc(name, y_true, y_pred, path=None, **kwargs):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(100 * fpr, 100 * tpr, label=name, **kwargs)
    plt.xlabel("False Alarm Rate [%]")
    plt.ylabel("Hit Rate [%]")
    plt.xlim([-1, 100])
    plt.ylim([0, 105])
    plt.grid(True, linestyle=':')
    plt.legend(loc='lower right')
    if path:
        plt.savefig(path)
    else:
        plt.show()



# === Main Pipeline ===
def main():
    torch.set_float32_matmul_precision('high')
    assert torch.cuda.is_available(), "CUDA is not available"

    lead_time = 1
    base_dir = Path(f'/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Zambia/64x64/full/timesteps/t{lead_time}')

    # Load data
    train_data_X = pd.read_csv(base_dir / f'train-zambia-input-t0-for-lt{lead_time}.csv')
    train_data_y_t = np.load(base_dir / f'train-zambia-output-at-lt{lead_time}.npy')
    test_data_X = pd.read_csv(base_dir / f'test-zambia-input-t0-for-lt{lead_time}.csv')
    test_data_y_t = np.load(base_dir / f'test-zambia-output-at-lt{lead_time}.npy')

    train_data_X, val_data_X, train_data_y_t, val_data_y_t = train_test_split(
        train_data_X, train_data_y_t, test_size=0.3, random_state=12)

    suffixes = ['', '_60', '_120']
    prefixes = ['size', 'wp', 'd']
    indices = ['1', '2']
    cols_to_log = [f'{prefix}{i}{suf}' for suf in suffixes for prefix in prefixes for i in indices]

    for col in cols_to_log:
        for df in [train_data_X, test_data_X, val_data_X]:
            df[col] = np.log1p(df[col].astype(float))

    mask_cols = [col for col in train_data_X.columns if col.startswith('mask')]
    cols_to_scale = [col for col in train_data_X.columns if col not in mask_cols]

    scaler = StandardScaler()
    X_train_scaled = train_data_X.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(train_data_X[cols_to_scale])
    X_val_scaled = val_data_X.copy()
    X_val_scaled[cols_to_scale] = scaler.transform(val_data_X[cols_to_scale])
    X_test_scaled = test_data_X.copy()
    X_test_scaled[cols_to_scale] = scaler.transform(test_data_X[cols_to_scale])

    # Reshape to (batch, 3, 17)
    x_train = X_train_scaled.values.reshape(len(train_data_y_t), 3, 17)
    x_val = X_val_scaled.values.reshape(len(val_data_y_t), 3, 17)
    x_test = X_test_scaled.values.reshape(len(test_data_y_t), 3, 17)

    y_train_tensor = torch.from_numpy(train_data_y_t).float().unsqueeze(1)
    y_val_tensor = torch.from_numpy(val_data_y_t).float().unsqueeze(1)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), y_train_tensor), batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), y_val_tensor), batch_size=16)

    model = LightningMultiStepConvLSTM()

    log_path = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/lead-time-1h/submission/wandblogger/lt{lead_time}"
    Path(log_path).mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(project="zambia-nowcasting", name=f"lead-time-{lead_time}h", save_dir=log_path, group="multi-gpu-run")

    trainer = pl.Trainer(
        accelerator='gpu',
        strategy='ddp',
        devices=4,
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor='val_auc', patience=5),
            ModelCheckpoint(monitor='val_auc', save_top_k=1)
        ],
        log_every_n_steps=20
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint after training
    best_model_path = trainer.checkpoint_callback.best_model_path
    model = LightningMultiStepConvLSTM.load_from_checkpoint(best_model_path).to("cuda")

    # Batched inference function (same as before)
    def batched_predict(model, x, device):
        loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=64)
        preds = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                preds.append(model(batch[0].to(device)).cpu())
        return torch.cat(preds).numpy().flatten()

    # Generate predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_y_train = batched_predict(model, x_train, device)
    pred_y_test = batched_predict(model, x_test, device)

    auc_train = roc_auc_score(train_data_y_t.flatten(), pred_y_train)
    auc_test = roc_auc_score(test_data_y_t.flatten(), pred_y_test)

    plot_roc(f"Train (AUC={auc_train:.2f})", train_data_y_t.flatten(), pred_y_train,
             path=f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/roc-lead-time-{lead_time}-train-convlstm.png")
    plot_roc(f"Test (AUC={auc_test:.2f})", test_data_y_t.flatten(), pred_y_test,
             path=f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/roc-lead-time-{lead_time}-test-convlstm.png")


    # Create directory
    save_dir = Path(f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/lead-time-{lead_time}h/saved")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), save_dir / f"zambia-multistep-convlstm-t{lead_time}.pth")

    # Save scaler
    dump(scaler, save_dir / f"zambia-multistep-convlstm-scaler-t{lead_time}.bin")


if __name__ == "__main__":
    main()
