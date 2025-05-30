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


# === Model Classes ===
class LightLSTMToConv64(nn.Module):
    def __init__(self, input_size=17, hidden_sizes=[64, 32], initial_size=4):
        super().__init__()
        self.initial_size = initial_size
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[1], initial_size * initial_size * 8)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.relu(self.fc(x[:, -1]))
        x = x.view(-1, 8, self.initial_size, self.initial_size)
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2)
        return torch.sigmoid(self.out_conv(x))


class LightningUpsampleModel(pl.LightningModule):
    def __init__(self, input_dim=17, lr=1e-3, threshold=0.15):
        super().__init__()
        self.save_hyperparameters()
        self.model = LightLSTMToConv64(input_size=input_dim)
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
        y = y.long()
        loss = self.criterion(y_hat, y.float())
        self.log("val_loss", loss)
        self.precision.update(y_hat.view(-1), y.view(-1))
        self.recall.update(y_hat.view(-1), y.view(-1))
        self.f1.update(y_hat.view(-1), y.view(-1))
        self.auc.update(y_hat.view(-1), y.view(-1))
        self.prc.update(y_hat.view(-1), y.view(-1))
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


# === Main ===
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

    # Preprocess
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

    # Reshape
    num_time_steps = 3
    num_features = 17
    x_train = X_train_scaled.values.reshape(len(train_data_y_t), num_time_steps, num_features)
    x_val = X_val_scaled.values.reshape(len(val_data_y_t), num_time_steps, num_features)
    x_test = X_test_scaled.values.reshape(len(test_data_y_t), num_time_steps, num_features)

    # Tensors
    y_train_tensor = torch.from_numpy(train_data_y_t).float().unsqueeze(1)
    y_val_tensor = torch.from_numpy(val_data_y_t).float().unsqueeze(1)
    y_test_tensor = torch.from_numpy(test_data_y_t).float().unsqueeze(1)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), y_train_tensor), batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), y_val_tensor), batch_size=16)

    # Model + Logger
    model = LightningUpsampleModel()

    log_path = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/lead-time-1h/submission/wandblogger/lt{lead_time}"
    Path(log_path).mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(project="zambia-nowcasting", 
                               name=f"lead-time-{lead_time}h", 
                               save_dir=log_path,
                               group="multi-gpu-run",
                            )

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        strategy='ddp',
        devices=4,
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor='val_auc', patience=3),
            ModelCheckpoint(monitor='val_auc', save_top_k=1)
        ],
        log_every_n_steps=20
    )

    trainer.fit(model, train_loader, val_loader)

    # Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = trainer.checkpoint_callback.best_model_path
    model = LightningUpsampleModel.load_from_checkpoint(best_model_path).to(device)

    pred_y_train = batched_predict(model, x_train, device)
    pred_y_test = batched_predict(model, x_test, device)

    auc_train = roc_auc_score(train_data_y_t.flatten(), pred_y_train)
    auc_test = roc_auc_score(test_data_y_t.flatten(), pred_y_test)

    plot_roc(f"Train (AUC={auc_train:.2f})", train_data_y_t.flatten(), pred_y_train,
             path=f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/roc-lead-time-{lead_time}-train.png")
    plot_roc(f"Test (AUC={auc_test:.2f})", test_data_y_t.flatten(), pred_y_test,
             path=f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/roc-lead-time-{lead_time}-test.png")

    # Save model + scaler
    model_save_path = Path(f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/lead-time-{lead_time}h/saved")
    model_save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path / f"zambia-model-multi-timesteps-64x64-t{lead_time}.pth")
    dump(scaler, model_save_path / f"zambia-model-multi-timesteps-64x64-t{lead_time}.bin")


if __name__ == "__main__":
    main()
