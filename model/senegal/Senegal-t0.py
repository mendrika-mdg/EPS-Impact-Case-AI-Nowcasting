import torch
import torch.nn as nn
from joblib import dump
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# --- Data Loading ---
path = '/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Senegal/64x64/full/'
X_train = pd.read_csv(path + 'train-data-senegal-input-t0.csv')
X_test = pd.read_csv(path + 'test-data-senegal-input-t0.csv')
y_train = pd.read_csv(path + 'train-data-senegal-output-map-t1.csv', header=None)
y_test = pd.read_csv(path + 'test-data-senegal-output-map-t1.csv', header=None)

y_train = y_train.to_numpy().reshape(-1, 1, 64, 64)
y_test = y_test.to_numpy().reshape(-1, 1, 64, 64)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# --- Preprocessing ---
for col in ['size1', 'size2', 'size3', 'wp1', 'wp2', 'wp3', 'd1', 'd2', 'd3']:
    for df in [X_train, X_val, X_test]:
        df[col] = np.log1p(df[col])
scaler = StandardScaler()
cols = [c for c in X_train.columns if 'mask' not in c]
X_train[cols] = scaler.fit_transform(X_train[cols])
X_val[cols] = scaler.transform(X_val[cols])
X_test[cols] = scaler.transform(X_test[cols])

# --- Torch Tensors & Loaders ---
def to_tensor(x, y): return torch.tensor(x.values, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
train_ds = TensorDataset(*to_tensor(X_train, y_train))
val_ds = TensorDataset(*to_tensor(X_val, y_val))
test_X, test_y = to_tensor(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)

# --- Model ---
class LitConvUpsample(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 8 * 8 * 16), nn.ReLU(),
            nn.Unflatten(1, (16, 8, 8)),
            nn.Upsample(scale_factor=2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

    def forward(self, x): return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x).view(-1).detach().cpu()
        y = y.view(-1).detach().cpu()
        auc = roc_auc_score(y.numpy(), y_hat.numpy())
        self.log("val_auc", auc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# --- Training ---
model = LitConvUpsample(input_dim=X_train.shape[1])
trainer = pl.Trainer(max_epochs=500, accelerator="auto", callbacks=[
    pl.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10)
])
trainer.fit(model, train_loader, val_loader)

# --- Test AUC ---
model.eval()
with torch.no_grad():
    preds = model(test_X).view(-1).cpu().numpy()
    test_auc = roc_auc_score(test_y.view(-1).cpu().numpy(), preds)
print(f"Test AUC: {test_auc:.4f}")

torch.save(model.state_dict(), "lit_conv_model.pt")
dump(scaler, "scaler.joblib")
