import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ----------------------
# Data Loading
# ----------------------

# Paths
path = '/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Senegal/64x64/full/'

test_data_X = pd.read_csv(path + 'test-data-senegal-input-t0.csv')
test_data_y_t1 = pd.read_csv(path + 'test-data-senegal-output-map-t1.csv', header=None)
train_data_X = pd.read_csv(path + 'train-data-senegal-input-t0.csv')
train_data_y_t1 = pd.read_csv(path + 'train-data-senegal-output-map-t1.csv', header=None)

# Reshape target to (n, 64, 64)
resolution_y, resolution_x = 64, 64
test_data_y_t1 = test_data_y_t1.to_numpy().reshape(len(test_data_y_t1), resolution_y, resolution_x)
train_data_y_t1 = train_data_y_t1.to_numpy().reshape(len(train_data_y_t1), resolution_y, resolution_x)

# ----------------------
# Preprocessing
# ----------------------

# Split
train_data_X, val_data_X, train_y_t1, val_y_t1 = train_test_split(
    train_data_X, train_data_y_t1, test_size=0.3, random_state=12
)

# Log-transform
for col in ['size1', 'size2', 'size3', 'wp1', 'wp2', 'wp3', 'd1', 'd2', 'd3']:
    for df in [train_data_X, val_data_X, test_data_X]:
        df[col] = np.log1p(df[col])

# Scale
mask_cols = ['mask1', 'mask2', 'mask3']
cols_to_scale = [col for col in train_data_X.columns if col not in mask_cols]
scaler = StandardScaler()

X_train_scaled = train_data_X.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(train_data_X[cols_to_scale])

X_val_scaled = val_data_X.copy()
X_val_scaled[cols_to_scale] = scaler.transform(val_data_X[cols_to_scale])

X_test_scaled = test_data_X.copy()
X_test_scaled[cols_to_scale] = scaler.transform(test_data_X[cols_to_scale])

# Convert to torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = torch.tensor(X_train_scaled.values, dtype=torch.float32).to(device)
y_train = torch.tensor(train_y_t1, dtype=torch.float32).unsqueeze(1).to(device)

x_val = torch.tensor(X_val_scaled.values, dtype=torch.float32).to(device)
y_val = torch.tensor(val_y_t1, dtype=torch.float32).unsqueeze(1).to(device)

x_test = torch.tensor(X_test_scaled.values, dtype=torch.float32).to(device)
y_test = torch.tensor(test_data_y_t1, dtype=torch.float32).unsqueeze(1).to(device)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=1024, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=1024)

# ----------------------
# Model Definition
# ----------------------

class UpsampleConvNet(nn.Module):
    def __init__(self, input_dim, initial_size=8):
        super(UpsampleConvNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, initial_size * initial_size * 16)

        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)

        self.initial_size = initial_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = x.view(-1, 16, self.initial_size, self.initial_size)

        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv3(x))

        x = torch.sigmoid(self.conv_out(x))
        return x

model = UpsampleConvNet(input_dim=x_train.shape[1]).to(device)

# ----------------------
# Training
# ----------------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

best_auc = 0
patience, patience_counter = 10, 0

for epoch in range(500):
    model.train()
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(x_val).view(-1).cpu().numpy()
        val_labels = y_val.view(-1).cpu().numpy()
        val_auc = roc_auc_score(val_labels, val_preds)

    print(f"Epoch {epoch+1}, Val AUC: {val_auc:.4f}")
    if val_auc > best_auc:
        best_auc = val_auc
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

model.load_state_dict(best_model_state)

# ----------------------
# Evaluation
# ----------------------

with torch.no_grad():
    pred_y_train = model(x_train).view(-1).cpu().numpy()
    pred_y_test = model(x_test).view(-1).cpu().numpy()

train_auc = roc_auc_score(y_train.view(-1).cpu().numpy(), pred_y_train)
test_auc = roc_auc_score(y_test.view(-1).cpu().numpy(), pred_y_test)

print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")

# ----------------------
# Reliability Curve
# ----------------------

def reliability_curve(y_true, y_pred, bin_size=0.1, min_predictions_per_bin=50):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    bins = np.arange(0, 1 + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_positive_rates = []
    bin_centers_output = []
    bin_counts = []

    for lower, upper, center in zip(bins[:-1], bins[1:], bin_centers):
        in_bin = (y_pred >= lower) & (y_pred < upper)
        count_in_bin = np.sum(in_bin)

        if count_in_bin >= min_predictions_per_bin:
            observed_rate = np.mean(y_true[in_bin])
            bin_positive_rates.append(round(observed_rate, 3))
            bin_centers_output.append(round(center, 3))
            bin_counts.append(count_in_bin)

    return bin_centers_output, bin_positive_rates, bin_counts

prob_pred, prob_true, no_pred_per_bin = reliability_curve(y_test.view(-1).cpu().numpy(), pred_y_test)

plt.figure(figsize=(5,4))
plt.plot(prob_pred, prob_true, label='Reliability')
plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), linestyle='--', color='gray')

scaled_hist = [i / (1.1 * np.max(no_pred_per_bin)) for i in no_pred_per_bin]
plt.bar(prob_pred, scaled_hist, width=0.1, edgecolor="green", fill=False)

plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.legend(loc='upper left')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
