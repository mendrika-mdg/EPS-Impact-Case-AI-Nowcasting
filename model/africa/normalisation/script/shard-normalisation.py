import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset

class StormNowcastingShardedDataset(Dataset):
    def __init__(self, shard_dir, norm_path, transform=None):
        self.shard_dir = shard_dir
        self.shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".pt")])
        self.transform = transform

        # Load normalisation
        with open(norm_path, "r") as f:
            self.norm = json.load(f)

        # Build index
        self.index_map = []
        self.shard_sizes = []
        for shard_idx, fname in enumerate(self.shard_files):
            shard_path = os.path.join(shard_dir, fname)
            shard = torch.load(shard_path, map_location="cpu")
            num_samples = shard["inputs"].shape[0]
            self.shard_sizes.append(num_samples)
            for i in range(num_samples):
                self.index_map.append((shard_idx, i))

        # Cache for loaded shard
        self.loaded_shard_idx = None
        self.loaded_shard = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shard_idx, sample_idx = self.index_map[idx]
        if shard_idx != self.loaded_shard_idx:
            shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
            self.loaded_shard = torch.load(shard_path, map_location="cpu")
            self.loaded_shard_idx = shard_idx

        inputs = self.loaded_shard["inputs"][sample_idx].float()
        targets = self.loaded_shard["targets"][sample_idx].float()

        inputs = self.preprocess_inputs(inputs)

        if self.transform:
            inputs, targets = self.transform(inputs, targets)

        return inputs, targets

    def preprocess_inputs(self, x):
        x = x[:, 1:]
        out = torch.zeros((x.shape[0], 10))

        month = x[:, 0]
        out[:, 0] = torch.sin(2 * np.pi * (month - 1) / 12.0)
        out[:, 1] = torch.cos(2 * np.pi * (month - 1) / 12.0)

        hour = x[:, 2]
        minute = x[:, 3]
        time_in_hours = hour + minute / 60.0
        out[:, 2] = torch.sin(2 * np.pi * time_in_hours / 24.0)
        out[:, 3] = torch.cos(2 * np.pi * time_in_hours / 24.0)

        lat = x[:, 4]
        out[:, 4] = (lat - self.norm["lat_min"]) / (self.norm["lat_max"] - self.norm["lat_min"])

        lon = x[:, 5]
        out[:, 5] = (lon - self.norm["lon_min"]) / (self.norm["lon_max"] - self.norm["lon_min"])

        wp = x[:, 6]
        out[:, 6] = torch.log1p(wp) / self.norm["wp_max"]

        tir = x[:, 7]
        out[:, 7] = (tir - self.norm["tir_min"]) / (self.norm["tir_max"] - self.norm["tir_min"])

        size = x[:, 8]
        out[:, 8] = size / self.norm["size_max"]

        out[:, 9] = x[:, 9]
        return out
