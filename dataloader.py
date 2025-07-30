import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tsaug import TimeWrap, Drift, AddNoise
from config import *

"""define the dataloaders"""

class ECGDataset(Dataset):
    def __init__(self, file_path, augment=False):
        data = np.load(file_path)
        self.X = data['X']
        self.y = data['y']
        self.augment = augment

        if self.augment:
            # Define the augmentation pipeline
            self.augmenter = (
                TimeWarp(n_speed_change=3, max_speed_ratio=2) * 0.5
                + Drift(max_drift=(0.05, 0.1), n_drift_points=5) * 0.5
                + AddNoise(scale=0.05)
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            x = self.augmenter.augment(x[np.newaxis, :])[0]

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # shape: [1, 260]
        y = torch.tensor(y, dtype=torch.long)
        return x, y

def get_dataloaders(data_dir=DATA_DIR, batch_size=64):
    train_ds = ECGDataset(os.path.join(data_dir, "ecg_train.npz"), augment=True)  # Augment here
    val_ds   = ECGDataset(os.path.join(data_dir, "ecg_val.npz"))
    test_ds  = ECGDataset(os.path.join(data_dir, "ecg_test.npz"))
    trainval_ds = ConcatDataset([train_ds, val_ds])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)
    trainval_loader = DataLoader(trainval_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader, trainval_loader

