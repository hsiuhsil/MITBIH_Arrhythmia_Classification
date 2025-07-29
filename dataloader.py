import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from config import *

"""define the dataloaders"""

class ECGDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.X = torch.tensor(data['X'], dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(data['y'], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(data_dir=DATA_DIR, batch_size=64):
    train_ds = ECGDataset(os.path.join(data_dir, "ecg_train.npz"))
    val_ds   = ECGDataset(os.path.join(data_dir, "ecg_val.npz"))
    test_ds  = ECGDataset(os.path.join(data_dir, "ecg_test.npz"))
    trainval_ds = ConcatDataset([train_ds, val_ds])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)
    trainval_loader = DataLoader(trainval_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader, trainval_loader
