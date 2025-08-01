import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, TensorDataset,  DataLoader, ConcatDataset
from augmenter import get_ecg_augmenter
from config import *
from collections import Counter

"""define the dataloaders"""

class ECGDataset(Dataset):
    def __init__(self, file_path, augment=False, target_size_per_class=6000):
        data = np.load(file_path)
        self.X = data['X']
        self.y = data['y']
        self.augment = augment

        if self.augment:
            # Count original class distribution
            label_counts = Counter(self.y)
            self.X_aug = []
            self.y_aug = []

            self.augmenter = get_ecg_augmenter()

            for cls in np.unique(self.y):
                cls_indices = np.where(self.y == cls)[0]
                X_cls = self.X[cls_indices]
                y_cls = self.y[cls_indices]

                self.X_aug.extend(X_cls)
                self.y_aug.extend(y_cls)

                if label_counts[cls] < target_size_per_class:
                    num_needed = target_size_per_class - label_counts[cls]
                    # If too few, repeat samples before augmentation
                    reps = int(np.ceil(num_needed / len(X_cls)))
                    X_repeat = np.tile(X_cls, (reps, 1))[:num_needed]

                    X_augmented = self.augmenter.augment(X_repeat)
                    self.X_aug.extend(X_augmented)
                    self.y_aug.extend([cls] * len(X_augmented))

            self.X_aug = np.array(self.X_aug)
            self.y_aug = np.array(self.y_aug)

        else:
            self.X_aug = self.X
            self.y_aug = self.y

    def __len__(self):
        return len(self.y_aug)

    def __getitem__(self, idx):
        x = self.X_aug[idx]
        y = self.y_aug[idx]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # shape: [1, 260]
        y = torch.tensor(y, dtype=torch.long)
        return x, y

def get_dataloaders(data_dir=DATA_DIR, batch_size=64, augment=False):
    train_ds = ECGDataset(os.path.join(data_dir, "ecg_train.npz"), augment=augment)
    val_ds   = ECGDataset(os.path.join(data_dir, "ecg_val.npz"))
    test_ds  = ECGDataset(os.path.join(data_dir, "ecg_test.npz"))
    trainval_ds = ConcatDataset([train_ds, val_ds])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)
    trainval_loader = DataLoader(trainval_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader, trainval_loader

def get_cross_validation_loaders(data_dir=DATA_DIR, batch_size=64, k=5, augment=True):
    """Return a generator of (train_loader, val_loader) for each fold"""
    data = np.load(os.path.join(data_dir, "ecg_train.npz"))
    X = data['X']
    y = data['y']
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Subsets for current fold
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Save temporary npz files to use ECGDataset logic
        np.savez(os.path.join(data_dir, "ecg_fold_train.npz"), X=X_train, y=y_train)
        np.savez(os.path.join(data_dir, "ecg_fold_val.npz"),   X=X_val,   y=y_val)

        train_ds = ECGDataset(os.path.join(data_dir, "ecg_fold_train.npz"), augment=augment)
        val_ds   = ECGDataset(os.path.join(data_dir, "ecg_fold_val.npz"), augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size)

        yield fold, train_loader, val_loader

def get_full_trainval_loader(data_dir, batch_size=64, augment=True):
    train_npz = np.load(os.path.join(data_dir, "ecg_train.npz"))
    val_npz = np.load(os.path.join(data_dir, "ecg_val.npz"))

    X_combined = np.concatenate([train_npz['X'], val_npz['X']], axis=0)
    y_combined = np.concatenate([train_npz['y'], val_npz['y']], axis=0)

    dataset = TensorDataset(torch.tensor(X_combined[:, None, :], dtype=torch.float32),
                            torch.tensor(y_combined, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
