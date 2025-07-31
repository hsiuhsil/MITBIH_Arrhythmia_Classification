import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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

def get_cv_dataloaders(data_path, batch_size=64, n_splits=5, augment=False, seed=42):
    features_np, labels_np = extract_features_labels_numpy(data_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in skf.split(features_np, labels_np):
        train_dataset = ECGDataset(features_np[train_idx], labels_np[train_idx], augment=augment)
        val_dataset = ECGDataset(features_np[val_idx], labels_np[val_idx], augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        folds.append((train_loader, val_loader))

    return folds
