import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_and_split_data(X, y, test_size=0.2, val_size=0.1, batch_size=64, seed=42):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size, random_state=seed)

    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), torch.LongTensor(y_val)), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test)), batch_size=batch_size)

    return train_loader, val_loader, test_loader
