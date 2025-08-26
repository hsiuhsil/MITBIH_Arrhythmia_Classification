"""
optuna_utils.py

This module defines utility functions for hyperparameter optimization using Optuna
to tune an ECG classification model (ECGCNN). It includes model objective definition,
study management, and retraining the best model.
"""

import optuna
import torch
import torch.nn as nn
import joblib
from mitbih.models.model_definitions import ECGCNN
from mitbih.utils.config import *
from collections import Counter

def compute_soft_class_weights(loader, device, alpha=0.5):
    """
    Compute class weights with optional softening for imbalanced classification.

    Args:
        loader (DataLoader): DataLoader for computing class frequencies.
        device (str): Device to send the weights ('cpu' or 'cuda').
        alpha (float): Softening factor (0 = no weighting, 1 = full weighting).

    Returns:
        torch.Tensor: Softened class weights tensor on the specified device.
    """
    label_counts = Counter()
    for _, labels in loader:
        label_counts.update(labels.numpy().tolist())

    total = sum(label_counts.values())
    num_classes = len(label_counts)
    raw_weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        raw_weights.append(total / (num_classes * count))

    raw_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)

    # Soften the weights: interpolate between 1 and the weight
    soft_weights = (1 - alpha) * torch.ones_like(raw_weights) + alpha * raw_weights
    return soft_weights

def optuna_objective(trial, train_loader, val_loader, device="cpu"):
    """
    Objective function used by Optuna to optimize ECGCNN hyperparameters.

    Args:
        trial (optuna.trial.Trial): Current Optuna trial.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Torch device.

    Returns:
        float: Validation error (1 - accuracy) to minimize.
    """
    # Suggested hyperparams

    model_params = {
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "filters1": trial.suggest_categorical("filters1", [16, 32, 64]),
        "filters2": trial.suggest_categorical("filters2", [32, 64, 128]),
        "fc1_size": trial.suggest_categorical("fc1_size", [64, 130, 256]),
        "use_third_conv": trial.suggest_categorical("use_third_conv", [False, True]),
    }
    if model_params["use_third_conv"]:
        model_params["filters3"] = trial.suggest_categorical("filters3", [64, 128, 256])

    optimizer_params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    }
 
    # model setup
    model = ECGCNN(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    # class weights
    alpha = trial.suggest_float("class_weight_alpha", 0.0, 1.0)
    class_weights = compute_soft_class_weights(train_loader, device, alpha=alpha)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # training loop
    trial_number = trial.number
    short_mode = trial_number < 5
    num_epochs = 10 if short_mode else 25
    max_batches = 50 if short_mode else None
    
    best_acc, patience, epochs_no_improve = 0, 3, 0
    for epoch in range(num_epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            if max_batches and i >= max_batches:
                break
        
        # Validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 1.0 - best_acc  # minimize error

def run_optuna_study(train_loader, val_loader, n_trials=30):
    """
    Run an Optuna hyperparameter optimization study.

    Args:
        train_loader (DataLoader): Training set loader.
        val_loader (DataLoader): Validation set loader.
        n_trials (int): Number of optimization trials.

    Returns:
        optuna.Study: Completed Optuna study object.
    """
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda trial: optuna_objective(trial, train_loader, val_loader), n_trials=n_trials)

    return study

def get_or_run_study(study_path, train_loader, val_loader, n_trials=30):
    """
    Load a saved Optuna study if available; otherwise run a new study and save it.

    Args:
        study_path (str): File path to save or load the Optuna study.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        n_trials (int): Number of trials if creating a new study.

    Returns:
        optuna.Study: Loaded or newly created Optuna study.
    """
    if os.path.exists(study_path):
        print(f"Loading existing Optuna study from: {study_path}")
        study = load_study(study_path)
    else:
        print("No existing study found. Running new Optuna study...")
        study = run_optuna_study(train_loader, val_loader, n_trials)
        save_study(study, study_path)
        print(f"Study saved to: {study_path}")
    return study

def save_best_trial_model(params_or_study, trainval_loader, num_epoches=30, save_path="results/best_model.pth", device="cpu"):
    """
    Retrain the model using the best parameters and save it.

    Args:
        params_or_study (dict or optuna.Study): Best trial parameters or Optuna study object.
        trainval_loader (DataLoader): Combined training and validation data.
        num_epoches (int): Number of training epochs.
        save_path (str): Path to save the trained model.
        device (str): Device for training.

    Returns:
        Tuple: (trained model, list of train accuracies, list of train losses)
    """
    if isinstance(params_or_study, dict):
        best_params = params_or_study
    else:
        best_params = params_or_study.best_trial.params

    best_model_params = {
        "kernel_size": best_params['kernel_size'],
        "dropout": best_params['dropout'],
        "filters1": best_params['filters1'],
        "filters2": best_params['filters2'],
        "fc1_size": best_params['fc1_size'],
        "use_third_conv": best_params['use_third_conv']
    }
    if best_model_params["use_third_conv"]:
        best_model_params["filters3"] = best_params['filters3']

    best_optimizer_params = {
        "lr": best_params['lr'],
        "weight_decay": best_params['weight_decay']
    }

    model = ECGCNN(**best_model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_optimizer_params["lr"], weight_decay=best_optimizer_params["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # Initialize tracking
    train_loss_list = []
    train_acc_list = []

    for epoch in range(num_epoches):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in trainval_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)

        print(f"Epoch {epoch+1}/30 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, train_acc_list, train_loss_list

def save_study(study, path):
    """
    Save an Optuna study object to disk.

    Args:
        study (optuna.Study): The study object to save.
        path (str): Destination file path.
    """
    joblib.dump(study, path)

def load_study(path):
    """
    Load an Optuna study object from disk.

    Args:
        path (str): Path to the saved study.

    Returns:
        optuna.Study: Loaded Optuna study.
    """
    return joblib.load(path)
