import optuna
import torch
import torch.nn as nn
import joblib
from model_definitions import ECGCNN
from config import *


""" define the utility during the optimization with optuna """

def optuna_objective(trial, train_loader, val_loader, device="cpu"):
    """ the objectives of the optimization process with optuna""" 
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

    model = ECGCNN(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    criterion = nn.CrossEntropyLoss()

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
    """ running the optuna study """
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda trial: optuna_objective(trial, train_loader, val_loader), n_trials=n_trials)

    return study

def save_best_trial_model(study, trainval_loader, save_path="results/best_model.pth", device="cpu"):
    """ save the best trial model """
    best_params = study.best_trial.params

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

    for epoch in range(30):
        model.train()
        for X_batch, y_batch in trainval_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/30 complete")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

def save_study(study, path):
    """ save the study"""
    joblib.dump(study, path)

def load_study(path):
    """ load the study"""
    return joblib.load(path)
