"""
train_utils.py

This module provides utility functions for training, evaluating, and managing deep learning models
for ECG beat classification using PyTorch. It includes:
- A full training loop with early stopping
- Evaluation metrics and classification report
- Model save/load functions
- Utility to compute class weights for imbalanced data
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20, device='cpu', early_stopping_patience=5):
    """
    Train a PyTorch model with optional early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        criterion (nn.Module): Loss function.
        epochs (int): Number of training epochs.
        device (str): 'cpu' or 'cuda'.
        early_stopping_patience (int): Number of epochs to wait without improvement before stopping.

    Returns:
        Tuple: (trained model, train accuracies, val accuracies, train losses, val losses)
    """
    model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_accs, val_accs, train_losses, val_losses

def evaluate_model(model, test_loader, device='cpu', class_names=None):
    """
    Evaluate a trained model on the test dataset.

    Args:
        model (nn.Module): Trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test data.
        device (str): 'cpu' or 'cuda'.
        class_names (list): List of class names for reporting.

    Returns:
        Tuple: (accuracy, predicted labels, true labels, probability scores)
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    return acc, all_preds, all_labels, all_probs

def save_model(model, path):
    """
    Save a PyTorch model to the specified path.

    Args:
        model (nn.Module): Model to save.
        path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)

def load_model(model_class, path, device='cpu'):
    """
    Load a PyTorch model from a saved state dict.

    Args:
        model_class (nn.Module): The model class to instantiate.
        path (str): Path to the saved model weights.
        device (str): 'cpu' or 'cuda'.

    Returns:
        nn.Module: The loaded model in evaluation mode.
    """
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def get_class_weights(labels, num_classes):
    """
    Compute balanced class weights for imbalanced datasets.

    Args:
        labels (array-like): List or array of labels.
        num_classes (int): Total number of classes.

    Returns:
        torch.Tensor: Class weights as a tensor.
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)
