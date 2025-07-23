import time
import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, results_path):
    model.to(device)
    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_path, "best_model.pt"))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    plot_training(history, results_path)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            total_loss += loss.item()
    return total_loss / len(loader), correct / total

def test_report(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds += torch.argmax(preds, dim=1).cpu().tolist()
            all_labels += y_batch.tolist()
    print(classification_report(all_labels, all_preds, target_names=class_names))

def plot_training(history, results_path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "training_plot.png"))
    plt.close()
