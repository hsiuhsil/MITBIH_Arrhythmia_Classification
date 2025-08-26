"""
metrics.py

This module provides visualization and evaluation utilities for model training
and performance analysis in ECG arrhythmia classification.

Includes:
- Confusion matrix plotting (both raw counts and percentage)
- Classification report saving
- ROC and Precision-Recall curve plotting per class
- Training accuracy/loss curve visualization
"""
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from mitbih.utils.config import CLASS_NAMES

def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, title="Confusion Matrix", save_path=None):
    """
    Plots both count-based and percentage-based confusion matrices side by side.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list): List of class names for display.
        title (str): Plot title.
        save_path (str or None): If provided, saves the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp1.plot(cmap="Blues", values_format="d", ax=axs[0], colorbar=True)
    axs[0].set_title("Counts")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=class_names)
    disp2.plot(cmap="viridis", values_format=".1f", ax=axs[1], colorbar=True)
    axs[1].set_title("Percentage")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def save_classification_report(y_true, y_pred, class_names, path):
    """
    Saves a classification report as a formatted text file.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        class_names (list): List of class labels.
        path (str): File path to save the report.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {path}")

def plot_roc_pr_curves(y_true, probs, class_names, save_dir):
    """
    Plots ROC and Precision-Recall curves for each class.

    Args:
        y_true (array-like): True class labels.
        probs (array-like): Predicted probabilities for each class (shape: [n_samples, n_classes]).
        class_names (list): Names of the classes.
        save_dir (str): Directory where the plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], [p[i] for p in probs])
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], [p[i] for p in probs])
        roc_auc = auc(fpr, tpr)

        # ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {class_name}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, f"roc_{class_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # PR
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall - {class_name}")
        plt.grid()
        plt.savefig(os.path.join(save_dir, f"pr_{class_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_training_curves(train_acc, val_acc, train_loss, val_loss, save_path=None):
    """
    Plots training and validation accuracy and loss over epochs.

    Args:
        train_acc (list): Training accuracy per epoch.
        val_acc (list or None): Validation accuracy per epoch.
        train_loss (list): Training loss per epoch.
        val_loss (list or None): Validation loss per epoch.
        save_path (str or None): File path to save the plot. If None, displays the plot.
    """
    epochs = range(1, len(train_acc)+1)
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Acc')
    if val_loss is not None:
        plt.plot(epochs, val_acc, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
        plt.close()
    else:
        plt.show()
