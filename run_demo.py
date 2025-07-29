"""save beats into a npz for demo"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from config import OUTPUT_DIR, CLASS_NAMES, DEMO_PATH, MODEL_SAVE_PATH, DEVICE
from predict import load_model_and_predict
from model_definitions import ECGCNN

def create_demo_npz(save_path=DEMO_PATH, num_per_class=3):
    """
    Save some demo data (from the training dataset) to a .npz file.
    """
    train = np.load(os.path.join(OUTPUT_DIR,"ecg_train.npz"))
    X, y = train["X"], train["y"]

    # Group indices by class
    idx_by_cls = defaultdict(list)
    for idx, label in enumerate(y):
        idx_by_cls[label].append(idx)

    # Randomly choose 'num_per_class' samples from each class
    selected_indices = []
    for label, indices in idx_by_cls.items():
        if len(indices) >= num_per_class:
            selected_indices.extend(random.sample(indices, num_per_class))
        else:
            # Use all if not enough samples
            selected_indices.extend(indices)

    X_demo = X[selected_indices]
    y_demo = y[selected_indices]

    np.savez(DEMO_PATH, X=X_demo, y=y_demo)
    print("Saved demo_beats.npz with shape:", X_demo.shape)

def load_demo_data(npz_path=DEMO_PATH):
    """
    Load demo data from .npz into a list of (label, beat) tuples.
    """
    loaded = np.load(npz_path)
    X, y = loaded['X'], loaded['y']

    true_labels = [CLASS_NAMES[i] for i in y]

    return X, true_labels

def plot_ecg_predictions(X, y_true, y_pred, save_path=None):
    """
    Plot ECG beats with true and predicted labels.

    Args:
        X: numpy array of shape (N, 260), one ECG beat per row.
        y_true: list or array of true class names.
        y_pred: list or array of predicted class names.
        save_path: if provided, saves the plot to this path.
    """

    num_samples = len(X)
    n_rows, n_cols = 3, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True, sharey=True)

    axes = axes.flatten()

    time_axis = np.arange(X.shape[1]) / 360.0  # in seconds

    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < num_samples:
            ax.plot(time_axis, X[i], color='black')
            ax.set_title(f"True: {y_true[i]} | Pred: {y_pred[i]}", fontsize=9)
        else:
            ax.axis('off')  # Hide unused subplots

        ax.set_xticks(np.linspace(0, time_axis[-1], 4))
        ax.grid(True)

    # Only set xlabel on bottom row
    for i in range(n_rows * n_cols):
        if i // n_cols == n_rows - 1:
            axes[i].set_xlabel("Time (sec)", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def main():
    # Step 1: Load or create demo .npz
    if not os.path.exists(DEMO_PATH):
        create_demo_npz(DEMO_PATH)

    # Step 2: Load demo beats
    X, true_classes = load_demo_data(DEMO_PATH)

    # Step 3: Predict
    print("Running predictions on demo beats...\n")
    pred_classes = load_model_and_predict(X, ECGCNN, model_path=MODEL_SAVE_PATH, device=DEVICE)
    
    # Step 4: Plot
    plot_ecg_predictions(X, true_classes, pred_classes, save_path=None)

if __name__ == "__main__":
    main()
