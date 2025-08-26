"""
run_demo.py

This script demonstrates loading a trained ECG classifier to:
1. Generate a demo set of ECG beats from the training dataset.
2. Run predictions on the demo beats.
3. Plot and save the results with true and predicted labels.

Dependencies:
- Assumes a trained model and saved dataset exist in configured directories.
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from config import OUTPUT_DIR, PLOT_DIR, CLASS_NAMES, DEMO_PATH, MODEL_SAVE_PATH, DEVICE
from predict import load_model_and_predict
from model_definitions import ECGCNN

def create_demo_npz(save_path=DEMO_PATH, num_per_class=3):
    """
    Create and save a demo subset of ECG beats for visualization.

    Selects a small number of samples from each class in the training set and saves them as an `.npz` file.

    Args:
        save_path (str): Path to save the demo `.npz` file.
        num_per_class (int): Number of beats per class to include.
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
    Load ECG beat samples and corresponding labels from a demo `.npz` file.

    Args:
        npz_path (str): Path to the saved `.npz` file.

    Returns:
        Tuple[np.ndarray, List[str]]: ECG beats and their corresponding class names.
    """
    loaded = np.load(npz_path)
    X, y = loaded['X'], loaded['y']

    true_labels = [CLASS_NAMES[i] for i in y]

    return X, true_labels

def plot_ecg_predictions(X, y_true, y_pred, save_path=None):
    """
    Plot ECG beat waveforms along with their true and predicted class labels.

    Args:
        X (np.ndarray): Array of shape (N, 260), where each row is a beat.
        y_true (List[str]): Ground truth class names.
        y_pred (List[str]): Predicted class names.
        save_path (str, optional): Path to save the plot image. If None, display instead.
    """
    num_samples = len(X)
    n_rows, n_cols = 5, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 8), sharex=True, sharey=True)

    axes = axes.flatten()

    time_axis = np.arange(X.shape[1]) / 360.0  # in seconds

    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < num_samples:
            ax.plot(time_axis, X[i], color='blue')
            ax.set_title(f"True: {y_true[i]} | Pred: {y_pred[i]}", fontsize=9)
        else:
            ax.axis('off')  # Hide unused subplots

        tick_locs = np.linspace(0, time_axis[-1], 4)
        ax.set_xticks(tick_locs)
        formatted_labels = [f'{val:.3f}' for val in tick_locs]
        ax.set_xticklabels(formatted_labels)
        ax.grid(True)

    # Only set xlabel on bottom row
    for i in range(n_rows * n_cols):
        if i // n_cols == n_rows - 1:
            axes[i].set_xlabel("Time (sec)", fontsize=9)

        if i % n_cols == 0:
            axes[i].set_ylabel("Amplitude (arb.)", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    """
    Main function to execute the demo pipeline:
    1. Generate or load demo ECG beats.
    2. Predict classes using a trained model.
    3. Plot and save the prediction results.
    """
    # Step 1: Load or create demo .npz
    if not os.path.exists(DEMO_PATH):
        create_demo_npz(DEMO_PATH)

    # Step 2: Load demo beats
    X, true_classes = load_demo_data(DEMO_PATH)

    # Step 3: Predict
    print("Running predictions on demo beats...\n")
    pred_classes = load_model_and_predict(X, ECGCNN, model_path=MODEL_SAVE_PATH, device=DEVICE)
    
    # Step 4: Plot
    save_demo_path = curve_path = os.path.join(PLOT_DIR, "demo.png")
    plot_ecg_predictions(X, true_classes, pred_classes, save_path=save_demo_path)

if __name__ == "__main__":
    main()
