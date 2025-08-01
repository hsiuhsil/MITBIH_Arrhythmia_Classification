"""
check_augmentation.py

This script visualizes the effects of ECG data augmentation using the `tsaug` library.
It selects a small number of samples (e.g., 3) from each ECG class, applies the defined
augmentation pipeline, and plots the original vs. augmented signals side-by-side.

Purpose:
- Help verify that augmentations are applied correctly and meaningfully.
- Provide visual intuition about the diversity introduced by augmentation.

Main Steps:
1. Load preprocessed ECG training data.
2. Select a few samples per class.
3. Apply the augmenter defined in `augmenter.py`.
4. Plot and save the comparison figures.

Output:
- A figure saved at `PLOT_DIR/augmented_comparison.png` showing 3 original and augmented
  samples for each class.

Dependencies:
- numpy, matplotlib
- Local modules: `config.py`, `augmenter.py`
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from config import OUTPUT_DIR, PLOT_DIR, CLASS_NAMES
from augmenter import get_ecg_augmenter

SAMPLES_PER_CLASS = 3
sample_rate = 360 #Hz
NUM_CLASSES = len(CLASS_NAMES)
TOTAL_SAMPLES = SAMPLES_PER_CLASS * NUM_CLASSES

# Load original dataset
data = np.load(os.path.join(OUTPUT_DIR, "ecg_train.npz"))
X = data["X"]
y = data["y"]
print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")

# ========== SELECT 3 SAMPLES PER CLASS ==========
selected_X = []
selected_y = []

for class_idx, label in enumerate(CLASS_NAMES):
    class_samples = X[y == class_idx]
    indices = np.random.choice(len(class_samples), size=SAMPLES_PER_CLASS, replace=False)
    selected_X.append(class_samples[indices])
    selected_y.extend([label] * SAMPLES_PER_CLASS)

selected_X = np.vstack(selected_X)
selected_y = np.array(selected_y)

# ========== APPLY AUGMENTATION ==========
augmenter = get_ecg_augmenter()
augmented_X = augmenter.augment(selected_X)

num_samples = selected_X.shape[1]
time = np.arange(num_samples) / sample_rate  # Time in seconds

fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(12, 6), sharex=True, sharey=True)
fig.suptitle("Original vs. Augmented ECG Samples (3 per class)", fontsize=16)

for i, ax in enumerate(axs.flat):
    ax.plot(time, selected_X[i], label="Original", color="blue")
    ax.plot(time, augmented_X[i], label="Augmented", color="orange", alpha=0.7)
    ax.set_title(f"Class {selected_y[i]}")

    # Show x-axis ticks and label only for bottom row
    if i // 5 == 2:
        ax.set_xlabel("Time (sec)")
        ax.set_xticks(np.linspace(0, num_samples / sample_rate, 4))
        ax.set_xticklabels([f"{x:.2f}" for x in np.linspace(0, num_samples / sample_rate, 4)])
    else:
        ax.set_xticks([])

    # Show y-axis only for leftmost column
    if i % 5 == 0:
        ax.set_ylabel("Amplitude")
    else:
        ax.set_yticks([])

    ax.grid(True)
# Add legend to the top-left plot only
axs[0, 0].legend(loc='upper right', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()
plt.savefig(os.path.join(PLOT_DIR, "augmented_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
