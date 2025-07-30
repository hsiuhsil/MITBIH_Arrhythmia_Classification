import os
import numpy as np
import matplotlib.pyplot as plt
from tsaug import TimeWarp, Drift, AddNoise
from config import OUTPUT_DIR, PLOT_DIR, CLASS_NAMES

samples_per_class = 3
num_classes = len(CLASS_NAMES)
total_samples = samples_per_class * num_classes

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
augmenter = (
    TimeWarp(n_speed_change=3, max_speed_ratio=2) * 0.5
    + Drift(max_drift=(0.05, 0.1), n_drift_points=5) * 0.5
    + AddNoise(scale=0.05)
)

augmented_X = augmenter.augment(selected_X)

# ========== PLOT ==========
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 6), sharex=True, sharey=True)
fig.suptitle("Original vs. Augmented ECG Samples (3 per class)", fontsize=16)

for i, ax in enumerate(axs.flat):
    ax.plot(selected_X[i], label="Original", color="blue")
    ax.plot(augmented_X[i], label="Augmented", color="orange", alpha=0.7)
    ax.set_title(f"Class {selected_y[i]}")
    ax.set_xticks([])
    ax.set_yticks([])

axs[0, 0].legend(loc='upper right', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR,"augmented_comparison.png"))
print(f"Saved visualization to {SAVE_PATH}")
plt.close()
