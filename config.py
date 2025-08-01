"""
config.py

This module defines global configuration settings, constants, and directory paths
used throughout the ECG arrhythmia classification project.

Contents:
- General settings such as random seed, batch size, training epochs, and device.
- AAMI-to-class label mappings and class definitions.
- File and directory paths for input data, outputs, plots, model checkpoints, etc.

Key Variables:
- SEED: Reproducibility seed for training and data splits.
- BATCH_SIZE: Batch size for DataLoader.
- WINDOW_SIZE: Segment length (in timesteps) for ECG beats.
- CLASS_NAMES: ['N', 'S', 'V', 'F', 'Q'] corresponding to AAMI classes.
- LABEL_MAP: Dictionary mapping class labels to integer indices.
- USE_OPTUNA: Flag to control whether Optuna hyperparameter tuning is used.
- OUTPUT_DIR / PLOT_DIR / MODEL_SAVE_PATH / STUDY_PATH: Centralized file structure.

Used by:
- All major components, including data loading, training, and evaluation scripts.
"""

import os

# General config
SEED = 42
BATCH_SIZE = 64
WINDOW_SIZE = 130
EPOCHS = 20
DEVICE = 'cpu'
USE_OPTUNA = True

# Class mappings
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
NUM_CLASSES = len(CLASS_NAMES)
LABEL_MAP = {label: idx for idx, label in enumerate(CLASS_NAMES)}

# Path
DATA_DIR = "./data/physionet.org/files/mitdb/1.0.0"
OUTPUT_DIR = "./temp"
PLOT_DIR = "./figures"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "ecgcnn_optuna_with_aug.pth")
STUDY_PATH = os.path.join(OUTPUT_DIR,"optuna_study_with_aug.pkl")
DEMO_PATH = os.path.join(OUTPUT_DIR,"demo_beats.npz")
