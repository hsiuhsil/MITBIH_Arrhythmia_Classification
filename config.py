"""define config """
import os

# General config
SEED = 42
BATCH_SIZE = 64
WINDOW_SIZE = 130
EPOCHS = 20
DEVICE = 'cpu'

# Class mappings
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
LABEL_MAP = {label: idx for idx, label in enumerate(CLASS_NAMES)}

DATA_DIR = "./data/physionet.org/files/mitdb/1.0.0"
OUTPUT_DIR = "./temp"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "ecgcnn_optuna.pth")
