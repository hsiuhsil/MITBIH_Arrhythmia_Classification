import torch

# Paths
DATA_PATH = "./data/"
RESULTS_PATH = "./results/"
MODEL_PATH = "./results/best_model.pt"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Parameters
NUM_CLASSES = 5
INPUT_SHAPE = (1, 187)

# Training Parameters
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
SEED = 42
