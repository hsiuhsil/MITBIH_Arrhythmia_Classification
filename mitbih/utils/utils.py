"""
utils.py

General utility functions for experiment reproducibility and result exporting.

Includes:
- `set_seed`: Ensures deterministic behavior across numpy, PyTorch, and system hash seed.
- `export_results_json`: Saves experiment results as a formatted JSON file.
"""
import os
import random
import numpy as np
import torch
import json

def set_seed(seed=42):
    """
    Set seeds for reproducibility across NumPy, PyTorch, and Python environment.

    Args:
        seed (int): The seed value to use for all random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def export_results_json(results_dict, save_path):
    """
    Export experiment results to a JSON file.

    Args:
        results_dict (dict): Dictionary of results to save.
        save_path (str): Full file path where the JSON will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to JSON: {save_path}")
