""" define the utils """
import os
import random
import numpy as np
import torch
import json

def set_seed(seed=42):
    """define the random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def export_results_json(results_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to JSON: {save_path}")
