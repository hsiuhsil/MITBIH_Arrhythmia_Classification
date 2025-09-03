"""
api_demo.py

Demo script to:
1) Convert an NPZ file of ECG beats to CSV.
2) Send the CSV to a FastAPI endpoint (/predict_csv) for prediction.
"""

import numpy as np
import pandas as pd
import requests
from pathlib import Path

# -------------------------------
# Config
# -------------------------------
NPZ_PATH = Path("results/temp/demo_beats.npz")
CSV_PATH = Path("results/temp/demo_beats.csv")
API_URL = "http://18.188.196.254/predict_csv"  # Using /predict_json if the input is in the json format. 
#API_URL = "http://127.0.0.1:8000/predict_csv"

# -------------------------------
# Step 1: Convert NPZ to CSV
# -------------------------------
if not NPZ_PATH.exists():
    raise FileNotFoundError(f"{NPZ_PATH} not found. Make sure the NPZ file exists.")

# Load NPZ file
data = np.load(NPZ_PATH)
X = data['X']  # shape (num_samples, num_features)
y = data['y']  # shape (num_samples,)

# Convert to DataFrame
df = pd.DataFrame(X)
df['label'] = y 

# Save to CSV
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV_PATH, index=False, header=False)  # drop header
print(f"[INFO] Saved CSV to {CSV_PATH}")

# -------------------------------
# Step 2 & 3: Send CSV to API
# -------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"{CSV_PATH} not found. CSV conversion failed.")

with open(CSV_PATH, "rb") as f:
    files = {"file": (CSV_PATH.name, f, "text/csv")}
    try:
        response = requests.post(API_URL, files=files)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to call API: {e}")
    else:
        print("[INFO] Prediction result:", response.json())
