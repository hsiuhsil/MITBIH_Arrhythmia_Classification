import os
from collections import defaultdict
import numpy as np
import scipy
import wfdb
from sklearn.model_selection import train_test_split
from config import *

"""define functions for preprocessing"""

def extract_beats_from_record(rec_path, aami_map, label_map, window_size=130):
    """extract beats from the raw data """
    try:
        record = wfdb.rdrecord(rec_path)
        annotation = wfdb.rdann(rec_path, 'atr')
        signal = record.p_signal[:, 0]  # Use Lead II

        r_peaks = annotation.sample
        symbols = annotation.symbol

        X, y = [], []
        for i, peak in enumerate(r_peaks):
            label = symbols[i]
            if label not in aami_map:
                continue

            cls = aami_map[label]
            label_idx = label_map[cls]

            start = peak - window_size
            end = peak + window_size

            if start < 0 or end >= len(signal):
                continue

            segment = signal[start:end]
            X.append(segment)
            y.append(label_idx)

        return X, y

    except Exception as e:
        print(f"Error processing {rec_path}: {e}")
        return [], []

def process_all_records(records, data_dir=DATA_DIR, aami_map=AAMI_MAP, label_map=LABEL_MAP, window_size=WINDOW_SIZE, class_names=CLASS_NAMES):
    """get all beats from the raw data"""
    all_X, all_y = [], []
    class_counts = defaultdict(int)

    for rec in records:
        rec_path = os.path.join(data_dir, rec)
        X, y = extract_beats_from_record(rec_path, aami_map, label_map, window_size)
        all_X.extend(X)
        all_y.extend(y)

        for label in y:
            class_name = class_names[label]
            class_counts[class_name] += 1

    return np.array(all_X), np.array(all_y), class_counts

# Preprocessing and Splitting

def preprocess_and_split(X_raw, y, train_size=0.7, val_size=0.2, test_size=0.1, seed=42):
    """Using the Z-score to normalize the beats, then splitting into train, val, and test dataseets"""
    def preprocess_beat(beat):
        smoothed = scipy.signal.savgol_filter(beat, window_length=11, polyorder=3)
        return (smoothed - np.mean(smoothed)) / np.std(smoothed)

    X_processed = np.array([preprocess_beat(b) for b in X_raw])

    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=(val_size+test_size), random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size/(val_size+test_size)), random_state=seed, stratify=y_temp)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_npz_datasets(save_dir, splits):
    """save the datasets into npz files"""
    split_names = ['train', 'val', 'test']
    os.makedirs(save_dir, exist_ok=True)
    for name, (X, y) in zip(split_names, splits):
        np.savez(os.path.join(save_dir, f"ecg_{name}.npz"), X=X, y=y)
        print(f"Saved {name}: {X.shape}")

def run_pipeline(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, aami_map=AAMI_MAP, label_map=LABEL_MAP, window_size=WINDOW_SIZE, class_names=CLASS_NAMES):
    """define the preprocess pipeline"""
    # Get all records
    records = [f"{i:03d}" for i in range(100, 235) if os.path.exists(os.path.join(data_dir, f"{i:03d}.dat"))]

    X_raw, y, class_counts = process_all_records(records, data_dir,  aami_map, label_map, window_size, class_names)

    print(f"Total beats: {len(y)}")
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # Save raw
    np.savez(os.path.join(output_dir, "ecg_beat_dataset.npz"), X=X_raw, y=y, label_map=label_map)

    # Preprocess & split
    train, val, test = preprocess_and_split(X_raw, y)
    save_npz_datasets(output_dir, [train, val, test])

def extract_features_labels_numpy(data_dir):
    features_path = os.path.join(data_dir, "features.npy")
    labels_path = os.path.join(data_dir, "labels.npy")
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels
