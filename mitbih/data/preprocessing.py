"""
preprocess.py

This module provides utilities for:
- Extracting ECG beats from MIT-BIH arrhythmia records.
- Preprocessing signals using Savitzky–Golay filtering and Z-score normalization.
- Splitting the dataset into train, validation, and test sets.
- Saving processed datasets to disk in `.npz` format.
"""
import os
from collections import defaultdict
import numpy as np
import scipy
import wfdb
from sklearn.model_selection import train_test_split
from mitbih.utils.config import *

def extract_beats_from_record(rec_path, aami_map, label_map, window_size=130):
    """
    Extract R-peak-centered ECG segments and labels from a single MIT-BIH record.

    Args:
        rec_path (str): Path to the WFDB record (excluding extension).
        aami_map (dict): Mapping from annotation symbols to AAMI class labels.
        label_map (dict): Mapping from AAMI class labels to integer indices.
        window_size (int): Half-length of the segment (total = 2 * window_size).

    Returns:
        Tuple[List[np.ndarray], List[int]]: Extracted ECG beat segments and label indices.
    """
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
    """
    Process all specified ECG records to extract beats and class distributions.

    Args:
        records (List[str]): List of record IDs to process (e.g., ["100", "101"]).
        data_dir (str): Directory containing the WFDB record files.
        aami_map (dict): Mapping from annotation symbols to AAMI labels.
        label_map (dict): Mapping from AAMI labels to numeric indices.
        window_size (int): Half-length of extracted beat segment.
        class_names (List[str]): Ordered list of class names.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, int]]: All beat data, labels, and class count.
    """
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

def preprocess_beats(X_raw):
    """
    Apply Savitzky–Golay smoothing and z-score normalization to ECG beats.

    Args:
        X_raw (np.ndarray): Raw ECG segments, shape (N, L).

    Returns:
        np.ndarray: Preprocessed beats, same shape as input.
    """
    def preprocess_beat(beat):
        smoothed = scipy.signal.savgol_filter(beat, window_length=11, polyorder=3)
        return (smoothed - np.mean(smoothed)) / np.std(smoothed)

    return np.array([preprocess_beat(b) for b in X_raw])


def split_data(X, y, train_size=0.7, val_size=0.2, test_size=0.1, seed=42):
    """
    Split data into train/val/test sets.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Labels.
        train_size (float): Proportion of training data.
        val_size (float): Proportion of validation data.
        test_size (float): Proportion of test data.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1."

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(test_size / (val_size + test_size)),
        random_state=seed,
        stratify=y_temp
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_npz_datasets(save_dir, splits):
    """
    Save the train, validation, and test datasets as compressed .npz files.

    Args:
        save_dir (str): Directory to save the files.
        splits (List[Tuple[np.ndarray, np.ndarray]]): List of (X, y) tuples for train/val/test.
    """
    split_names = ['train', 'val', 'test']
    os.makedirs(save_dir, exist_ok=True)
    for name, (X, y) in zip(split_names, splits):
        np.savez(os.path.join(save_dir, f"ecg_{name}.npz"), X=X, y=y)
        print(f"Saved {name}: {X.shape}")

def run_pipeline(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, aami_map=AAMI_MAP, label_map=LABEL_MAP, window_size=WINDOW_SIZE, class_names=CLASS_NAMES):
    """
    Full preprocessing pipeline to extract, normalize, split, and save ECG beat data.

    Args:
        data_dir (str): Path to raw WFDB data.
        output_dir (str): Directory to save processed data.
        aami_map (dict): Symbol-to-label mapping.
        label_map (dict): Label-to-index mapping.
        window_size (int): Half-beat segment length.
        class_names (List[str]): Class names for reporting.
    """
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
    X_preprocessed = preprocess_beats(X_raw)
    train, val, test = split_data(X_preprocessed, y)
    save_npz_datasets(output_dir, [train, val, test])

def extract_features_labels_numpy(data_dir):
    """
    Load pre-saved features and labels from .npy files.

    Args:
        data_dir (str): Directory containing features.npy and labels.npy.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels arrays.
    """
    features_path = os.path.join(data_dir, "features.npy")
    labels_path = os.path.join(data_dir, "labels.npy")
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels
