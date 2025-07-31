# MIT-BIH Arrhythmia Classification 🔬🫀

This project implements deep learning models to classify heartbeats in the MIT-BIH Arrhythmia dataset. It includes full preprocessing, data augmentation, and evaluation using cross-validation. It also supports hyperparameter tuning using Optuna.

---

## 📁 Project Structure

├── main.py                # Main training and evaluation pipeline
├── model_definitions.py  # CNN and Transformer model architectures
├── train_utils.py        # Training utilities and loss functions
├── optuna_utils.py        # Fine-tuning utilities
├── preprocessing.py      # Data preprocessing and augmentation
├── predict.py            # Script for predicting on new ECG data
├── config.py             # Hyperparameters and paths
├── utils.py                # Shared functions
├── README.md             # This file
└── /data                 # Processed and raw data

---

## 🚀 Models

Implemented:
- ✅ AcharyaCNN (baseline model from literature)
- ✅ ECGCNN (custom enhanced CNN)
- ✅ iTransformer (attention-based architecture)

Each model is evaluated using:
- Accuracy
- Confusion matrix
- Per-class precision & recall
- Cross-validation (5-fold)

---

## 🏋️ Training Details

- **Dataset:** MIT-BIH Arrhythmia Dataset (260-length segments)
- **Input shape:** `(batch_size, 1, 1, 260)`
- **Loss Function:** CrossEntropyLoss with optional class weighting
- **Optimizer:** Adam
- **Data Augmentation:** Random rescaling + jitter (optional, configurable)
- **Hyperparameter Tuning:** Optuna with dynamic model selection and parameter sweeps

---

## 📊 Results Summary

| Model         | Augmentation | Accuracy (mean ± std) |
|---------------|--------------|------------------------|
| AcharyaCNN    | No           |  XX.XX% ± X.XX         |
| ECGCNN        | ✅ Yes       |  **YY.YY% ± Y.YY**      |
| iTransformer  | ✅ Yes       |  ZZ.ZZ% ± Z.ZZ         |

> 💡 *ECGCNN with augmentation currently shows the best overall performance.*

---

## ⚙️ Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a model
```bash
python main.py 
```

### 3. Run Optuna tuning
```bash
TBD
```

### 4. Predict new data
```bash
TBD
```

---

## 🧪 Hyperparameter Tuning (Optuna)

Supported search space:
- Learning rate
- Dropout rate
- Weight decay
- Class weight alpha (softening)

---

## 📎 References
- MIT-BIH Dataset
- Acharya et al., “Deep CNN for ECG Classification”, 2017.
- Optuna for efficient hyperparameter optimization 

---
##  Author

Hsiu-Hsien (Leo) Lin
Email: [hhlin.work@gmail.com]
GitHub: [https://github.com/hsiuhsil]
LinkedIn: [https://www.linkedin.com/in/hsiuhsil/]
