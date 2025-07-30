from config import *
from preprocessing import run_pipeline
from dataloader import get_dataloaders
from model_definitions import AcharyaCNN, ECGCNN, iTransformer
from train_utils import train_model, evaluate_model
from metrics import plot_training_curves, plot_confusion_matrix, save_classification_report, plot_roc_pr_curves
from optuna_utils import get_or_run_study, save_best_trial_model
from utils import set_seed

import os
import torch
import time

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    run_pipeline(DATA_DIR, OUTPUT_DIR, window_size=WINDOW_SIZE)
    train_loader, val_loader, test_loader, trainval_loader = get_dataloaders(OUTPUT_DIR, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = [("AcharyaCNN", AcharyaCNN), ("ECGCNN", ECGCNN), ("iTransformer", iTransformer)]
    results = {}

    for name, model_cls in models:
        print(f"\nTraining {name}...")
        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()
        model, train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, device)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{name} training time: {elapsed:.2f} seconds")

        acc, preds, labels, probs = evaluate_model(model, test_loader, device, class_names=CLASS_NAMES)
        
        # Plot training curves
        curve_path = os.path.join(PLOT_DIR, f"{name}_training_curves.png")
        plot_training_curves(train_acc, val_acc, train_loss, val_loss, save_path=curve_path)

        # Save Confusion Matrix
        cm_path = os.path.join(PLOT_DIR, f"{name}_confusion_matrix.png")
        plot_confusion_matrix(labels, preds, CLASS_NAMES, title=f"{name} Confusion Matrix", save_path=cm_path)

        # Save Classification Report
        report_path = os.path.join(PLOT_DIR, f"{name}_classification_report.txt")
        save_classification_report(labels, preds, CLASS_NAMES, path=report_path)

        # Save ROC and PR curves
        plot_roc_pr_curves(labels, probs, CLASS_NAMES, save_dir=os.path.join(PLOT_DIR, f"{name}_curves"))

        results[name] = acc

    print("\nRunning Optuna tuning...")
    study = get_or_run_study(STUDY_PATH, train_loader, val_loader, n_trials=30)
    best_model, train_acc, train_loss = save_best_trial_model(study, trainval_loader, save_path=MODEL_SAVE_PATH, device=device)

    # Plot training curves
    plot_training_curves(train_acc, val_acc=None, train_loss=train_loss, val_loss=None,
                         save_path=os.path.join(PLOT_DIR, "Optuna_ECGCNN_training_curves.png"))

    print("\nFinal Evaluation on Test Set (Optuna-Tuned ECGCNN):")
    acc, preds, labels, probs = evaluate_model(best_model, test_loader, device, CLASS_NAMES)

    # Save Confusion Matrix
    cm_path = os.path.join(PLOT_DIR, "Optuna_ECGCNN_confusion_matrix.png")
    plot_confusion_matrix(labels, preds, CLASS_NAMES, title="Optuna-Tuned ECGCNN", save_path=cm_path)

    # Save Classification Report
    report_path = os.path.join(PLOT_DIR, "Optuna_ECGCNN_classification_report.txt")
    save_classification_report(labels, preds, CLASS_NAMES, path=report_path)

    # Save ROC and PR curves
    curves_dir = os.path.join(PLOT_DIR, "Optuna_ECGCNN_curves")
    plot_roc_pr_curves(labels, probs, CLASS_NAMES, save_dir=curves_dir)

    results["Optuna_ECGCNN"] = acc

    print("\n=== Final Accuracy Summary ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
