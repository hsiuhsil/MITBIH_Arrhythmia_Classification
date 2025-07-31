from config import *
from preprocessing import run_pipeline
from dataloader import get_dataloaders
from model_definitions import AcharyaCNN, ECGCNN, iTransformer
from train_utils import train_model, evaluate_model, get_class_weights
from metrics import plot_training_curves, plot_confusion_matrix, save_classification_report, plot_roc_pr_curves
from optuna_utils import get_or_run_study, save_best_trial_model
from utils import set_seed, export_results_json

import os
import torch
import torch.nn as nn
import time
import json

set_seed()

def train_and_evaluate(augment: bool, plot_subdir: str):
    os.makedirs(plot_subdir, exist_ok=True)
    train_loader, val_loader, test_loader, trainval_loader = get_dataloaders(
        OUTPUT_DIR, batch_size=BATCH_SIZE, augment=augment
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [("AcharyaCNN", AcharyaCNN), ("ECGCNN", ECGCNN), ("iTransformer", iTransformer)]
    results = {}

    # Extract labels from train_loader
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    
    class_weights = get_class_weights(all_labels, NUM_CLASSES).to(device)
    # Soften the weights:
    alpha = 0.5  # tune this between 0 (no weighting) and 1 (full weights)
    soft_weights = (1 - alpha) * torch.ones_like(class_weights) + alpha * class_weights
    print(f"Softened class weights: {soft_weights}")
    if alpha == 0:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=soft_weights.to(device))

    for name, model_cls in models:
        print(f"\nTraining {name} {'with' if augment else 'without'} augmentation...")
        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        start_time = time.time()
        model, train_acc, val_acc, train_loss, val_loss = train_model(
            model, train_loader, val_loader, optimizer, criterion, EPOCHS, device
        )
        elapsed = time.time() - start_time
        print(f"{name} training time: {elapsed:.2f} seconds")

        acc, preds, labels, probs = evaluate_model(model, test_loader, device, class_names=CLASS_NAMES)

        # Save plots and metrics
        plot_training_curves(train_acc, val_acc, train_loss, val_loss,
                             save_path=os.path.join(plot_subdir, f"{name}_training_curves.png"))
        plot_confusion_matrix(labels, preds, CLASS_NAMES,
                              title=f"{name} Confusion Matrix",
                              save_path=os.path.join(plot_subdir, f"{name}_confusion_matrix.png"))
        save_classification_report(labels, preds, CLASS_NAMES,
                                   path=os.path.join(plot_subdir, f"{name}_classification_report.txt"))
        plot_roc_pr_curves(labels, probs, CLASS_NAMES,
                           save_dir=os.path.join(plot_subdir, f"{name}_curves"))

        results[name] = acc

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    set_seed(SEED)

    run_pipeline(DATA_DIR, OUTPUT_DIR, window_size=WINDOW_SIZE)

    print("\n=== Training without augmentation ===")
    no_aug_results = train_and_evaluate(augment=False, plot_subdir=os.path.join(PLOT_DIR, "no_aug"))

    print("\n=== Training with augmentation ===")
    aug_results = train_and_evaluate(augment=True, plot_subdir=os.path.join(PLOT_DIR, "with_aug"))

    # Compare before and after augmentation
    print("\n=== Accuracy Comparison ===")
    combined_results = {}
    for name in no_aug_results:
        print(f"{name} - No Aug: {no_aug_results[name]:.4f}, With Aug: {aug_results[name]:.4f}")
        combined_results[name] = {
            "no_augmentation": no_aug_results[name],
            "with_augmentation": aug_results[name]
        }

    # Save results to JSON log
    export_results_json(combined_results, save_path=os.path.join(OUTPUT_DIR, "augmentation_comparison_results.json"))

def run_optuna(augment=False):
    aug_tag = "with_aug" if augment else "no_aug"
    print(f"\nRunning Optuna tuning with augmentation: {augment}")

    # Create plot/output directory
    plot_subdir = os.path.join(PLOT_DIR, f"Optuna_ECGCNN_{aug_tag}")
    os.makedirs(plot_subdir, exist_ok=True)

    # Get data loaders
    train_loader, val_loader, test_loader, trainval_loader = get_dataloaders(
        OUTPUT_DIR, batch_size=BATCH_SIZE, augment=augment
    )

    # Run Optuna study
    study = get_or_run_study(STUDY_PATH, train_loader, val_loader, n_trials=1)

    # Save best model
    model_save_path = os.path.join(MODEL_SAVE_PATH, f"best_model_{aug_tag}.pt")
    best_model, train_acc, train_loss = save_best_trial_model(
        study, trainval_loader, save_path=model_save_path, device=DEVICE
    )

    # Plot training curves
    plot_training_curves(
        train_acc, val_acc=None, train_loss=train_loss, val_loss=None,
        save_path=os.path.join(plot_subdir, "training_curves.png")
    )

    # Evaluate on test set
    print(f"\nFinal Evaluation on Test Set ({aug_tag}):")
    acc, preds, labels, probs = evaluate_model(
        best_model, test_loader, device, CLASS_NAMES
    )

    # Save confusion matrix and classification report
    plot_confusion_matrix(
        labels, preds, CLASS_NAMES,
        title=f"ECGCNN ({aug_tag})",
        save_path=os.path.join(plot_subdir, "confusion_matrix.png")
    )
    save_classification_report(
        labels, preds, CLASS_NAMES,
        path=os.path.join(plot_subdir, "classification_report.txt")
    )

    # Save ROC and PR curves
    plot_roc_pr_curves(
        labels, probs, CLASS_NAMES,
        save_dir=os.path.join(plot_subdir, "curves")
    )

    # Store result
    results[f"Optuna_ECGCNN_{aug_tag}"] = acc

if __name__ == "__main__":
    
    if USE_OPTUNA:
        results={}
        run_optuna(augment=True)
        
        # Print and export final results
        print("\n=== Combined Final Accuracy Summary ===")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        # Save results to JSON log
        export_results_json(results, save_path=os.path.join(OUTPUT_DIR, "optuna_augmentation_comparison_results.json"))

    else:
        main()
