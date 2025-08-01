from config import *
from preprocessing import run_pipeline
from dataloader import get_cross_validation_loaders, get_dataloaders, get_full_trainval_loader
from model_definitions import AcharyaCNN, ECGCNN, iTransformer
from train_utils import train_model, evaluate_model, get_class_weights
from metrics import plot_training_curves, plot_confusion_matrix, save_classification_report, plot_roc_pr_curves
from optuna_utils import get_or_run_study, save_best_trial_model, load_study
from utils import set_seed, export_results_json

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import json

set_seed()

def run_single_fold(model_cls, fold_id, train_loader, val_loader, device, plot_subdir, class_names):
    # === Prepare model, optimizer, criterion ===
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())

    class_weights = get_class_weights(all_labels, NUM_CLASSES).to(device)
    alpha = 0.5
    soft_weights = (1 - alpha) * torch.ones_like(class_weights) + alpha * class_weights
    criterion = nn.CrossEntropyLoss(weight=soft_weights) if alpha > 0 else nn.CrossEntropyLoss()

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === Train ===
    start_time = time.time()
    model, train_acc, val_acc, train_loss, val_loss = train_model(
        model, train_loader, val_loader, optimizer, criterion, EPOCHS, device
    )
    elapsed = time.time() - start_time
    print(f"Fold {fold_id} training time: {elapsed:.2f} seconds")

    # === Evaluate ===
    acc, preds, labels, probs = evaluate_model(model, val_loader, device, class_names=class_names)

    # === Save plots ===
    base_path = os.path.join(plot_subdir, f"{model_cls.__name__}_fold{fold_id}")
    plot_training_curves(train_acc, val_acc, train_loss, val_loss,
                         save_path=f"{base_path}_training_curves.png")
    plot_confusion_matrix(labels, preds, class_names,
                          title=f"{model_cls.__name__} Fold {fold_id} Confusion Matrix",
                          save_path=f"{base_path}_confusion_matrix.png")
    save_classification_report(labels, preds, class_names,
                               path=f"{base_path}_classification_report.txt")
    plot_roc_pr_curves(labels, probs, class_names, save_dir=f"{base_path}_curves")

    return acc

def train_and_evaluate(augment: bool, plot_subdir: str, use_kfold: bool = False, num_folds: int = 5):
    os.makedirs(plot_subdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [("AcharyaCNN", AcharyaCNN), ("ECGCNN", ECGCNN), ("iTransformer", iTransformer)]
    #models = [("ECGCNN", ECGCNN), ("iTransformer", iTransformer)]
    results = {}

    if use_kfold:
        print(f"\nRunning {num_folds}-fold cross-validation...")
        kfold_loaders = list(get_cross_validation_loaders(OUTPUT_DIR, k=num_folds, batch_size=BATCH_SIZE, augment=augment))

        for name, model_cls in models:
            fold_accuracies = []
            for fold_id, train_loader, val_loader in kfold_loaders:
                print(f"\n{name} - Fold {fold_id}/{num_folds}")
                acc = run_single_fold(model_cls, fold_id, train_loader, val_loader, device, plot_subdir, CLASS_NAMES)
                fold_accuracies.append(acc)

            avg_acc = sum(fold_accuracies) / num_folds
            results[name] = avg_acc
            print(f"{name} average accuracy: {avg_acc:.4f}")

    else:
        print("\nRunning single train/val split...")
        train_loader, val_loader, test_loader, _ = get_dataloaders(
            OUTPUT_DIR, batch_size=BATCH_SIZE, augment=augment
        )

        for name, model_cls in models:
            print(f"\nTraining {name} {'with' if augment else 'without'} augmentation...")
            acc = run_single_fold(model_cls, fold_id=1, train_loader=train_loader,
                                  val_loader=test_loader,  # Evaluate on test set here
                                  device=device, plot_subdir=plot_subdir,
                                  class_names=CLASS_NAMES)
            results[name] = acc

    return results

def old_train_and_evaluate(augment: bool, plot_subdir: str):
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
    no_aug_results = train_and_evaluate(augment=False, plot_subdir=os.path.join(PLOT_DIR, "no_aug"),
                                        use_kfold=True, num_folds=5)

    print("\n=== Training with augmentation ===")
    aug_results = train_and_evaluate(augment=True, plot_subdir=os.path.join(PLOT_DIR, "with_aug"),
                                     use_kfold=True, num_folds=5)

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

def old_run_optuna(augment=False):
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
    study = get_or_run_study(STUDY_PATH, train_loader, val_loader, n_trials=30)

    # Save best model
    best_model, train_acc, train_loss = save_best_trial_model(
        study, trainval_loader, num_epoches=30, save_path=MODEL_SAVE_PATH, device=DEVICE
    )

    # Plot training curves
    plot_training_curves(
        train_acc, val_acc=None, train_loss=train_loss, val_loss=None,
        save_path=os.path.join(plot_subdir, "training_curves.png")
    )

    # Evaluate on test set
    print(f"\nFinal Evaluation on Test Set ({aug_tag}):")
    acc, preds, labels, probs = evaluate_model(
        best_model, test_loader, DEVICE, CLASS_NAMES
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

def optuna_fold_pipeline(
    train_loader, val_loader, save_prefix, fold_tag,
    test_loader, plot_dir, augment, retrain_loader=None
):
    print(f"\n>> Running Optuna Study {fold_tag}")
    study_path = f"{save_prefix}_{fold_tag}.pkl"
    model_path = f"{save_prefix}_{fold_tag}.pth"
    plot_subdir = os.path.join(plot_dir, fold_tag)
    os.makedirs(plot_subdir, exist_ok=True)

    study = get_or_run_study(study_path, train_loader, val_loader, n_trials=20)

    # Use train+val for final training (if provided), otherwise just train
    final_loader = retrain_loader if retrain_loader else train_loader

    best_model, train_acc, train_loss = save_best_trial_model(
        study, final_loader, num_epoches=30, save_path=model_path, device=DEVICE
    )

    plot_training_curves(
        train_acc, val_acc=None, train_loss=train_loss, val_loss=None,
        save_path=os.path.join(plot_subdir, "training_curves.png")
    )

    acc, preds, labels, probs = evaluate_model(best_model, test_loader, DEVICE, CLASS_NAMES)

    plot_confusion_matrix(labels, preds, CLASS_NAMES,
                          title=f"ECGCNN ({fold_tag})",
                          save_path=os.path.join(plot_subdir, "confusion_matrix.png"))
    save_classification_report(labels, preds, CLASS_NAMES,
                               path=os.path.join(plot_subdir, "classification_report.txt"))
    plot_roc_pr_curves(labels, probs, CLASS_NAMES,
                       save_dir=os.path.join(plot_subdir, "curves"))

    return acc

def get_best_hyperparams_from_folds(save_prefix, num_folds):
    best_trials = []
    
    for fold in range(num_folds):
        fold_tag = f"fold{fold + 1}"
        study_path = f"{save_prefix}_{fold_tag}.pkl"
        if os.path.exists(study_path):
            # study = optuna.load_study(study_name=None, storage=None, study_path=study_path)
            study = load_study(study_path)
            best_trials.append(study.best_trial)
    
    # Choose the trial with the lowest value (objective)
    best_trial = min(best_trials, key=lambda t: t.value)
    return best_trial.params

def run_optuna(augment=False, use_kfold=True, num_folds=5):
    aug_tag = "with_aug" if augment else "no_aug"
    print(f"\n[Optuna] Augment: {augment} | K-Fold: {use_kfold}")

    save_prefix = os.path.join(OUTPUT_DIR, f"ecgcnn_optuna_{aug_tag}")
    plot_dir = os.path.join(PLOT_DIR, f"Optuna_ECGCNN_{aug_tag}")
    os.makedirs(plot_dir, exist_ok=True)

    _, _, test_loader, _ = get_dataloaders(OUTPUT_DIR, batch_size=BATCH_SIZE, augment=False)

    if use_kfold:
        fold_accuracies = []
        for fold, train_loader, val_loader in get_cross_validation_loaders(
            OUTPUT_DIR, batch_size=BATCH_SIZE, k=num_folds, augment=augment
        ):
            # Load fold train/val data to make a combined loader for retraining
            train_npz = np.load(os.path.join(OUTPUT_DIR, "ecg_fold_train.npz"))
            val_npz = np.load(os.path.join(OUTPUT_DIR, "ecg_fold_val.npz"))
            X_combined = np.concatenate([train_npz['X'], val_npz['X']], axis=0)
            y_combined = np.concatenate([train_npz['y'], val_npz['y']], axis=0)
            combined_ds = TensorDataset(torch.tensor(X_combined[:, None, :], dtype=torch.float32),
                                        torch.tensor(y_combined, dtype=torch.long))
            retrain_loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True)

            acc = optuna_fold_pipeline(
                train_loader=train_loader,
                val_loader=val_loader,
                save_prefix=save_prefix,
                fold_tag=f"fold{fold+1}",
                test_loader=test_loader,
                plot_dir=plot_dir,
                augment=augment,
                retrain_loader=retrain_loader
            )
            fold_accuracies.append(acc)

                # After k-fold, retrain with best hyperparameters
        best_params = get_best_hyperparams_from_folds(save_prefix, num_folds)
        
        print("\n>> Retraining final model using best hyperparameters across folds...")

        # Load full training set (train + val)
        trainval_loader = get_full_trainval_loader(OUTPUT_DIR, batch_size=BATCH_SIZE, augment=augment)
        
        final_model_path = f"{save_prefix}_final_retrained.pth"
        final_plot_dir = os.path.join(plot_dir, "final_model")
        os.makedirs(final_plot_dir, exist_ok=True)

        best_model, train_acc, train_loss = save_best_trial_model(
            best_params, trainval_loader, num_epoches=30, save_path=final_model_path, device=DEVICE
        )

        plot_training_curves(
            train_acc, val_acc=None, train_loss=train_loss, val_loss=None,
            save_path=os.path.join(final_plot_dir, "training_curves.png")
        )

        acc, preds, labels, probs = evaluate_model(best_model, test_loader, DEVICE, CLASS_NAMES)

        plot_confusion_matrix(labels, preds, CLASS_NAMES,
                              title=f"ECGCNN Final Model",
                              save_path=os.path.join(final_plot_dir, "confusion_matrix.png"))
        save_classification_report(labels, preds, CLASS_NAMES,
                                   path=os.path.join(final_plot_dir, "classification_report.txt"))
        plot_roc_pr_curves(labels, probs, CLASS_NAMES,
                           save_dir=os.path.join(final_plot_dir, "curves"))

        print(f"[Final Retrained] Accuracy on Test Set: {acc:.4f}")
        results[f"Optuna_ECGCNN_{aug_tag}_final"] = acc

    else:
        train_loader, val_loader, test_loader, trainval_loader = get_dataloaders(
            OUTPUT_DIR, batch_size=BATCH_SIZE, augment=augment
        )

        acc = optuna_fold_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            save_prefix=save_prefix,
            fold_tag="single_split",
            test_loader=test_loader,
            plot_dir=plot_dir,
            augment=augment,
            retrain_loader=trainval_loader
        )
        results[f"Optuna_ECGCNN_{aug_tag}"] = acc

if __name__ == "__main__":
    
    if USE_OPTUNA:
        results={}
        run_optuna(augment=True, use_kfold=True, num_folds=5)
        
        # Print and export final results
        print("\n=== Combined Final Accuracy Summary ===")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        # Save results to JSON log
        export_results_json(results, save_path=os.path.join(OUTPUT_DIR, "optuna_augmentation_comparison_results.json"))


    else:
        main()
