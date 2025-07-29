from config import *
from preprocessing import run_pipeline
from dataloader import get_dataloaders
from model_definitions import AcharyaCNN, ECGCNN, iTransformer
from train_utils import train_model, evaluate_model
from metrics import plot_confusion_matrix
from optuna_utils import get_or_run_study, save_best_trial_model
from utils import set_seed

import torch
import os

def main():
#    os.makedirs(OUTPUT_DIR, exist_ok=True)
#    set_seed(SEED)

#    run_pipeline(DATA_DIR, OUTPUT_DIR, window_size=WINDOW_SIZE)
    train_loader, val_loader, test_loader, trainval_loader = get_dataloaders(OUTPUT_DIR, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = [("AcharyaCNN", AcharyaCNN), ("ECGCNN", ECGCNN), ("iTransformer", iTransformer)]
    results = {}

#    for name, model_cls in models:
#        print(f"\nTraining {name}...")
#        model = model_cls().to(device)
#        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#        criterion = torch.nn.CrossEntropyLoss()
#        model = train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, device)
#        acc, preds, labels = evaluate_model(model, test_loader, device, class_names=CLASS_NAMES)
#        plot_confusion_matrix(labels, preds, CLASS_NAMES, title=f"{name} Confusion Matrix")
#        results[name] = acc

    print("\nRunning Optuna tuning...")
    study = get_or_run_study(STUDY_PATH, train_loader, val_loader, n_trials=30)
    best_model = save_best_trial_model(study, trainval_loader, save_path=MODEL_SAVE_PATH, device=device)

    print("\nFinal Evaluation on Test Set:")
    acc, preds, labels = evaluate_model(best_model, test_loader, device, CLASS_NAMES)
    plot_confusion_matrix(labels, preds, CLASS_NAMES, title="Optuna-Tuned ECGCNN")
    results["Optuna_ECGCNN"] = acc

    print("\n=== Final Accuracy Summary ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
