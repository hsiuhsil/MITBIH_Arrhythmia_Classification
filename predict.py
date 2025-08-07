"""
predict.py

This script defines utilities to load a trained ECG classification model 
and make predictions on one or multiple ECG heartbeats.

Supports loading Optuna-optimized ECGCNN model and mapping predicted indices 
back to human-readable AAMI classes.
"""
import torch
import torch.nn.functional as F
import numpy as np
from model_definitions import AcharyaCNN, ECGCNN, iTransformer  
from optuna_utils import load_study
from config import DEVICE, MODEL_SAVE_PATH, STUDY_PATH, LABEL_MAP, CLASS_NAMES 

def predict_all_beats(beats, model, device):
    """
    Predict the AAMI class of each ECG beat in the input batch using a trained model.

    Args:
        beats (torch.Tensor): Tensor of shape [N, 1, 260] representing N ECG beats.
        model (torch.nn.Module): Trained ECG classification model.
        device (torch.device): Device to perform inference on ('cpu' or 'cuda').

    Returns:
        List[str]: List of predicted class labels (e.g., ['N', 'V', 'S']).
    """
    model.eval()
    beats = beats.to(device).float()
    with torch.no_grad():
        output = model(beats)
        probs = F.softmax(output, dim=1)
        predicted_classes = torch.argmax(probs, dim=1)
    predicted_indices = predicted_classes.tolist()
    predicted_labels = [CLASS_NAMES[i] for i in predicted_indices]
    return predicted_labels

def load_model_and_predict(beat_array, model_class, model_path=MODEL_SAVE_PATH, device=DEVICE):
    """
    Load the trained model (with best Optuna parameters) and predict labels for ECG beats.

    Args:
        beat_array (np.ndarray): Array of shape [N, 260] representing N ECG beats.
        model_class (type): Model class to instantiate (e.g., ECGCNN).
        model_path (str): Path to the saved model weights (.pth file).
        device (str or torch.device): Device to run the model on ('cpu' or 'cuda').

    Returns:
        List[str]: Predicted class labels for each beat.
    """
    # Convert the beat into a tensor with the expected input shape: [# of beats, 1, 260]
    beat_tensor = torch.from_numpy(beat_array).unsqueeze(1).float()

    # Load the optimized model    
    study = load_study(STUDY_PATH)
    best_params = study.best_trial.params

    best_model_params = {
        "kernel_size": best_params['kernel_size'],
        "dropout": best_params['dropout'],
        "filters1": best_params['filters1'],
        "filters2": best_params['filters2'],
        "fc1_size": best_params['fc1_size'],
        "use_third_conv": best_params['use_third_conv']
    }
    if best_model_params["use_third_conv"]:
        best_model_params["filters3"] = best_params['filters3']


    model = ECGCNN(**best_model_params).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    return predict_all_beats(beat_tensor, model, device)


if __name__ == "__main__":
    """
    Example usage: Generate random ECG beats and print predicted classes.
    Replace `np.random.randn` with real, preprocessed ECG beat data.
    """
    # Example beat (should be of shape (10 beats, 260 samples/beat))
    sample_beats = np.random.randn(10, 260)  # Replace with actual preprocessed beat

    predicted = load_model_and_predict(sample_beats, ECGCNN, model_path=MODEL_SAVE_PATH, device=DEVICE)
    print("Predicted class:", predicted)
