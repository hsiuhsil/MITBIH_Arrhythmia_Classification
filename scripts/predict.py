"""
predict.py

This script defines utilities to load a trained ECG classification model 
and make predictions on one or multiple ECG heartbeats.

Supports loading Optuna-optimized ECGCNN model and mapping predicted indices 
back to human-readable AAMI classes.
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from mitbih.models.model_definitions import AcharyaCNN, ECGCNN, iTransformer  
from mitbih.training.optuna_utils import load_study
from mitbih.utils.config import DEVICE, MODEL_SAVE_PATH, STUDY_PATH, LABEL_MAP, CLASS_NAMES 

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

    print(best_model_params)
    model = ECGCNN(**best_model_params).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    return predict_all_beats(beat_tensor, model, device)


if __name__ == "__main__":
    """
    This script allows you to predict the class labels of ECG beats using a trained ECGCNN model.

    Usage options:

    1. Generate random beats (for testing):
       python predict.py --num_beats 10
       - Generates 10 random ECG beats (each of length 260 samples)
       - Prints predicted labels

    2. Use a CSV file containing ECG beats:
       python predict.py --csv path/to/beats.csv
       - CSV should have N rows (beats) and 260 columns (samples per beat)
       - Prints predicted labels for all beats

    3. Use a JSON file containing ECG beats:
       python predict.py --json path/to/beats.json
       - JSON should be a list of lists, shape [N, 260]
       - Prints predicted labels for all beats

    Notes:
    - Ensure your ECG beats are preprocessed the same way as during training (filtering, normalization).
    - Output is a dictionary with keys:
        "num_beats": number of input beats
        "predictions": list of predicted AAMI class labels (e.g., ["N", "V", "S"])
    """

    # Example beat (should be of shape (10 beats, 260 samples/beat))
    #sample_beats = np.random.randn(10, 260)  # Replace with actual preprocessed beat

    parser = argparse.ArgumentParser(description="Predict ECG beat classes.")
    parser.add_argument("--csv", type=str, help="Path to CSV file containing ECG beats (N x 260).")
    parser.add_argument("--json", type=str, help="Path to JSON file containing ECG beats.")
    parser.add_argument("--num_beats", type=int, default=5, help="Generate random beats if no input provided.")
    args = parser.parse_args()

    if args.csv:
        beat_array = pd.read_csv(args.csv, header=None).to_numpy()
    elif args.json:
        import json
        with open(args.json) as f:
            beat_array = np.array(json.load(f))
    else:
        beat_array = np.random.randn(args.num_beats, 260)

    predicted_labels = load_model_and_predict(beat_array, ECGCNN, model_path=MODEL_SAVE_PATH, device=DEVICE)
    result = {"num_beats": len(predicted_labels), "predictions": predicted_labels}
    print(result)
