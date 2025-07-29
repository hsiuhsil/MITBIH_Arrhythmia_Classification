"""to predict the class of the heartbeat"""
import torch
import numpy as np
from model_definitions import AcharyaCNN, ECGCNN, iTransformer  
from config import DEVICE, MODEL_SAVE_PATH, LABEL_MAP 

def predict_single_beat(model, beat_tensor):
    """
    Predict the class of a single ECG beat using a trained model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        beat_tensor (torch.Tensor): Input tensor of shape [1, 1, 260].

    Returns:
        int: Predicted class index.
    """
    model.eval()
    with torch.no_grad():
        beat_tensor = beat_tensor.to(DEVICE)
        output = model(beat_tensor)
        _, predicted_class = torch.max(output.data, 1)
        return predicted_class.item()

def load_model_and_predict(beat_array, model_type="ECGCNN", model_path=MODEL_SAVE_PATH):
    """
    Load the trained model, preprocess a beat, and predict its class.

    Args:
        beat_array (np.ndarray): Raw beat data as a 1D NumPy array of shape (260,).
        model_type (str): Model architecture name.
        model_path (str): Path to the trained model file.

    Returns:
        str: Predicted class label.
    """
    # Convert the beat into the expected input shape: [1, 1, 260]
    beat_tensor = torch.tensor(beat_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Instantiate the model
    if model_type == "AcharyaCNN":
        model = AcharyaCNN()
    elif model_type == "ECGCNN":
        model = ECGCNN()
    elif model_type == "iTransformer":
        model = iTransformer()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    pred_class_idx = predict_single_beat(model, beat_tensor)
    return LABEL_MAP[pred_class_idx]

if __name__ == "__main__":

    # Example beat (should be of shape (260,))
    sample_beat = np.random.randn(260)  # Replace with actual preprocessed beat

    prediction = load_model_and_predict(sample_beat, model_type="ECGCNN")
    print("Predicted label:", prediction)
