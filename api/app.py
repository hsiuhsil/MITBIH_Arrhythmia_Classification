from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import logging

from predict import load_model_and_predict
from mitbih.models.model_definitions import ECGCNN
from mitbih.utils.config import DEVICE, MODEL_SAVE_PATH
from mitbih.data.preprocessing import preprocess_beats

# Initialize FastAPI app
app = FastAPI(title="ECG Classification API", version="1.0")

# Input schema
class ECGInput(BaseModel):
    data: list  # Expect a list of lists, each with 260 samples


@app.post("/predict")
def predict_ecg(input_data: ECGInput):
    try:
        # Convert to numpy
        beat_array = np.array(input_data.data, dtype=np.float32)

        # Validate shape
        if beat_array.ndim != 2 or beat_array.shape[1] != 260:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape: expected (N, 260), got {beat_array.shape}",
            )

        # Preprocess
        logging.info("Preprocessing beats...")
        beat_preprocessed = preprocess_beats(beat_array)

        # Run prediction
        predicted_labels = load_model_and_predict(
            beat_preprocessed,
            ECGCNN,
            model_path=MODEL_SAVE_PATH,
            device=DEVICE,
        )

        return {
            "num_beats": len(predicted_labels),
            "predictions": predicted_labels,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
