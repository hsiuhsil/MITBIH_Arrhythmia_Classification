import torch
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List

from scripts.predict import load_model_and_predict
from mitbih.models.model_definitions import ECGCNN
from mitbih.utils.config import DEVICE, MODEL_SAVE_PATH
from mitbih.data.preprocessing import preprocess_beats

# Initialize FastAPI app
app = FastAPI(title="ECG Classification API", version="1.0")

# Input schema
class ECGInput(BaseModel):
    data: List[List[float]]  # Expect a list of lists, each with 260 samples

# JSON endpoint
@app.post("/predict_json")
def predict_ecg_json(input_data: ECGInput):
    try:
        beat_array = np.array(input_data.data, dtype=np.float32)

        # Validate shape
        if beat_array.ndim != 2 or beat_array.shape[1] != 260:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape: expected (N, 260), got {beat_array.shape}",
            )

        # Preprocess and predict
        logging.info("Preprocessing beats...")
        beat_preprocessed = preprocess_beats(beat_array)
        predicted_labels = load_model_and_predict(beat_preprocessed, ECGCNN)

        return {"predictions": predicted_labels}

    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


# CSV endpoint
@app.post("/predict_csv")
async def predict_ecg_csv(file: UploadFile = File(...)):
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(file.file, header=None)
        beat_array = df.to_numpy(dtype=np.float32)

        # Validate shape
        if beat_array.ndim != 2 or beat_array.shape[1] != 260:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid CSV shape: expected (N, 260), got {beat_array.shape}",
            )

        # Preprocess and predict
        logging.info("Preprocessing beats from CSV...")
        beat_preprocessed = preprocess_beats(beat_array)
        predicted_labels = load_model_and_predict(beat_preprocessed, ECGCNN)

        return {"predictions": predicted_labels}

    except Exception as e:
        logging.exception("CSV prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
