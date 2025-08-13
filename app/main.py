from fastapi import FastAPI, UploadFile, File
import numpy as np
import torch
from .predict import load_model_and_predict
from .model_definitions import ECGCNN
from .config import MODEL_SAVE_PATH, DEVICE

app = FastAPI(title="MITBIH Heartbeat Classifier API")

@app.get("/")
def home():
    return {"message": "MITBIH Heartbeat Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded .npy file
    contents = await file.read()
    beats = np.load(contents)  # shape [N, 260]

    predictions = load_model_and_predict(beats, ECGCNN, MODEL_SAVE_PATH, DEVICE)
    return {"predictions": predictions}
