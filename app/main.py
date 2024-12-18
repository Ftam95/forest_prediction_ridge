from  fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os

# Paths to the models
scaler_path = os.path.join("models", "scaler.pkl")
model_path = os.path.join("models", "ridge.pkl")

# Load the scaler and model
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define the request schema
class PredictionRequest(BaseModel):
    Temperature: int
    RH: int
    Ws: int
    Rain: float
    FFMC: float
    DMC: float
    ISI: float
    Classes: int
    Region: int
    

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Forest Fire Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionRequest):
    try:
        # Extract features from the request
        features = np.array([[input_data.Temperature, input_data.RH, input_data.Ws, 
                               input_data.Rain, input_data.FFMC, input_data.DMC, 
                              input_data.ISI, input_data.Classes, input_data.Region]])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make a prediction
        prediction = model.predict(scaled_features)
        
        # Return the prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
