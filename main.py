# main.py
# checking the commit and push to github

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Define the path to the pickled model
MODEL_PATH = os.path.join('models', 'linear_regression_model.pkl')

# Initialize FastAPI app
app = FastAPI(
    title="Linear Regression Model Inference API",
    description="API for predicting values using a trained Linear Regression model.",
    version="1.0.0"
)

# Load the trained model at startup
model = None
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run linear_regression.py first.")
    # Exit or handle this gracefully in a production environment
    # For this example, we'll allow the app to start but predictions will fail
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Ensure model is None if loading fails

# Define the input data model using Pydantic
class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.
    Expects two float features.
    """
    feature_1: float
    feature_2: float

# Define the prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predicts the target value based on the input features using the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Cannot make predictions.")

    try:
        # Prepare the input data as a numpy array for the model
        input_data = np.array([[request.feature_1, request.feature_2]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Health check endpoint
@app.get("/")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "ok", "message": "Linear Regression API is running!"}

# You can run this app using Uvicorn:
# uvicorn main:app --reload
