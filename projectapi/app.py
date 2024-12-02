from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained Gradient Boosting model and encoder
model = joblib.load('gradient_boosting_model.pkl')
encoder = joblib.load("encoder.pkl")

# Define categorical columns
categorical_columns = [
    "job", "marital", "education", "default", "housing", 
    "loan", "contact", "month", "poutcome"
]

# Define the input data schema
class PredictionInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# Utility function to preprocess the input data
def preprocess_input(data: PredictionInput):
    # Convert input data to a DataFrame for compatibility
    input_df = {
        "age": [data.age],
        "job": [data.job],
        "marital": [data.marital],
        "education": [data.education],
        "default": [data.default],
        "balance": [data.balance],
        "housing": [data.housing],
        "loan": [data.loan],
        "contact": [data.contact],
        "day": [data.day],
        "month": [data.month],
        "duration": [data.duration],
        "campaign": [data.campaign],
        "pdays": [data.pdays],
        "previous": [data.previous],
        "poutcome": [data.poutcome],
    }
    input_df = pd.DataFrame(input_df)

    # Apply OneHotEncoder to categorical columns
    encoded_features = encoder.transform(input_df[categorical_columns])

    # Combine encoded features with numerical features
    numerical_features = input_df.drop(columns=categorical_columns).values
    final_features = np.concatenate([numerical_features, encoded_features], axis=1)

    return final_features

# Define the GET endpoint to show the structure
@app.get("/structure")
async def get_structure():
    # Example data to return
    example_data = {
        "age": 0,
        "job": "admin",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 0,
        "housing": "no",
        "loan": "no",
        "contact": "telephone",
        "day": 0,
        "month": "jan",
        "duration": 0,
        "campaign": 0,
        "pdays": 0,
        "previous": 0,
        "poutcome": "success"
    }
    return example_data

# Define a POST endpoint for predictions
@app.post("/predict")
async def predict(data: PredictionInput):
    # Preprocess the input
    input_data = preprocess_input(data)

    # Make a prediction
    prediction = model.predict(input_data)

    # Convert prediction to "yes"/"no"
    response = "yes" if prediction[0] == 1 else "no"

    return {"prediction": response}
