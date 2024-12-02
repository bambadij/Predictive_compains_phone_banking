from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')

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

# Define the mapping for categorical variables (if encoded)
categorical_mapping = {
    "job": ['admin.', 'technician', 'blue-collar', 'management', 'retired', 'services', 'self-employed', 'entrepreneur', 'unemployed', 'housemaid', 'student', 'unknown'],
    "marital": ['married', 'single', 'divorced'],
    "education": ['secondary', 'tertiary', 'primary', 'unknown'],
    "default": ['no', 'yes'],
    "housing": ['no', 'yes'],
    "loan": ['no', 'yes'],
    "contact": ['unknown', 'telephone', 'cellular'],
    "month": ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    "poutcome": ['unknown', 'other', 'failure', 'success']
}

# Utility function to preprocess the input data
def preprocess_input(data: PredictionInput):
    # Convert input data to a feature array
    features = [
        data.age,
        categorical_mapping['job'].index(data.job),
        categorical_mapping['marital'].index(data.marital),
        categorical_mapping['education'].index(data.education),
        categorical_mapping['default'].index(data.default),
        data.balance,
        categorical_mapping['housing'].index(data.housing),
        categorical_mapping['loan'].index(data.loan),
        categorical_mapping['contact'].index(data.contact),
        data.day,
        categorical_mapping['month'].index(data.month),
        data.duration,
        data.campaign,
        data.pdays,
        data.previous,
        categorical_mapping['poutcome'].index(data.poutcome),
    ]
    return np.array([features])

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
