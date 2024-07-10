# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# 2. Create the app object
app = FastAPI()

#. Load trained Pipeline
model = load_model('fraud')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud detection system for Fastag transactions API!"}

# Define predict function
@app.post('/predict')
def predict(Vehicle_Type, Lane_Type, Vehicle_Dimensions, Vehicle_Speed):
    data = pd.DataFrame([[Vehicle_Type, Lane_Type, Vehicle_Dimensions, Vehicle_Speed]])
    data.columns = ['Vehicle_Type', 'Lane_Type', 'Vehicle_Dimensions', 'Vehicle_Speed']

    predictions = predict_model(model, data=data) 
    return {'prediction': int(predictions['Label'][0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)