from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="IRIS Prediction API")
model = joblib.load("model.joblib")

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"status": "IRIS API is running"}

@app.post("/predict")
def predict(data: IrisInput):
    X = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    pred = model.predict(X)[0]
    return {"prediction": CLASS_NAMES[int(pred)]}
