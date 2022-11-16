import json
from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import model_predict, get_config_file
from app.model.model import __version__ as model_version

# Initialize an instance of FastAPI
app = FastAPI()

# Define the input and output models
class InputData(BaseModel):
    gender: int
    hemoglobin: float
    MCH: float
    MCHC : float
    MCV : float 
class PredictionOut(BaseModel):
    output: int


# Define the default route
@app.get("/")
def root():
    return {"message": "Welcome to Anemia Predict Model FastAPI",
            "predict": "POST = /predict_anemia", "config_file": "GET = /config_file",
            "model accuracy": "..."}

# Define the route to predict anemia
@app.post("/predict", response_model=PredictionOut)
def predict(data: InputData):
    output = model_predict(data)
    return {"output": output}

# Define the config_file route
@app.get("/config_file")
def config_file():
    config_file = get_config_file()
    return json.load(config_file)
