# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Lab API")

# pretend registry
MODEL_REGISTRY = {"modelA": object(), "modelB": object()}

class PredictIn(BaseModel):
    features: dict

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return {"models": list(MODEL_REGISTRY.keys())}

# split prediction to a function so we can mock it in tests
def run_prediction(model_name: str, features: dict) -> dict:
    # imagine a heavy ML call here
    return {"model": model_name, "prediction": 42}

@app.post("/predict/{model_name}")
def predict(model_name: str, payload: PredictIn):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found")
    return run_prediction(model_name, payload.features)
