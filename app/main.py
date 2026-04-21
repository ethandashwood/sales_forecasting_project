from fastapi import FastAPI, HTTPException

from app.predictor import ModelRegistry
from app.schemas import PredictionRequest

app = FastAPI(title="Sales Forecast API")
registry = ModelRegistry(base_dir="models")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict/{horizon}")
def predict(horizon: str, request: PredictionRequest):
    if horizon not in {"weekly", "monthly", "yearly"}:
        raise HTTPException(status_code=400, detail="Invalid horizon")

    try:
        return registry.predict(horizon, request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))