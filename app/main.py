from fastapi import FastAPI, HTTPException

from app.predictor import ModelRegistry
from app.schemas import PredictionRequest
from src.database import get_db_connection, get_latest_user_features

app = FastAPI(title="Sales Forecast API")
registry = ModelRegistry(base_dir="models")


@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/db-check")
def check_db():
    """Simple endpoint to verify the API can reach the MySQL database."""
    try:
        conn = get_db_connection()
        conn.close()
        return {"database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{horizon}")
def predict(horizon: str, request: PredictionRequest):
    if horizon not in {"weekly", "monthly", "yearly"}:
        raise HTTPException(status_code=400, detail="Invalid horizon")

    try:
        return registry.predict(horizon, request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@app.get("/predict/user/{user_id}/{horizon}")
def predict_for_user(user_id: int, horizon: str):
    """
    Fetches latest sales for a specific user from MySQL and returns a forecast.
    """
    if horizon not in {"weekly", "monthly", "yearly"}:
        raise HTTPException(status_code=400, detail="Invalid horizon")

    features = get_latest_user_features(user_id)

    if not features:
        raise HTTPException(
            status_code=404, 
            detail=f"No sales data found for user {user_id}. Please log sales first."
        )

    try:
        return registry.predict(horizon, features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))