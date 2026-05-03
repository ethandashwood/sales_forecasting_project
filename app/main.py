from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import ModelRegistry
from app.schemas import PredictionRequest
from src.database import get_db_connection, get_latest_user_features

app = FastAPI(title="Sales Forecast API")
registry = ModelRegistry(base_dir="models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def check_db():
    try:
        conn = get_db_connection()
        conn.close()
        return {"database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


@app.post("/predict/{horizon}")
def predict(horizon: str, request: PredictionRequest):
    if horizon not in {"weekly", "monthly"}:
        raise HTTPException(status_code=400, detail="Invalid horizon")

    try:
        return registry.predict(horizon, request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@app.get("/predict/user/{user_id}/{horizon}")
def predict_for_user(user_id: int, horizon: str):
    if horizon not in {"weekly", "monthly"}:
        raise HTTPException(status_code=400, detail="Invalid horizon")

    try:
        features = get_latest_user_features(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    if not features:
        raise HTTPException(
            status_code=404,
            detail=f"No saved prediction features found for user_id={user_id}"
        )

    try:
        return registry.predict(horizon, features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")