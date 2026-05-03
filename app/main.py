from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import ModelRegistry
from app.schemas import PredictionRequest
from src.database import (
    get_db_connection,
    get_user_predictions,
    save_or_update_user_predictions,
)

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
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )


@app.post("/predict/weekly")
def predict_weekly(request: PredictionRequest):
    try:
        return registry.predict("weekly", request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/monthly")
def predict_monthly(request: PredictionRequest):
    try:
        return registry.predict("monthly", request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/user/{user_id}")
def predict_and_save_for_user(user_id: int, request: PredictionRequest):
    try:
        payload = request.model_dump()

        weekly_result = registry.predict("weekly", payload)
        monthly_result = registry.predict("monthly", payload)

        weekly_prediction = weekly_result["predicted_revenue"]
        monthly_prediction = monthly_result["predicted_revenue"]
        yearly_prediction = monthly_prediction * 12

        save_or_update_user_predictions(
            user_id=user_id,
            weekly_prediction=weekly_prediction,
            monthly_prediction=monthly_prediction,
            yearly_prediction=yearly_prediction,
        )

        return {
            "user_id": user_id,
            "weekly_prediction": round(weekly_prediction, 2),
            "monthly_prediction": round(monthly_prediction, 2),
            "yearly_prediction": round(yearly_prediction, 2),
            "message": "Predictions saved successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/user/{user_id}")
def get_predictions_for_user(user_id: int):
    result = get_user_predictions(user_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for user_id={user_id}"
        )

    return result