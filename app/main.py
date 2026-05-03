from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import ModelRegistry
from app.schemas import PredictionRequest


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

@app.post("/predict/weekly")
def predict_weekly(request: PredictionRequest):
    try:
        return registry.predict("weekly", request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weekly prediction failed: {str(e)}")


@app.post("/predict/monthly")

@@ -53,11 +35,11 @@ def predict_monthly(request: PredictionRequest):
    try:
        return registry.predict("monthly", request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monthly prediction failed: {str(e)}")


@app.post("/predict/user/{user_id}")
def predict_for_user(user_id: int, request: PredictionRequest):
    try:
        payload = request.model_dump()


@@ -68,33 +50,12 @@ def predict_and_save_for_user(user_id: int, request: PredictionRequest):
        monthly_prediction = monthly_result["predicted_revenue"]
        yearly_prediction = monthly_prediction * 12








        return {
            "user_id": user_id,
            "weekly_prediction": round(weekly_prediction, 2),
            "monthly_prediction": round(monthly_prediction, 2),
            "yearly_prediction": round(yearly_prediction, 2)

        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")