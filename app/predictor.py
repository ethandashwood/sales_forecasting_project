import json
from pathlib import Path

import joblib
import pandas as pd 
class ModelRegistry:
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.models = {}
        self.metadata = {}

        for horizon in ["weekly", "monthly", "yearly"]:
            model_path = self.base_dir / horizon / "model.joblib"
            meta_path = self.base_dir / horizon / "metadata.json"

            if model_path.exists():
                self.models[horizon] = joblib.load(model_path)

            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata[horizon] = json.load(f)

    def predict(self, horizon: str, payload: dict) -> dict:
        if horizon not in self.models:
            raise ValueError(f"No trained model found for horizon: {horizon}")

        model = self.models[horizon]
        df = pd.DataFrame([payload])

        month = df["month"].iloc[0] if "month" in df.columns and pd.notna(df["month"].iloc[0]) else 1
        df["quarter"] = ((int(month) - 1) // 3) + 1

        pred_per_sqft = float(model.predict(df)[0])
        predicted_sales = pred_per_sqft * float(payload["store_size_sqft"])

        return {
            "horizon": horizon,
            "predicted_sales": round(predicted_sales, 2),
            "predicted_sales_per_sqft": round(pred_per_sqft, 4),
            "model_version": "v1"
        }