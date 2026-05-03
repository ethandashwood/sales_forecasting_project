import json
from pathlib import Path

import joblib
import pandas as pd


class ModelRegistry:
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.models = {}
        self.metadata = {}

        for horizon in ["weekly", "monthly"]:
            model_path = self.base_dir / horizon / "model.joblib"
            meta_path = self.base_dir / horizon / "metadata.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file: {model_path}")

            if not meta_path.exists():
                raise FileNotFoundError(f"Missing metadata file: {meta_path}")

            self.models[horizon] = joblib.load(model_path)

            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata[horizon] = json.load(f)

    def predict(self, horizon: str, payload: dict) -> dict:
        if horizon not in self.models:
            raise ValueError(f"No trained model found for horizon: {horizon}")

        model = self.models[horizon]
        features = self.metadata[horizon]["features"]

        df = pd.DataFrame([payload])
        df = df[features]

        prediction = float(model.predict(df)[0])

        return {
            "horizon": horizon,
            "predicted_revenue": round(prediction, 2),
            "model_version": "v1"
        }