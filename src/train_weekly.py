import json
from math import sqrt
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import aggregate_sales, add_group_lags, add_target, add_time_features


FEATURES = [
    "business_type",
    "store_type",
    "store_size_sqft",
    "region",
    "product_category",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "lag_1",
    "lag_2",
    "lag_4_mean",
    "lag_8_mean",
]

CATEGORICAL = ["business_type", "store_type", "region", "product_category"]
NUMERICAL = ["store_size_sqft", "year", "month", "quarter", "week_of_year", "lag_1", "lag_2", "lag_4_mean", "lag_8_mean"]


def build_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/interim/combined.csv", parse_dates=["date"])
    df = aggregate_sales(df, "W")
    df = add_time_features(df)
    df = add_group_lags(df, ["store_id", "product_category"])
    df = add_target(df)
    df = df.dropna(subset=["target"])
    return df


def main():
    df = build_dataset()

    split_date = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split_date].copy()
    valid_df = df[df["date"] > split_date].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["target"]
    X_valid = valid_df[FEATURES]
    y_valid = valid_df["target"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                CATEGORICAL,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                NUMERICAL,
            ),
        ]
    )

    model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds)
    rmse = sqrt(mean_squared_error(y_valid, preds))

    save_dir = Path("models/weekly")
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, save_dir / "model.joblib")
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "mae": mae, "rmse": rmse}, f, indent=2)

    print({"weekly_mae": mae, "weekly_rmse": rmse})


if __name__ == "__main__":
    main()