from pydantic import BaseModel


class PredictionRequest(BaseModel):
    year: int
    month: int
    week_of_year: int
    lag_1: float
    lag_2: float
    lag_4_mean: float
    lag_8_mean: float