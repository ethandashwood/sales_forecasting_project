from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    business_type: str
    store_type: str
    store_size_sqft: float = Field(gt=0)
    region: str
    product_category: str
    year: int
    month: Optional[int] = None
    week_of_year: Optional[int] = None
    lag_1: Optional[float] = None
    lag_2: Optional[float] = None
    lag_4_mean: Optional[float] = None
    lag_8_mean: Optional[float] = None


class PredictionResponse(BaseModel):
    horizon: str
    predicted_sales: float
    predicted_sales_per_sqft: float
    model_version: str