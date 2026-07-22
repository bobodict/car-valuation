# schemas.py
from pydantic import BaseModel
from datetime import datetime

class PredictRequest(BaseModel):
    brand: str
    model: str
    city: str
    mileage: float
    year: int
    month: int
    gearbox: str
    emission: str

class PredictResponse(BaseModel):
    price: float
    range: dict   # {"low": float, "high": float}
    confidence: float
    comment: str

class HistoryOut(BaseModel):
    id: int
    model: str
    city: str
    mileage: float
    year: int
    month: int
    gearbox: str
    emission: str
    price: float
    created_at: datetime
    status: str

    class Config:
        from_attributes = True  # pydantic v2：从 ORM 对象读取字段
