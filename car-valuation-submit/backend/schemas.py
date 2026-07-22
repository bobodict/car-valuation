# schemas.py
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


CURRENT_YEAR = date.today().year


class PredictRequest(BaseModel):
    brand: str = Field(min_length=1, max_length=100)
    model: str | None = Field(default=None, max_length=255)
    city: str = Field(min_length=1, max_length=100)
    mileage: float = Field(ge=0, le=100)
    year: int = Field(ge=1980, le=CURRENT_YEAR)
    month: int = Field(ge=1, le=12)
    gearbox: Literal["自动", "手动", "其他"] = "自动"
    emission: Literal["国六", "国五", "国四", "其他"] = "国六"

    @field_validator("brand", "city")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("model")
    @classmethod
    def normalize_optional_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class HistoryQuery(BaseModel):
    limit: int = Field(default=20, ge=1, le=200)


class PriceRange(BaseModel):
    low: float = Field(ge=0)
    high: float = Field(ge=0)


class TestMetrics(BaseModel):
    mse: float
    rmse: float
    mae: float
    r2: float
    acc_10: float = Field(ge=0, le=1)


class MetricsResponse(BaseModel):
    model_status: Literal["experimental"]
    best_val_rmse: float | None = None
    test_metrics: TestMetrics


class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class Citation(BaseModel):
    source_id: str
    title: str


class PredictResponse(BaseModel):
    price: float
    range: PriceRange
    confidence: float | None = None
    model_status: Literal["experimental"]
    metrics: TestMetrics
    comment: str


class AssistantResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    estimate: PredictResponse | None = None
    llm_status: Literal["configured"]


class HistoryOut(BaseModel):
    id: int
    model: str | None
    city: str
    mileage: float
    year: int
    month: int
    gearbox: str
    emission: str
    price: float
    created_at: datetime
    status: str

    model_config = ConfigDict(from_attributes=True)
