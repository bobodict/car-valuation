from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


CURRENT_YEAR = date.today().year


class PredictRequest(BaseModel):
    brand: str = Field(min_length=1, max_length=100)
    model: str | None = Field(default=None, max_length=255)
    city: str = Field(min_length=1, max_length=100)
    mileage: float = Field(ge=0, le=10_000_000)
    year: int = Field(ge=1980, le=CURRENT_YEAR)
    month: int = Field(ge=1, le=12)
    gearbox: str = Field(default="Automatic", min_length=1, max_length=50)
    emission: str = Field(default="unknown", min_length=1, max_length=50)
    fuel_type: str = Field(default="unknown", min_length=1, max_length=50)
    displacement: float = Field(default=0.0, ge=0, le=10)
    seats: int = Field(default=5, ge=1, le=20)
    owner_count: int = Field(default=1, ge=1, le=20)
    vehicle_type: str = Field(default="car", min_length=1, max_length=50)
    color: str = Field(default="unknown", min_length=1, max_length=50)
    accident_history: str = Field(default="unknown", min_length=1, max_length=50)
    seller_type: str | None = Field(default=None, min_length=1, max_length=50)
    drivetrain: str | None = Field(default=None, min_length=1, max_length=50)
    max_power_bhp: float | None = Field(default=None, gt=0, le=2000)
    power_rpm: float | None = Field(default=None, gt=0, le=25000)
    max_torque_nm: float | None = Field(default=None, gt=0, le=10000)
    torque_rpm: float | None = Field(default=None, gt=0, le=25000)
    length_mm: float | None = Field(default=None, ge=1000, le=10000)
    width_mm: float | None = Field(default=None, ge=1000, le=5000)
    height_mm: float | None = Field(default=None, ge=500, le=5000)
    fuel_tank_liter: float | None = Field(default=None, gt=0, le=1000)

    @field_validator(
        "brand",
        "city",
        "gearbox",
        "emission",
        "fuel_type",
        "vehicle_type",
        "color",
        "accident_history",
    )
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

    @field_validator("seller_type", "drivetrain", mode="before")
    @classmethod
    def normalize_optional_non_blank_text(cls, value):
        if value is None or not isinstance(value, str):
            return value
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


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
    baseline_rmse: float | None = None
    baseline_r2: float | None = None
    evaluation_scope: str | None = None


class MetricsResponse(BaseModel):
    model_status: Literal["experimental", "usable"] = "experimental"
    quality_gate: Literal["pass", "fail"] = "fail"
    best_val_rmse: float | None = None
    test_metrics: TestMetrics
    currency: str = "INR"
    price_unit: str = "INR"
    mileage_unit: str = "km"
    data_source: dict = Field(default_factory=dict)
    model_version: str = "unknown"
    model_type: str = "mlp"
    feature_version: str = "2.0.0"
    sample_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ModelHealthResponse(BaseModel):
    model_status: Literal["experimental", "usable"]
    quality_gate: Literal["pass", "fail"]
    warnings: list[str] = Field(default_factory=list)
    metrics: TestMetrics
    data_status: str
    currency: str = "INR"
    price_unit: str = "INR"
    mileage_unit: str = "km"
    data_source: dict = Field(default_factory=dict)
    model_version: str = "unknown"
    model_type: str = "mlp"
    feature_version: str = "2.0.0"
    sample_count: int = 0


class ModelCardResponse(BaseModel):
    artifact_version: str
    feature_version: str
    model_version: str
    model_type: str = "mlp"
    currency: str
    price_unit: str
    mileage_unit: str
    sample_count: int = Field(gt=0)
    data_source: dict
    split: dict
    thresholds: dict
    quality_gate: Literal["pass", "fail"]
    test_metrics: TestMetrics
    category_options: dict[str, list[str]] = Field(default_factory=dict)
    feature_descriptions: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    cv_selection: dict = Field(default_factory=dict)
    independent_holdout: dict = Field(default_factory=dict)
    leaderboard: dict = Field(default_factory=dict)
    error_analysis: dict = Field(default_factory=dict)


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
    model_status: Literal["experimental", "usable"]
    quality_gate: Literal["pass", "fail"] = "fail"
    currency: str = "INR"
    price_unit: str = "INR"
    mileage_unit: str = "km"
    model_version: str = "unknown"
    model_type: str = "mlp"
    feature_version: str = "2.0.0"
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
    currency: str = "INR"
    model_version: str = "unknown"
    created_at: datetime
    status: str

    model_config = ConfigDict(from_attributes=True)

    @field_validator("currency", mode="before")
    @classmethod
    def default_currency_for_legacy_rows(cls, value):
        return value or "INR"

    @field_validator("model_version", mode="before")
    @classmethod
    def default_model_version_for_legacy_rows(cls, value):
        return value or "unknown"
