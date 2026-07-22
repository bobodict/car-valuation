from datetime import datetime, timezone
from typing import List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import settings
from database import Base, engine, get_db
from models import History
from schemas import (
    AssistantRequest,
    AssistantResponse,
    HistoryOut,
    HistoryQuery,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)
from services.assistant_service import answer_user_message
from services.llm_client import LLMClientError, LLMNotConfiguredError
from services.metrics_service import load_metrics
from services.model_service import ModelServiceError, call_model_api


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="二手车价格预测系统后端",
    description="提供实验性二手车估值、历史记录和模型指标接口。",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.post("/api/predict", response_model=PredictResponse)
def predict_car(body: PredictRequest, db: Session = Depends(get_db)):
    try:
        result = call_model_api(body)
    except ModelServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    history = History(
        model=body.model or f"{body.brand}车型",
        city=body.city,
        mileage=body.mileage,
        year=body.year,
        month=body.month,
        gearbox=body.gearbox,
        emission=body.emission,
        price=result["price"],
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        status="experimental",
    )
    try:
        db.add(history)
        db.commit()
        db.refresh(history)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="历史记录暂时无法保存。") from exc

    return result


@app.get("/api/history", response_model=List[HistoryOut])
def get_history(
    query: HistoryQuery = Depends(),
    db: Session = Depends(get_db),
):
    return (
        db.query(History)
        .order_by(History.created_at.desc())
        .limit(query.limit)
        .all()
    )


@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics():
    return load_metrics()


@app.post("/api/assistant", response_model=AssistantResponse)
def assistant(body: AssistantRequest):
    try:
        return answer_user_message(body.message)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/")
def home():
    return {"message": "二手车价格预测系统后端运行正常。"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
