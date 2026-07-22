# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List

from database import engine, Base, get_db
from models import History
from schemas import PredictRequest, PredictResponse, HistoryOut
from services.model_service import call_model_api

import uvicorn

# ======================================
# 1. 创建所有表（如果不存在）
# ======================================
Base.metadata.create_all(bind=engine)

# ======================================
# 2. 创建 FastAPI 应用
# ======================================
app = FastAPI(
    title="二手车价格预测系统后端",
    description="基于 FastAPI + MariaDB 的二手车价格预测 API 服务（等待接入真实模型）",
    version="1.0.0",
)

# ======================================
# 3. CORS 配置（允许前端跨域访问）
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 将来可以改成前端实际域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# 4. 核心接口：二手车价格预测
#    URL: POST /api/predict
#    Request JSON 字段：
#      brand, model, city, mileage, year, month, gearbox, emission
# ======================================
@app.post("/api/predict", response_model=PredictResponse)
def predict_car(
    body: PredictRequest,
    db: Session = Depends(get_db),
):
    # 4.1 调用“模型服务”（现在是规则估值，将来换成真实模型）
    try:
        result = call_model_api(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")

    # 4.2 写入历史记录
    history = History(
        model=body.model,
        city=body.city,
        mileage=body.mileage,
        year=body.year,
        month=body.month,
        gearbox=body.gearbox,
        emission=body.emission,
        price=result["price"],
        created_at=datetime.now(),
        status="success",
    )
    db.add(history)
    db.commit()
    db.refresh(history)

    # 4.3 返回给前端
    return PredictResponse(
        price=result["price"],
        range={"low": result["lower"], "high": result["upper"]},
        confidence=result["confidence"],
        comment=result["comment"],
    )

# ======================================
# 5. 历史记录查询接口（可选，但很实用）
#    URL: GET /api/history
#    参数：暂时不做筛选，返回最近 N 条
# ======================================
@app.get("/api/history", response_model=List[HistoryOut])
def get_history(
    limit: int = 20,
    db: Session = Depends(get_db),
):
    rows = (
        db.query(History)
        .order_by(History.created_at.desc())
        .limit(limit)
        .all()
    )
    return rows

# ======================================
# 6. 首页测试接口
# ======================================
@app.get("/")
def home():
    return {"message": "二手车价格预测系统后端运行正常！"}

# ======================================
# 7. 本地运行
# ======================================
# 也可以用命令行：
#   uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
