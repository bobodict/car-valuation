一、环境准备

1. 创建并激活虚拟环境（示例）：
   python -m venv venv
   venv\Scripts\activate   （Windows）

2. 安装依赖：
   pip install -r requirements.txt

二、启动后端服务

1. 进入 backend 目录：
   cd backend

2. 启动 FastAPI 服务：
   uvicorn main:app --reload

默认地址：
- 接口文档（Swagger）：http://127.0.0.1:8000/docs
- 预测接口：POST /api/predict
- 历史记录接口：GET /api/history
