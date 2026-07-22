运行说明：

一、后端（FastAPI）

1. 创建并激活虚拟环境（略）
2. 安装依赖：
   pip install -r backend/requirements.txt
3. 启动服务：
   cd backend
   uvicorn main:app --reload

二、前端（Vite + Vue）

1. 安装依赖：
   cd frontend
   npm install
2. 启动开发服务器：
   npm run dev

默认访问地址：
- 前端页面：http://localhost:5173
- 后端接口文档：http://127.0.0.1:8000/docs
