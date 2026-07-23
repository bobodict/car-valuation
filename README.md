# 二手车估值研究平台

一个面向二手车价格估计、模型评估和研究展示的全栈项目。项目由 Vue 3 前端、FastAPI 后端、SQLAlchemy 数据层和可复现的 Python 训练流水线组成。

它不是只提供一个“输入车辆、输出价格”的页面，而是把模型选择、独立测试、误差分组、数据来源、特征契约和估值结果放在同一个可检查的研究控制台中，适合展示完整的机器学习工程流程和可解释的模型评估。

## 项目亮点

- **车辆估值（四步向导）**：默认入口按“确认车辆身份、描述使用情况、补充车辆配置、确认车况并提交”四步收集品牌、车型、城市、里程、上牌年份、排量等字段。
- **估值结果与模型依据**：调用当前正式发布模型返回 INR 参考价格和价格区间，并可展开查看模型指标、数据来源和适用边界。
- **模型说明**：集中展示模型版本、质量门禁、开发集交叉验证、独立测试集、候选模型排行榜、误差分组、特征说明和数据来源。
- **历史记录**：每次估值会记录车辆摘要、价格、模型版本和时间，便于回看估值结果。
- **解释助手**：使用本地知识库和 OpenAI-compatible 接口；未配置 LLM 时明确返回禁用状态，不生成脱离数据的答案。
- **模型发布与兼容性校验**：v3 发布物包含 manifest、feature config、metrics、leaderboard、error analysis、model card 和发布代次标识；运行时会校验发布身份，避免读取不完整或切换中的模型。
- **可复现训练流程**：包含公开数据下载、字段标准化、特征工程、候选模型竞争、交叉验证、独立留出测试、误差分析和模型发布流程。

## 当前模型结果

当前仓库中正式发布的是 v3 CatBoost 模型。指标来自 `backend/models/model_card.json` 和 `backend/models/metrics.json`，金额单位为印度卢比 INR。

| 项目 | Development CV | Recorded test |
| --- | ---: | ---: |
| 用途 | 候选模型选择与调参 | 发布后的独立测试 |
| 样本数 | 1,750 | 309 |
| R² | 0.8713 ± 0.0475 | 0.8728 |
| RMSE | 873,565 INR | 795,609 INR |
| MAE | 266,789 INR | 231,376 INR |
| 10% 误差内准确率 | 54.1% | 60.2% |

质量门禁当前为 **PASS**，门槛为 `R² >= 0.0` 且 `10% 误差内准确率 >= 50%`。这里的 10% 准确率表示预测价格与真实价格的相对误差不超过 10%。

需要注意：质量门禁通过不等于模型具备生产级准确率，也不代表输出了经过校准的置信区间。独立测试集在候选模型选择完成后只使用一次，前端会明确区分 CV 指标和独立测试指标。

## 系统结构

```text
浏览器
  │
  ├── Vue 3 + Vite 前端
  │       ├── 车辆估值（默认入口，四步向导）
  │       ├── 估值结果与模型依据
  │       ├── 模型说明
  │       ├── 历史记录
  │       └── 解释助手
  │
  └── FastAPI
          ├── /api/predict          车辆估值与历史写入
          ├── /api/model-card       模型卡与研究证据
          ├── /api/model-health     质量门禁与运行状态
          ├── /api/metrics          指标文件
          ├── /api/history          历史估值记录
          └── /api/assistant         本地知识库 + 可选 LLM
                  │
                  ├── CatBoost v3 发布物
                  ├── legacy v2 MLP 兼容加载
                  └── SQLite / MariaDB 历史记录
```

## 目录说明

```text
.
├── backend/
│   ├── main.py                       FastAPI 应用与 API 路由
│   ├── schemas.py                    请求/响应数据契约
│   ├── config.py                     环境变量和模型目录配置
│   ├── services/                     特征工程、模型运行时、指标和助手服务
│   ├── scripts/                      下载、训练、评估和发布脚本
│   ├── models/                       当前正式模型及模型卡
│   ├── data/knowledge_base.json      本地可审计知识库
│   └── tests/                        后端测试
├── frontend/
│   ├── src/App.vue                   前端应用壳和导航
│   ├── src/components/               研究、估值、助手和历史组件
│   ├── src/researchEvidence.js       研究证据数据适配层
│   └── src/assets/                   视觉 token 和页面样式
├── docs/superpowers/                 设计说明和实现计划
├── PRODUCT.md                        产品定位和设计原则
└── README.md                         项目说明
```

训练产生的原始数据、标准化数据和 experiments 目录默认位于 `backend/data/raw/`、`backend/data/processed/` 和 `backend/experiments/`，这些内容不提交到 Git。本地导出目录也不属于 GitHub 源码仓库。

## 环境要求

- Python 3.11 或更高版本
- Node.js 20.19+ 或 22.12+
- npm
- 如果使用 MariaDB，需要可访问的 MariaDB 服务
- 如果启用解释助手，需要一个 OpenAI-compatible 的 `/chat/completions` 服务

## 快速启动

### 1. 启动后端

在项目根目录执行：

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

本地只想使用 SQLite 时，编辑 `backend/.env`，清空 `DB_USER`、`DB_PASSWORD`、`DB_NAME` 等 MariaDB 配置，或者直接设置：

```dotenv
DATABASE_URL=sqlite:///D:/car-valuation/backend/car_valuation.db
```

如果使用 MariaDB，则填写 `DB_HOST`、`DB_PORT`、`DB_USER`、`DB_PASSWORD` 和 `DB_NAME`。不要把真实密码提交到 Git。

启动服务：

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

后端接口文档：

- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>

### 2. 启动前端

另开一个终端：

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev -- --host 127.0.0.1 --port 5173
```

打开 <http://127.0.0.1:5173/>。前端通过 `VITE_API_BASE_URL` 或兼容旧配置的 `VITE_API_BASE` 指定后端地址，默认配置为 `http://127.0.0.1:8000`。

如果前端使用其他端口，例如 `5177`，需要在后端 `.env` 中把该地址加入 `FRONTEND_ORIGINS`，例如：

```dotenv
FRONTEND_ORIGINS=http://127.0.0.1:5177,http://127.0.0.1:5173
```

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/predict` | 校验车辆字段，返回价格、参考区间、模型版本和指标，并写入历史记录 |
| `GET` | `/api/history?limit=20` | 获取最近的估值记录，最多返回 200 条 |
| `GET` | `/api/model-card` | 获取数据来源、特征契约、切分、候选模型和限制条件 |
| `GET` | `/api/model-health` | 获取模型状态、质量门禁、指标和警告 |
| `GET` | `/api/metrics` | 获取当前正式模型的指标文件 |
| `POST` | `/api/assistant` | 查询本地知识库，并在配置 LLM 时执行可校验的结构化估值工具调用 |

完整请求字段和响应结构以 Swagger 文档及 `backend/schemas.py` 为准。

## 重新下载数据和训练

项目使用公开的 `car details v4` 数据集，来源是印度二手车市场，价格单位为 INR。当前数据源地址：

<https://raw.githubusercontent.com/chandanverma07/DataSets/master/car%20details%20v4.csv>

从 `backend` 目录执行：

```powershell
# 下载原始 CSV，并记录 URL、时间、字节数和 SHA-256
python -m scripts.download_public_dataset

# 下载、标准化、训练候选模型、评估独立测试集并发布通过门禁的模型
python -m scripts.train_model --download

# 评估一个已有 CSV，输出 RMSE、MAE、R²、10% 误差准确率和均值基线
python -m scripts.evaluate_model data/processed/normalized_training.csv
```

常用训练参数：

```powershell
python -m scripts.train_model `
  --dataset data/processed/normalized_training.csv `
  --models-dir models `
  --experiments-dir experiments `
  --seed 42 `
  --collection-year 2026
```

训练流程会执行以下步骤：

1. 标准化公开数据字段，并记录数据来源。
2. 生成车辆年龄、年均里程、单位排量功率等特征。
3. 只在 development 数据上进行候选模型竞争和交叉验证。
4. 固定候选模型后，在 recorded test 独立留出集上进行一次最终评估。
5. 生成 model card、leaderboard、error analysis 和发布 manifest。
6. 通过完整性、质量门禁和运行时 smoke test 后原子发布模型。

## 可选解释助手

在 `backend/.env` 中配置：

```dotenv
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-name
LLM_TIMEOUT_SECONDS=45
```

助手会先检索 `backend/data/knowledge_base.json`，并在需要时调用结构化 `estimate_vehicle` 工具。没有同时配置这三个核心变量时，接口返回明确的未配置状态，不会伪造解释内容。

## 测试和构建

后端：

```powershell
cd backend
python -m pytest -q
```

前端：

```powershell
cd frontend
npm test
npm run build
```

当前验证结果：后端 `295 passed, 4 skipped`；前端 35 项测试通过，生产构建通过。

## 已知限制

- 数据集只代表印度二手车市场，价格以 INR 记录，不能直接解释为其他国家或货币市场的价格。
- 原始公开数据没有可靠的车身类型字段，`vehicle_type` 使用 `car`；没有事故历史字段，`accident_history` 使用 `unknown`。
- 模型输出的参考区间是应用层展示范围，不是经过统计校准的置信区间。
- 独立测试集只用于最终检查一次，不能替代跨时间、跨地区或真实业务环境验证。
- 质量门禁通过只说明当前记录的评估规则通过，不代表生产环境可直接依赖。
- 解释助手依赖外部或本地 LLM 服务；未配置时，数值估值仍可独立运行。

## 项目讲解建议

建议按下面的顺序介绍项目：

1. 先说明公开数据、字段标准化和数据限制。
2. 展示 development CV 如何选择候选模型，以及为什么不能用测试集挑模型。
3. 展示 recorded test 的最终指标和质量门禁。
4. 展示价格分位、车型频率和 seen/unseen 分组误差，说明模型在哪些样本上更不稳定。
5. 最后运行一次估值，展示模型版本、数据来源和历史记录如何被保留下来。

这样项目的重点不只是“做了一个预测页面”，而是完整展示了从数据、训练、评估、发布到应用调用的机器学习工程闭环。

## 许可证和数据来源

本仓库用于展示一套可复现的车辆估值与模型评估流程。公开数据的使用请遵循原始数据集发布者的许可和平台规则。仓库不提交数据库密码、LLM 密钥、原始训练数据或本地导出目录。
