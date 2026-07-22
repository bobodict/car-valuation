# Car Valuation

Used-car valuation application built with Vue 3, FastAPI, SQLAlchemy, MariaDB, and a PyTorch model.

The delivery source is [`car-valuation-submit`](./car-valuation-submit). The root-level Python files and the `qianduan` directory were local duplicate copies and are intentionally excluded from version control.

## Structure

- `car-valuation-submit/backend`: FastAPI service, database models, model artifacts, and prediction adapter
- `car-valuation-submit/frontend`: Vue 3 + Vite client
- `docs`: design and implementation notes

The assistant upgrade adds a local auditable knowledge base and an optional OpenAI-compatible model client. Configure `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in the backend `.env` to enable it; without them the assistant returns an explicit disabled response.

## Current Model Status

The supplied model is an experimental artifact. Its recorded holdout metrics are kept in `car-valuation-submit/backend/models/metrics.json`; the project does not claim production accuracy or calibrated confidence. The public dataset, adapter, downloader, training script, evaluation script, and model card are included for reproducibility. The current test R2 is 0.8636, but the 10% error accuracy is 29.1%, so the quality gate remains failed.

## Local Setup

See:

- [`backend/README.md`](./car-valuation-submit/backend/README.md)
- [`frontend/Readme.md`](./car-valuation-submit/frontend/Readme.md)

Create `car-valuation-submit/backend/.env` from `.env.example` before starting the backend. Never commit database credentials.
