# Backend

FastAPI service for the used-car valuation application.

## Run locally

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --reload
```

When no database variables are configured, the service uses a local SQLite file at `backend/car_valuation.db`. For MariaDB, fill in `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME` in `.env`.

Swagger is available at `http://127.0.0.1:8000/docs`.

## API

- `POST /api/predict`: validates vehicle fields and returns a price in万元, a provisional reference range, real artifact metrics, and `model_status=experimental`.
- `GET /api/history?limit=20`: returns up to 200 recent records.
- `GET /api/metrics`: returns the test metrics stored in `models/metrics.json`.

The supplied model is experimental. It has no calibrated confidence interval, and the project does not claim production accuracy. The training dataset and original training script are not part of this delivery yet.
