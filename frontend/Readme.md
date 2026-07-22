# Frontend

Vue 3 + Vite client for the used-car valuation service.

## Run locally

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

The API base URL is read from `VITE_API_BASE_URL` (or the legacy `VITE_API_BASE`) and defaults to the same-origin path. For local Vite development, run the backend at `http://127.0.0.1:8000` and set `VITE_API_BASE_URL=http://127.0.0.1:8000` in `.env`.

The explanation assistant is available when the backend has `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` configured. Without them, the page shows the disabled provider state.

`npm run build` creates the production bundle in `dist`.
