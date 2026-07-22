# Frontend

Vue 3 + Vite client for the used-car valuation service.

## Run locally

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

The API base URL is read from `VITE_API_BASE` and defaults to `http://127.0.0.1:8000`.

`npm run build` creates the production bundle in `dist`.
