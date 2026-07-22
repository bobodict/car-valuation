# Car Valuation Baseline Repair Design

## Goal

Repair the existing used-car valuation project into an honest, reproducible baseline before adding any large-language-model features.

## Scope

This phase keeps Vue, FastAPI, SQLAlchemy, MariaDB, and the supplied PyTorch model. It does not add chat, RAG, fine-tuning, or external model APIs.

The phase addresses:

- consistent feature names, Chinese category values, and mileage units;
- strict request validation and bounded history queries;
- real model metrics exposed by the backend instead of metrics recomputed from predictions;
- explicit experimental-model status and evidence-based confidence text;
- environment configuration that can be reproduced without committing credentials;
- frontend loading/error states and removal of placeholder detail behavior;
- focused backend tests for the corrected contracts.

## Architecture

`backend/main.py` remains the HTTP entrypoint. `backend/services/model_service.py` becomes the single adapter between the public request schema and the model feature schema. The adapter owns category normalization, unit conversion, output validation, and response metadata.

`backend/services/metrics_service.py` reads the committed `models/metrics.json` artifact and returns it through `GET /api/metrics`. The frontend consumes that endpoint and labels the metrics as holdout/test metrics. It does not derive RMSE, MAE, or R2 from historical predictions.

`backend/config.py` owns environment settings and model paths resolved relative to the backend package, so startup does not depend on the current working directory. `database.py` uses the same configuration and keeps database session creation isolated.

The submission directory is the source of truth. The root-level backend mirror is updated only where it is currently identical, so existing local startup commands do not silently use stale behavior.

## Public Contracts

`PredictRequest` validates:

- brand and city as non-empty strings;
- model as an optional non-empty string;
- mileage in万公里 within a non-negative practical range;
- year between 1980 and the current year;
- month between 1 and 12;
- gearbox and emission against the supported UI values.

`PredictResponse` contains:

- `price` and `range` in万元;
- `confidence` as nullable because the supplied model has no calibrated confidence;
- `model_status` set to `experimental`;
- `metrics` containing the artifact's test metrics;
- a comment that does not claim unsupported accuracy.

The history endpoint validates `limit` between 1 and 200. The metrics endpoint is read-only.

## Model Handling

The adapter maps UI values to the categories present in `feature_config.json` and supplies Chinese defaults matching the fitted encoder: `自动`, `手动`, `汽油`, `轿车`, `白色`, `无事故`. Public fields that are not part of the fitted model remain visible as input metadata but are not described as predictive features.

The raw prediction is rejected when it is non-finite or non-positive. The service must not silently replace a failed model call with an unrelated heuristic while reporting success. A failure returns an HTTP 503 with an actionable message.

Because the supplied artifact has negative R2 and no calibrated uncertainty, the API does not return a numeric confidence. The displayed range is marked as a provisional reference interval, not a statistical confidence interval.

## Frontend Behavior

The valuation page shows the experimental status and the real metric values returned by the backend. The dashboard uses the metrics endpoint for model metrics and keeps historical prediction charts separate from model evaluation metrics.

Prediction and history requests show loading and error states. History detail displays the actual stored fields in a lightweight dialog or inline detail panel; it does not show a placeholder alert. The API base URL is configurable through Vite environment variables with a local default.

## Testing

Tests are written first for:

1. request validation and bounded history limits;
2. Chinese category normalization and mileage conversion;
3. response metadata and real metric loading;
4. invalid model output and model failure behavior;
5. frontend production compilation and backend Python compilation.

The existing model artifact is exercised with a deterministic smoke input. The tests do not claim that the supplied model is accurate; they verify that the application reports its limitations honestly and preserves the feature contract.

## Known Boundary

The supplied project does not include the training dataset, training script, or a reproducible split. Therefore this phase cannot establish better predictive accuracy. A later model-quality phase must add data provenance, a baseline comparison, retraining, calibration, and an evaluation set before any production accuracy claim.
