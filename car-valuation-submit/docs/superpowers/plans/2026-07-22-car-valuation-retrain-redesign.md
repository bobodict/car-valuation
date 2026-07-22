# Car Valuation Retraining and Research Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Replace the missing/non-reproducible training path and damaged UI with an evidence-driven INR vehicle valuation demo whose data source, model quality, units, and limitations are visible and testable.

**Architecture:** A downloader writes a cached public CSV and provenance manifest, a pure adapter normalizes it to the existing 14-field contract, and a deterministic training module produces compatible PyTorch/sklearn artifacts plus a model card. FastAPI exposes prediction, health, metrics, and model-card facts; Vue consumes those facts and renders the approved light Research Console using focused components.

**Tech Stack:** Python 3, pandas, scikit-learn 1.6.1, PyTorch CPU, FastAPI, Pydantic v2, SQLAlchemy/SQLite, Vue 3, Vite, Chart.js, browser smoke tests.

---

## File Map

Backend files to create:

- backend/services/public_dataset_adapter.py: pure transformation functions for car details v4.csv.
- backend/services/model_metadata.py: load and validate model_card.json.
- backend/scripts/download_public_dataset.py: download, hash, cache, and manifest the public source.
- backend/scripts/train_model.py: deterministic split, preprocessing, MLP training, metrics, gate, and atomic publishing.
- backend/models/model_card.json: generated provenance and feature metadata.
- backend/tests/test_public_dataset_adapter.py, test_dataset_downloader.py, test_training_pipeline.py, test_model_metadata.py: focused backend tests.

Backend files to modify:

- backend/services/dataset_contract.py: normalized column order, dtype rules, and validation.
- backend/services/model_quality_service.py and metrics_service.py: artifact-driven status and expanded metrics.
- backend/services/model_service.py and predict_service.py: full request mapping and explicit INR/km units.
- backend/schemas.py and main.py: response fields and GET /api/model-card.
- backend/models.py and database.py: currency/model-version history columns and a narrow SQLite migration.
- backend/data/README.md and backend/README.md: source, units, mapping, training, and evaluation documentation.
- Existing backend tests: replace corrupted literals with valid UTF-8 or escaped literals and update contracts.

Frontend files to create:

- frontend/src/api.js
- frontend/src/components/StatusStrip.vue
- frontend/src/components/ValuationForm.vue
- frontend/src/components/EstimatePanel.vue
- frontend/src/components/ModelEvidence.vue
- frontend/src/components/AssistantPanel.vue
- frontend/src/components/HistoryLog.vue

Frontend files to modify:

- frontend/src/App.vue
- frontend/src/assets/base.css
- frontend/src/assets/main.css
- frontend/Readme.md

Generated artifacts remain reviewable and tracked after training: backend/models/price_mlp.pt, preprocess.joblib, feature_config.json, metrics.json, and model_card.json. Raw CSV, cache directories, databases, secrets, node_modules, and .superpowers output remain ignored.

## Task 1: Lock the normalized dataset contract

Files:

- Modify backend/services/dataset_contract.py and backend/data/README.md.
- Create backend/services/public_dataset_adapter.py.
- Create backend/tests/test_public_dataset_adapter.py.

- [ ] Step 1: Write failing adapter tests.

Construct a raw two-row pandas frame and test the exact source mapping. The first row should contain Honda, Price 505000, Kilometer 87150, Engine 1198 cc, Owner First, and Seating Capacity 5. Assert normalized columns equal REQUIRED_DATASET_COLUMNS; mileage is 87150; displacement is 1.198; owner_count is 1; vehicle_type is car; accident_history is unknown. The second row should use Engine not supplied and Owner Fifth, then assert displacement is missing and owner_count is missing. Add tests for parse_engine_liters("2.0L"), parse_engine_liters("1498 cc"), map_owner_count("Second"), and missing source columns raising ValueError naming the column.

- [ ] Step 2: Run the focused tests and verify they fail.

Run:

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_public_dataset_adapter -v

Expected: collection fails because public_dataset_adapter and its functions do not exist.

- [ ] Step 3: Implement the pure adapter and contract validation.

Define RAW_REQUIRED_COLUMNS, REQUIRED_DATASET_COLUMNS, parse_engine_liters(value), map_owner_count(value), adapt_car_details_v4(frame), and validate_normalized_frame(frame). Parse case-insensitive cc values by dividing by 1000, keep L values as liters, and return numpy.nan for unparseable values. Use this mapping:

    normalized = pd.DataFrame({
        "price": raw["Price"],
        "mileage": raw["Kilometer"],
        "displacement": raw["Engine"].map(parse_engine_liters),
        "seats": pd.to_numeric(raw["Seating Capacity"], errors="coerce"),
        "owner_count": raw["Owner"].map(map_owner_count),
        "year": pd.to_numeric(raw["Year"], errors="coerce"),
        "brand": raw["Make"].fillna("unknown").astype(str).str.strip(),
        "model": raw["Model"].fillna("unknown").astype(str).str.strip(),
        "city": raw["Location"].fillna("unknown").astype(str).str.strip(),
        "transmission": raw["Transmission"].fillna("unknown").astype(str).str.strip(),
        "fuel_type": raw["Fuel Type"].fillna("unknown").astype(str).str.strip(),
        "vehicle_type": "car",
        "color": raw["Color"].fillna("unknown").astype(str).str.strip(),
        "accident_history": "unknown",
    })

Return columns in contract order. Preserve numeric missing values for the sklearn imputer. Reject non-positive price, negative mileage, and years outside 1980 through the current year.

- [ ] Step 4: Run tests and commit.

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_public_dataset_adapter -v
    git add backend/services/dataset_contract.py backend/services/public_dataset_adapter.py backend/tests/test_public_dataset_adapter.py backend/data/README.md
    git commit -m "feat: add public vehicle dataset adapter"

Expected: focused tests pass and no CSV data is staged.

## Task 2: Add reproducible download and provenance

Files:

- Create backend/scripts/download_public_dataset.py and backend/tests/test_dataset_downloader.py.
- Modify .gitignore and backend/data/README.md.

- [ ] Step 1: Write failing downloader tests.

Test download_dataset(url, destination, metadata_path, opener=...) with a fake response containing b"Make,Model,Price\nHonda,Amaze,505000\n". Assert destination bytes, URL, byte count, SHA-256, UTC timestamp, source filename, and raw_path in the manifest. Add a non-200 test that raises DownloadError and leaves an existing destination untouched.

- [ ] Step 2: Run the focused tests.

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_dataset_downloader -v

Expected: import failure for scripts.download_public_dataset.

- [ ] Step 3: Implement the downloader.

Use urllib.request.urlopen with a bounded timeout. Write bytes to a temporary file beside the destination, calculate SHA-256 from those exact bytes, replace the destination only after a successful response, and write JSON containing source_id car-details-v4, source_url, retrieved_at, sha256, byte_count, and raw_path. The CLI defaults to backend/data/raw/car-details-v4.csv and supports --url, --destination, and --metadata.

- [ ] Step 4: Run tests, verify ignore rules, and commit.

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_dataset_downloader -v
    git check-ignore backend/data/raw/car-details-v4.csv
    git add .gitignore backend/scripts/download_public_dataset.py backend/tests/test_dataset_downloader.py backend/data/README.md
    git commit -m "feat: add public dataset provenance downloader"

Expected: tests pass and git check-ignore prints the raw CSV path.

## Task 3: Build the deterministic training pipeline

Files:

- Create backend/scripts/train_model.py and backend/tests/test_training_pipeline.py.
- Modify backend/services/dataset_contract.py and backend/models/feature_config.json.

- [ ] Step 1: Write failing tests around split safety, baseline metrics, and atomic publishing.

Create a normalized fixture with at least 30 rows and varied price, brand, and year values. Test split_dataset(frame, seed=42) twice and assert identical train/validation/test indexes and pairwise disjoint indexes. Test calculate_metrics(actual, predicted, baseline_prediction) returns rmse, mae, r2, acc_10, baseline_rmse, and baseline_r2. Test assess_quality_gate fails for negative R2 and passes for perfect predictions. Test a failed gate does not replace an existing artifact directory unless allow_failed_publish is true.

- [ ] Step 2: Run focused tests and verify missing functions fail.

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_training_pipeline -v

Expected: import or attribute failures for the new training functions.

- [ ] Step 3: Implement split, preprocessing, compatible MLP training, metrics, and publishing.

Implement these functions in scripts.train_model:

    split_dataset(frame, seed=42)
    build_preprocessor(numeric_features, categorical_features)
    calculate_metrics(actual, predicted, baseline_prediction)
    assess_quality_gate(metrics, thresholds)
    train_and_publish(dataset_path, models_dir, seed=42, allow_failed_publish=False)

Use a 70/15/15 split with train_test_split in two stages and random_state=seed. Fit median imputation plus StandardScaler for numeric features and most-frequent imputation plus OneHotEncoder(handle_unknown="ignore") for categories, wrapped in ColumnTransformer; fit only on train. Train the existing MLPRegressor shape [128, 64] with a fixed torch seed, Adam, MSE loss, validation-RMSE early stopping, and at most 300 epochs. Evaluate test metrics against a baseline equal to the train-set mean.

Write all artifacts into a temporary sibling directory. The checkpoint contains input_dim, hidden_dims, dropout, model_state, and artifact_version. Replace the target model directory only if training succeeds and either the gate passes or allow_failed_publish is true. Include quality_gate, thresholds, data_source, currency INR, price_unit INR, mileage_unit km, seed, split sizes, training timestamp, and source SHA-256 in metrics.json and model_card.json.

- [ ] Step 4: Run tests, download data, and train the experimental artifact.

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_training_pipeline -v
    Push-Location backend
    D:/car-valuation/venv/Scripts/python.exe -m scripts.train_model --download --allow-failed-publish
    Pop-Location

Expected: JSON output includes sample count, test metrics, baseline metrics, and quality gate. A failed gate remains visible.

- [ ] Step 5: Evaluate with the exact served artifact and commit.

    Push-Location backend
    D:/car-valuation/venv/Scripts/python.exe -m scripts.evaluate_model data/processed/normalized_training.csv
    Pop-Location
    git add backend/scripts/train_model.py backend/tests/test_training_pipeline.py backend/services/dataset_contract.py backend/models/*.json
    git commit -m "feat: add reproducible vehicle model training"

Expected: evaluator metrics agree with metrics.json.

## Task 4: Expose model card and quality facts through FastAPI

Files:

- Create backend/services/model_metadata.py and backend/tests/test_model_metadata.py.
- Modify backend/services/metrics_service.py, backend/services/model_quality_service.py, backend/schemas.py, and backend/main.py.

- [ ] Step 1: Write failing metadata and health tests.

Use a fixture card with source_id, source_url, currency, price_unit, mileage_unit, sample_count, feature_version, model_version, split, thresholds, and category_options. Assert missing currency or non-positive sample_count raises a validation error. Assert get_model_card returns source facts and model_health reports the stored gate and warning without contradictory recomputation.

- [ ] Step 2: Run focused tests.

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_model_metadata -v

Expected: import failure for services.model_metadata or missing response fields.

- [ ] Step 3: Implement metadata loading and response schemas.

Create load_model_card() to read and validate backend/models/model_card.json. Extend MetricsResponse and ModelHealthResponse with quality_gate, currency, price_unit, mileage_unit, data_source, model_version, sample_count, and warnings. Add ModelCardResponse with category options and feature descriptions. Make get_model_health use the stored gate and thresholds; only add a warning when stored metrics disagree with the stored gate.

- [ ] Step 4: Register the endpoint and run all backend tests.

Add this endpoint:

    @app.get("/api/model-card", response_model=ModelCardResponse)
    def model_card():
        return load_model_card()

Update corrupted test literals and preserve the check that negative R2 cannot be promoted. Run:

    D:/car-valuation/venv/Scripts/python.exe -m unittest discover -s backend/tests -v

Expected: all existing and new backend tests pass.

- [ ] Step 5: Commit the metadata contract.

    git add backend/services/model_metadata.py backend/services/metrics_service.py backend/services/model_quality_service.py backend/schemas.py backend/main.py backend/tests
    git commit -m "feat: expose model card and quality metadata"

## Task 5: Repair prediction units, full feature mapping, and history metadata

Files:

- Modify backend/services/model_service.py, backend/predict_service.py, backend/schemas.py, backend/models.py, backend/database.py, and backend/main.py.
- Create backend/tests/test_prediction_contract.py.

- [ ] Step 1: Write failing request/response tests.

Create a valid request containing displacement, seats, owner_count, fuel_type, vehicle_type, color, and accident_history, with mileage explicitly measured in km. Assert build_model_input passes those values through, maps UI gearbox to the model category, returns currency INR and price_unit INR, and does not divide raw model prediction by 10000. Assert a blank required category or negative seat count raises Pydantic validation error, and model failure never falls back to a rule price.

- [ ] Step 2: Run focused tests and confirm old behavior fails.

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_prediction_contract -v

Expected: failures because the old adapter hardcodes fuel, displacement, color, seats, accident status, and divides prediction by 10000.

- [ ] Step 3: Implement the explicit input/output contract.

Keep the public field name mileage but define it as km in PredictRequest and ModelCardResponse. Add bounded numeric/category fields. Replace the old gearbox helper with a source-category mapping, use unknown for missing accident history, and build the exact FEATURE_COLS dictionary. Return raw INR price, a reference range computed as raw_price times 0.92 and 1.08, confidence null, quality_gate from the model card, explicit currency/price_unit/mileage_unit/model_version, metrics, and an experimental explanation when the gate fails.

Add currency and model_version to the history model. On SQLite startup inspect PRAGMA table_info(history) and issue only ALTER TABLE ADD COLUMN statements for missing nullable columns. Do not delete existing history.

- [ ] Step 4: Run prediction and full backend tests.

    D:/car-valuation/venv/Scripts/python.exe -m unittest backend.tests.test_prediction_contract backend.tests.test_baseline_contract -v
    D:/car-valuation/venv/Scripts/python.exe -m unittest discover -s backend/tests -v

Expected: all tests pass; response price remains INR and old 10000 conversion is absent.

- [ ] Step 5: Commit the serving contract.

    git add backend/services/model_service.py backend/predict_service.py backend/schemas.py backend/models.py backend/database.py backend/main.py backend/tests/test_prediction_contract.py
    git commit -m "fix: make prediction units and features explicit"

## Task 6: Build the Research Console frontend shell and API layer

Files:

- Create frontend/src/api.js, frontend/src/components/StatusStrip.vue, ValuationForm.vue, EstimatePanel.vue, ModelEvidence.vue, AssistantPanel.vue, and HistoryLog.vue.
- Modify frontend/src/App.vue, frontend/src/assets/base.css, frontend/src/assets/main.css, frontend/src/main.js, and frontend/Readme.md.

- [ ] Step 1: Implement the shared API wrapper.

Use one requestJson(path, options) that prepends VITE_API_BASE_URL when set and throws an Error containing backend detail on non-2xx responses. Export getModelCard, getModelHealth, getMetrics, predictVehicle, getHistory, and askAssistant. The prediction wrapper uses POST /api/predict with JSON and assistant uses POST /api/assistant with { message }. Keep loading/error state in App.vue and pass data down as props.

- [ ] Step 2: Replace App.vue with the approved page structure.

Use a semantic shell with a header, navigation tabs for valuation, evidence, assistant, and history, and a main content area. On mount load model card, health, metrics, and history in parallel. Render skeleton/loading and retryable error states. Use model-card data for currency, units, labels, categories, feature descriptions, and source-driven examples.

- [ ] Step 3: Implement the full valuation form.

ValuationForm.vue submits brand, model, city, mileage, year, month, gearbox, emission, fuel_type, displacement, seats, owner_count, vehicle_type, color, and accident_history. Brands, cities, and models use API options plus free text. The mileage label includes km; price is never labeled RMB or 万元; accident history says the public source does not provide it. Disable submit while loading and validate before network submission.

- [ ] Step 4: Implement evidence and workflow components.

StatusStrip shows gate, R2, MAE, 10% accuracy, and sample count. EstimatePanel formats price/range through Intl.NumberFormat using currency and warns before a failed-gate estimate. ModelEvidence renders a Chart.js bar chart comparing model RMSE with baseline RMSE, then source, hash, split, feature version, and limitations. AssistantPanel renders answer, citations, structured estimate, and disabled state. HistoryLog renders loading, empty, and populated states with currency and model version.

- [ ] Step 5: Replace starter CSS with the A-style visual system.

Use a light neutral canvas, near-black text, green status accent, warm red gate-fail accent, 1px borders, 6px or smaller radii, 8/12/16/24 spacing, and a two-column desktop grid collapsing below 860px. Add stable control dimensions, visible focus states, aria-live regions, and no decorative gradient blobs or marketing hero copy. Remove the Vite starter #app max-width/grid rules.

- [ ] Step 6: Build and commit the frontend.

    Push-Location frontend
    npm run build
    Pop-Location
    git add frontend/src frontend/Readme.md
    git commit -m "feat: redesign frontend as research console"

Expected: Vite build completes without malformed template errors.

## Task 7: Verify the complete workflow and publish the upgrade

Files:

- Modify backend/README.md and frontend/Readme.md only for documentation gaps discovered during verification.

- [ ] Step 1: Run all backend tests with the project virtual environment.

    D:/car-valuation/venv/Scripts/python.exe -m unittest discover -s backend/tests -v

Expected: every test passes with sklearn 1.6.1 and CPU PyTorch; do not use system Python for artifact loading.

- [ ] Step 2: Rebuild and evaluate from the public source.

    Push-Location backend
    D:/car-valuation/venv/Scripts/python.exe -m scripts.train_model --download --allow-failed-publish
    D:/car-valuation/venv/Scripts/python.exe -m scripts.evaluate_model data/processed/normalized_training.csv
    Pop-Location

Compare evaluator output with backend/models/metrics.json. Confirm model_card.json contains source URL, hash, sample count, INR, km, seed, split counts, thresholds, and gate result. A failing gate must remain visible and cannot become a positive accuracy claim.

- [ ] Step 3: Start services and run browser smoke checks.

Start FastAPI from backend and Vite from frontend using an available port. Use browser automation to verify desktop and mobile widths: status strip, form submit, prediction result, gate warning, evidence chart, history, assistant disabled state, no console errors, no horizontal overflow, and no mojibake in visible text. Save only ignored screenshots for inspection.

- [ ] Step 4: Run final repository checks.

    git diff --check
    git status --short
    git ls-files backend/data/raw backend/data/processed frontend/node_modules backend/.env

Expected: no whitespace errors; only intended source, tests, docs, and reviewed JSON artifacts are tracked; raw CSV, local DB, secrets, node modules, and companion output are absent.

- [ ] Step 5: Commit and push the verified upgrade.

    git add backend frontend docs/superpowers/plans/2026-07-22-car-valuation-retrain-redesign.md
    git commit -m "feat: upgrade valuation project for research demo"
    git push origin main

Expected: GitHub main contains the implementation commits and the final working tree is clean.
