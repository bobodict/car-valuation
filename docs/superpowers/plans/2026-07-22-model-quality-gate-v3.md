# Model Quality Gate v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe CatBoost-led model competition pipeline and publish a backward-compatible vehicle valuation artifact only when the unchanged holdout quality gate passes.

**Architecture:** Normalize the additional public dataset fields, derive stable features in one shared module, seal a stratified outer test set, and rank CatBoost, ExtraTrees, and MLP configurations with five-fold development-set cross-validation. Publish the winning model through a manifest-driven runtime, then expose its leaderboard and subgroup diagnostics through the existing FastAPI and Vue Research Console contracts.

**Tech Stack:** Python 3.11, pandas, scikit-learn 1.6.1, CatBoost 1.2.x, PyTorch, FastAPI, Pydantic v2, Vue 3, Chart.js, Node test runner, unittest

---

## File Map

- Create `backend/services/feature_engineering.py`: canonical v3 feature names, physical-unit parsing helpers, derived features, and target transforms.
- Create `backend/services/split_service.py`: deterministic stratified outer split and five-fold manifest.
- Create `backend/services/model_competition.py`: candidate definitions, fold training, metrics, ranking, winner refit, and model serialization.
- Create `backend/services/model_runtime.py`: manifest-driven CatBoost/sklearn/PyTorch loading and prediction.
- Modify `backend/services/public_dataset_adapter.py`: map all useful `car details v4.csv` fields into the normalized v3 contract.
- Modify `backend/services/dataset_contract.py`: validate the expanded normalized dataset and physical ranges.
- Modify `backend/scripts/train_model.py`: orchestrate download, split, competition, report generation, smoke loading, gate, and atomic publication.
- Modify `backend/scripts/evaluate_model.py`: evaluate the manifest-selected runtime against the recorded outer test set.
- Modify `backend/predict_service.py`: retain `predict_price_one()` as a thin compatibility wrapper over `ModelRuntime`.
- Modify `backend/config.py`: expose manifest and experiment paths without hard-coding `price_mlp.pt`.
- Modify `backend/schemas.py`, `backend/services/model_service.py`, `backend/services/model_metadata.py`, `backend/services/metrics_service.py`, and `backend/services/model_quality_service.py`: support optional v3 inputs and evidence metadata.
- Modify `frontend/src/components/ValuationForm.vue`: add an unframed expandable technical-parameters section.
- Modify `frontend/src/components/ModelEvidence.vue` and `frontend/src/components/StatusStrip.vue`: show winner, cross-validation, holdout, and subgroup evidence.
- Create `frontend/src/modelEvidence.js` and `frontend/src/modelEvidence.test.js`: testable evidence-view transformations.
- Modify `frontend/src/assets/main.css`: responsive technical-input and evidence-table styles.
- Modify `README.md`, `backend/README.md`, and `backend/data/README.md`: document v3 reproducibility and limitations.

### Task 1: Expand the Public Dataset Adapter

**Files:**
- Modify: `backend/services/public_dataset_adapter.py`
- Modify: `backend/services/dataset_contract.py`
- Modify: `backend/tests/test_public_dataset_adapter.py`
- Modify: `backend/tests/test_model_quality_contract.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add failing parser and v3-column tests**

Add assertions for source fields `Seller Type`, `Max Power`, `Max Torque`, `Drivetrain`, `Length`, `Width`, `Height`, and `Fuel Tank Capacity`:

```python
from services.public_dataset_adapter import parse_power, parse_torque

def test_parse_power_and_torque_extract_value_and_rpm(self):
    self.assertEqual(parse_power("87 bhp @ 6000 rpm"), (87.0, 6000.0))
    self.assertEqual(parse_torque("109 Nm @ 4500 rpm"), (109.0, 4500.0))
    self.assertTrue(all(pd.isna(value) for value in parse_power("not supplied")))

def test_adapter_maps_v3_source_fields(self):
    result = adapt_car_details_v4(make_raw_fixture())
    self.assertEqual(result.loc[0, "seller_type"], "Individual")
    self.assertEqual(result.loc[0, "drivetrain"], "FWD")
    self.assertEqual(result.loc[0, "max_power_bhp"], 87.0)
    self.assertEqual(result.loc[0, "power_rpm"], 6000.0)
    self.assertEqual(result.loc[0, "max_torque_nm"], 109.0)
    self.assertEqual(result.loc[0, "torque_rpm"], 4500.0)
```

- [ ] **Step 2: Run the adapter tests and confirm RED**

Run from `backend`:

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_public_dataset_adapter tests.test_model_quality_contract -v
```

Expected: import failure for `parse_power` and missing v3 columns.

- [ ] **Step 3: Implement unit parsers and v3 mapping**

Use one helper for value/RPM extraction and explicit unit conversion:

```python
def _parse_measurement(value, conversions):
    if value is None or pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip().lower()
    value_match = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z-]+)", text)
    rpm_match = re.search(r"@\s*(\d+(?:\.\d+)?)\s*rpm", text)
    if not value_match or value_match.group(2) not in conversions:
        return np.nan, np.nan
    amount = float(value_match.group(1)) * conversions[value_match.group(2)]
    rpm = float(rpm_match.group(1)) if rpm_match else np.nan
    return amount, rpm

def parse_power(value):
    return _parse_measurement(value, {"bhp": 1.0, "ps": 0.98632, "kw": 1.34102})

def parse_torque(value):
    return _parse_measurement(value, {"nm": 1.0, "kgm": 9.80665, "kg-m": 9.80665})
```

Expand `REQUIRED_DATASET_COLUMNS` with the ten normalized v3 source fields and keep `vehicle_type`/`accident_history` for API compatibility.
Validate non-null physical values against fixed bounds: power `(0, 2000]` bhp,
RPM `(0, 25000]`, torque `(0, 10000]` Nm, length `[1000, 10000]` mm,
width `[1000, 5000]` mm, height `[500, 5000]` mm, and fuel tank
`(0, 1000]` liters. Missing values remain valid and are never replaced with zero.

- [ ] **Step 4: Add CatBoost dependency**

Append the bounded dependency:

```text
catboost>=1.2.8,<2.0
```

Install it with:

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: `Successfully installed catboost` or an already-satisfied message.

- [ ] **Step 5: Run tests and commit**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_public_dataset_adapter tests.test_model_quality_contract -v
git add services/public_dataset_adapter.py services/dataset_contract.py tests/test_public_dataset_adapter.py tests/test_model_quality_contract.py requirements.txt
git commit -m "feat: expand public vehicle data contract"
```

Expected: adapter tests pass.

### Task 2: Add Shared Feature Engineering

**Files:**
- Create: `backend/services/feature_engineering.py`
- Create: `backend/tests/test_feature_engineering.py`
- Modify: `backend/tests/test_baseline_contract.py`

- [ ] **Step 1: Write failing feature and target-transform tests**

```python
class FeatureEngineeringTests(unittest.TestCase):
    def test_enrich_features_builds_stable_v3_features(self):
        frame = pd.DataFrame([{
            "model": "Amaze 1.2 VX MT Petrol [2018-2020]",
            "year": 2020, "mileage": 60000, "displacement": 1.2,
            "max_power_bhp": 90, "length_mm": 4000, "width_mm": 1700,
        }])
        result = enrich_features(frame, collection_year=2026)
        self.assertEqual(result.loc[0, "model_family"], "Amaze")
        self.assertEqual(result.loc[0, "car_age"], 6)
        self.assertEqual(result.loc[0, "mileage_per_year"], 10000)
        self.assertEqual(result.loc[0, "power_per_liter"], 75)
        self.assertAlmostEqual(result.loc[0, "footprint_m2"], 6.8)

    def test_log_target_round_trip(self):
        values = np.array([49_000.0, 825_000.0, 35_000_000.0])
        np.testing.assert_allclose(inverse_target(transform_target(values)), values)
```

- [ ] **Step 2: Run and confirm RED**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_feature_engineering -v
```

Expected: `ModuleNotFoundError: services.feature_engineering`.

- [ ] **Step 3: Implement the single shared feature contract**

Define immutable feature lists and safe ratio helpers:

```python
NUMERIC_FEATURES = (
    "mileage", "displacement", "seats", "owner_count", "car_age",
    "max_power_bhp", "power_rpm", "max_torque_nm", "torque_rpm",
    "length_mm", "width_mm", "height_mm", "fuel_tank_liter",
    "mileage_per_year", "power_per_liter", "footprint_m2",
)
CATEGORICAL_FEATURES = (
    "brand", "model", "model_family", "city", "transmission",
    "fuel_type", "color", "seller_type", "drivetrain",
)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def _safe_ratio(numerator, denominator):
    result = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)

def transform_target(values):
    return np.log1p(np.asarray(values, dtype=float))

def inverse_target(values):
    return np.expm1(np.asarray(values, dtype=float))
```

`enrich_features()` must be the only place that derives features for both training and runtime.
It must normalize categorical values with `fillna("unknown").astype(str)` and
leave missing numeric values as `np.nan`, so all three model adapters consume the
same semantic input.

- [ ] **Step 4: Run tests and commit**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_feature_engineering tests.test_baseline_contract -v
git add services/feature_engineering.py tests/test_feature_engineering.py tests/test_baseline_contract.py
git commit -m "feat: add shared vehicle feature engineering"
```

Expected: all selected tests pass.

### Task 3: Seal the Outer Test Set and Development Folds

**Files:**
- Create: `backend/services/split_service.py`
- Create: `backend/tests/test_split_service.py`
- Modify: `backend/tests/test_training_pipeline.py`

- [ ] **Step 1: Write failing deterministic split tests**

```python
def test_split_manifest_is_deterministic_complete_and_disjoint(self):
    frame = make_price_fixture(200)
    first = build_split_manifest(frame, seed=42)
    second = build_split_manifest(frame, seed=42)
    self.assertEqual(first, second)
    self.assertTrue(set(first["development"]).isdisjoint(first["test"]))
    self.assertEqual(set(first["development"]) | set(first["test"]), set(frame.index))
    self.assertEqual(len(first["folds"]), 5)

def test_each_fold_keeps_validation_out_of_training(self):
    manifest = build_split_manifest(make_price_fixture(200), seed=42)
    for fold in manifest["folds"]:
        self.assertTrue(set(fold["train"]).isdisjoint(fold["validation"]))
```

- [ ] **Step 2: Run and confirm RED**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_split_service -v
```

Expected: missing `split_service` module.

- [ ] **Step 3: Implement stratified outer split and folds**

Use quantile bins over `log1p(price)`, `train_test_split(..., test_size=0.15, stratify=bins)`, and `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` on development indices. Return JSON-safe integer lists and include `split_version`, `seed`, and stratification description.

```python
def _price_bins(prices, bins=10):
    ranked = pd.Series(np.log1p(prices), index=prices.index).rank(method="first")
    return pd.qcut(ranked, q=min(bins, len(prices) // 5), labels=False, duplicates="drop")
```

- [ ] **Step 4: Run tests and commit**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_split_service tests.test_training_pipeline -v
git add services/split_service.py tests/test_split_service.py tests/test_training_pipeline.py
git commit -m "feat: add leakage-safe model split manifest"
```

Expected: split tests pass with five folds.

### Task 4: Add Metrics and Winner Ranking

**Files:**
- Create: `backend/services/model_competition.py`
- Create: `backend/tests/test_model_competition.py`
- Modify: `backend/tests/test_training_pipeline.py`

- [ ] **Step 1: Write failing metric and ranking tests**

```python
def test_metrics_include_relative_and_log_errors(self):
    metrics = calculate_metrics(
        actual=np.array([100.0, 200.0]),
        predicted=np.array([110.0, 180.0]),
        baseline=np.array([150.0, 150.0]),
    )
    self.assertEqual(metrics["acc_10"], 1.0)
    self.assertIn("acc_20", metrics)
    self.assertIn("median_ape", metrics)
    self.assertIn("rmsle", metrics)

def test_rank_candidates_ignores_test_metrics(self):
    winner = rank_candidates([
        {"name": "cat", "cv": {"acc_10_mean": .55, "median_ape_mean": .12, "r2_mean": .8}, "complexity": 2, "test_metrics": {"acc_10": 0}},
        {"name": "tree", "cv": {"acc_10_mean": .40, "median_ape_mean": .10, "r2_mean": .9}, "complexity": 1, "test_metrics": {"acc_10": 1}},
    ])
    self.assertEqual(winner["name"], "cat")
```

- [ ] **Step 2: Run and confirm RED**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_model_competition -v
```

Expected: missing competition functions.

- [ ] **Step 3: Implement metrics, bounded configs, and deterministic ranking**

Define `CandidateConfig(name, model_type, params, complexity)` and the following
checked-in bounded defaults:

```python
CATBOOST_CONFIGS = (
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3.0, "loss_function": "RMSE"},
    {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 5.0, "loss_function": "RMSE"},
    {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 8.0, "loss_function": "RMSE"},
    {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 10.0, "loss_function": "RMSE"},
)
EXTRA_TREES_CONFIGS = tuple(
    {"n_estimators": 600, "min_samples_leaf": leaf, "max_features": features, "n_jobs": -1}
    for leaf in (1, 2) for features in (0.7, 1.0)
)
MLP_CONFIGS = (
    {"hidden_dims": (128, 64), "dropout": 0.0, "learning_rate": 0.001},
    {"hidden_dims": (256, 128), "dropout": 0.1, "learning_rate": 0.001},
    {"hidden_dims": (128, 64, 32), "dropout": 0.1, "learning_rate": 0.0005},
)
```

CatBoost uses at most 1,500 iterations and patience 80; MLP uses at most 400
epochs and patience 40. Ranking must use only the `cv` object:

```python
def candidate_sort_key(result):
    cv = result["cv"]
    return (-cv["acc_10_mean"], cv["median_ape_mean"], -cv["r2_mean"], result["complexity"])

def rank_candidates(results):
    return sorted(results, key=candidate_sort_key)[0]
```

Implement the 0.01 `acc_10` tie window explicitly before applying the remaining tie-breakers.

- [ ] **Step 4: Run tests and commit**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_model_competition tests.test_training_pipeline -v
git add services/model_competition.py tests/test_model_competition.py tests/test_training_pipeline.py
git commit -m "feat: add auditable model competition metrics"
```

Expected: metric and ranking tests pass.

### Task 5: Implement Cross-Validated Candidate Training

**Files:**
- Modify: `backend/services/model_competition.py`
- Modify: `backend/tests/test_model_competition.py`
- Modify: `backend/scripts/train_model.py`

- [ ] **Step 1: Add failing fold-isolation and three-family tests**

Use tiny candidate configurations and patch fit functions to record received indices:

```python
def test_evaluate_candidate_uses_only_recorded_fold_rows(self):
    result = evaluate_candidate(frame, manifest, config, seed=42)
    self.assertEqual(len(result["fold_metrics"]), 5)
    for observed, expected in zip(result["observed_validation_indices"], manifest["folds"]):
        self.assertEqual(observed, expected["validation"])

def test_default_candidates_cover_all_model_families(self):
    self.assertEqual({item.model_type for item in default_candidates()}, {"catboost", "extra_trees", "mlp"})
```

- [ ] **Step 2: Run and confirm RED**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_model_competition -v
```

Expected: missing candidate evaluation behavior.

- [ ] **Step 3: Implement the three candidate adapters**

CatBoost must receive categorical column indexes, `loss_function="RMSE"`, fixed seeds, quiet logging, and validation-fold early stopping. ExtraTrees must use a fold-fitted `ColumnTransformer` with median numeric imputation and infrequent-safe one-hot encoding. MLP must reuse the existing early-stopped network through a focused adapter and train on transformed targets.

All adapters expose:

```python
class CandidateAdapter(Protocol):
    def fit(self, train_frame: pd.DataFrame, validation_frame: pd.DataFrame | None = None): ...
    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...
    def save(self, directory: Path) -> dict: ...
```

`evaluate_candidate()` enriches each fold independently, transforms only the target, computes fold metrics in INR, and returns mean/std summaries plus CatBoost best iterations.

- [ ] **Step 4: Run focused and full backend tests**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_model_competition -v
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all backend tests pass.

- [ ] **Step 5: Commit**

```powershell
git add services/model_competition.py tests/test_model_competition.py scripts/train_model.py
git commit -m "feat: train cross-validated model candidates"
```

### Task 6: Build v3 Reports and Atomic Publication

**Files:**
- Modify: `backend/scripts/train_model.py`
- Modify: `backend/scripts/evaluate_model.py`
- Modify: `backend/tests/test_training_pipeline.py`
- Modify: `backend/tests/test_evaluation_contract.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add failing artifact and failure-safety tests**

```python
def test_v3_artifact_set_contains_auditable_reports(self):
    paths = build_artifacts(fixture_experiment, output_dir)
    self.assertTrue({
        "model_manifest.json", "feature_config.json", "metrics.json",
        "leaderboard.json", "error_analysis.json", "model_card.json",
    }.issubset(path.name for path in paths))

def test_failed_holdout_gate_preserves_formal_artifact(self):
    published = publish_experiment(failed_dir, formal_dir)
    self.assertFalse(published)
    self.assertEqual((formal_dir / "marker.txt").read_text(), "old")
```

- [ ] **Step 2: Run and confirm RED**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_training_pipeline tests.test_evaluation_contract -v
```

Expected: v3 report functions are missing.

- [ ] **Step 3: Refactor `train_model.py` into an orchestrator**

The orchestration order is fixed:

```python
frame = load_dataset(dataset_path).drop_duplicates().reset_index(drop=True)
manifest = build_split_manifest(frame, seed)
leaderboard = run_competition(frame, manifest, default_candidates(), seed)
winner = select_winner(leaderboard)
fitted = refit_winner(frame.loc[manifest["development"]], winner, seed)
test_report = evaluate_holdout(fitted, frame.loc[manifest["test"]])
gate = assess_quality_gate(test_report["metrics"])
write_experiment_artifacts(...)
smoke_load_experiment(...)
published = publish_artifacts(..., gate["quality_gate"])
```

Generate subgroup reports for price quartiles, rare/common `model_family`, and seen/unseen full `model`. Add `backend/experiments/` to `.gitignore`.

- [ ] **Step 4: Update evaluator to use recorded v3 test indices**

`evaluate_model()` must load `model_manifest.json`, select only `split_indices.test`, call `ModelRuntime`, and use the development-set mean baseline recorded in metrics.

- [ ] **Step 5: Run tests and commit**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_training_pipeline tests.test_evaluation_contract -v
git add scripts/train_model.py scripts/evaluate_model.py tests/test_training_pipeline.py tests/test_evaluation_contract.py ..\.gitignore
git commit -m "feat: publish auditable v3 model experiments"
```

Expected: focused tests pass and failed gates preserve the old model directory.

### Task 7: Add the Manifest-Driven Runtime

**Files:**
- Create: `backend/services/model_runtime.py`
- Create: `backend/tests/test_model_runtime.py`
- Modify: `backend/predict_service.py`
- Modify: `backend/config.py`
- Modify: `backend/tests/test_prediction_artifact_scaling.py`

- [ ] **Step 1: Write failing new and legacy runtime tests**

```python
def test_runtime_loads_catboost_manifest_and_inverts_log_target(self):
    runtime = ModelRuntime.from_directory(catboost_fixture_dir)
    self.assertAlmostEqual(runtime.predict_one(make_vehicle()), 505000.0)

def test_runtime_loads_legacy_v2_mlp_without_manifest(self):
    runtime = ModelRuntime.from_directory(legacy_fixture_dir)
    self.assertAlmostEqual(runtime.predict_one(make_vehicle()), 200.0)

def test_runtime_rejects_non_finite_prediction(self):
    with self.assertRaisesRegex(ModelRuntimeError, "finite"):
        broken_runtime.predict_one(make_vehicle())
```

- [ ] **Step 2: Run and confirm RED**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_model_runtime -v
```

Expected: missing runtime module.

- [ ] **Step 3: Implement runtime loaders and compatibility wrapper**

```python
class ModelRuntime:
    @classmethod
    def from_directory(cls, models_dir: Path):
        manifest_path = models_dir / "model_manifest.json"
        if not manifest_path.exists():
            return LegacyTorchRuntime.from_directory(models_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaders = {"catboost": CatBoostRuntime, "extra_trees": SklearnRuntime, "mlp": TorchRuntime}
        try:
            return loaders[manifest["model_type"]].from_manifest(models_dir, manifest)
        except (KeyError, OSError, ValueError) as exc:
            raise ModelRuntimeError("model manifest or artifact is invalid") from exc
```

Make `predict_service.predict_price_one()` call a cached runtime and retain a cache-clear hook for tests and model reloads.

- [ ] **Step 4: Run runtime and prediction tests**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_model_runtime tests.test_prediction_artifact_scaling tests.test_prediction_contract -v
```

Expected: new and v2 artifacts both pass.

- [ ] **Step 5: Commit**

```powershell
git add services/model_runtime.py tests/test_model_runtime.py predict_service.py config.py tests/test_prediction_artifact_scaling.py
git commit -m "feat: load versioned valuation model runtimes"
```

### Task 8: Extend API Inputs and Evidence Metadata

**Files:**
- Modify: `backend/schemas.py`
- Modify: `backend/services/model_service.py`
- Modify: `backend/services/model_metadata.py`
- Modify: `backend/services/metrics_service.py`
- Modify: `backend/services/model_quality_service.py`
- Modify: `backend/main.py`
- Modify: `backend/tests/test_prediction_contract.py`
- Modify: `backend/tests/test_model_metadata.py`

- [ ] **Step 1: Write failing additive-contract tests**

Add optional physical inputs and new response metadata:

```python
request = PredictRequest(
    **base_request,
    seller_type="Individual", drivetrain="FWD", max_power_bhp=87,
    power_rpm=6000, max_torque_nm=109, torque_rpm=4500,
    length_mm=3990, width_mm=1680, height_mm=1505, fuel_tank_liter=35,
)
model_input = build_model_input(request)
self.assertEqual(model_input["max_power_bhp"], 87)
self.assertEqual(model_input["drivetrain"], "FWD")

result = model_card()
self.assertIn("model_type", result)
self.assertIn("leaderboard", result)
self.assertIn("error_analysis", result)
```

- [ ] **Step 2: Run and confirm RED**

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_prediction_contract tests.test_model_metadata -v
```

Expected: Pydantic rejects unknown expected fields or response metadata is absent.

- [ ] **Step 3: Implement backward-compatible optional fields**

Add optional categories `seller_type` and `drivetrain`, plus these exact Pydantic
bounds to `PredictRequest`: `max_power_bhp` `(0, 2000]`, `power_rpm` and
`torque_rpm` `(0, 25000]`, `max_torque_nm` `(0, 10000]`, `length_mm`
`[1000, 10000]`, `width_mm` `[1000, 5000]`, `height_mm` `[500, 5000]`, and
`fuel_tank_liter` `(0, 1000]`. Every field defaults to `None`, so old requests
continue to validate. Pass them through `build_model_input()`. Add `model_type`,
`feature_version`, `leaderboard`, and `error_analysis` to response schemas with
safe defaults for v2 artifacts. Accept both the legacy `train/validation/test`
split object and the v3 `development/test/folds` object in model-card validation.

Prediction responses include:

```python
{
    "model_version": metrics["model_version"],
    "model_type": metrics.get("model_type", "mlp"),
    "feature_version": metrics.get("feature_version", "2.0.0"),
    "quality_gate": metrics["quality_gate"],
}
```

Keep 422 for invalid inputs and translate `ModelRuntimeError` into the existing clear 503 path. Correct only user-facing strings in files touched by this task.

- [ ] **Step 4: Run full backend tests and commit**

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
git add schemas.py services/model_service.py services/model_metadata.py services/metrics_service.py services/model_quality_service.py main.py tests/test_prediction_contract.py tests/test_model_metadata.py
git commit -m "feat: expose v3 model inputs and evidence"
```

Expected: all backend tests pass.

### Task 9: Train and Gate the Real Public-Data Model

**Files:**
- Modify: `backend/models/model_manifest.json`
- Modify: `backend/models/feature_config.json`
- Modify: `backend/models/metrics.json`
- Modify: `backend/models/model_card.json`
- Create: `backend/models/leaderboard.json`
- Create: `backend/models/error_analysis.json`
- Create: `backend/models/price_model.cbm` or the winner's equivalent file
- Remove only if no longer referenced: `backend/models/price_mlp.pt`

- [ ] **Step 1: Run the full test suite before training**

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run the real competition without bypass flags**

```powershell
..\venv\Scripts\python.exe -m scripts.train_model --download
```

Expected: JSON containing the CV winner, `published: true`, and `quality_gate: pass`. The command must not offer `--allow-failed-publish` for v3 formal publication.

- [ ] **Step 3: Handle a failed real gate without contaminating the test set**

If Step 2 reports `quality_gate: fail`, stop publication and use `backend/experiments/<run-id>/leaderboard.json` plus fold-level error reports only. Invoke `systematic-debugging`; adjust feature parsing or the predeclared candidate set based on development-fold evidence, add a failing regression test, and rerun the entire competition with a new untouched outer split version only if the design itself changes. Never lower `min_acc_10` or select a model from test metrics.

- [ ] **Step 4: Independently evaluate the published artifact**

```powershell
..\venv\Scripts\python.exe -m scripts.evaluate_model data\processed\normalized_training.csv
```

Expected: `evaluation_scope` is `recorded_test`, `acc_10 >= 0.50`, `r2 > 0`, and `rmse < baseline_rmse`.

- [ ] **Step 5: Check artifact size, secrets, and commit**

```powershell
Get-ChildItem models | Sort-Object Length -Descending | Select-Object Name,Length
git status --short
git check-ignore data\raw\car-details-v4.csv data\processed\normalized_training.csv
git add models
git commit -m "feat: publish quality-gated valuation model v3"
```

Expected: no tracked model file exceeds 100 MB; raw and processed data remain ignored.

### Task 10: Upgrade the Research Console Evidence and Inputs

**Files:**
- Create: `frontend/src/modelEvidence.js`
- Create: `frontend/src/modelEvidence.test.js`
- Modify: `frontend/src/components/ValuationForm.vue`
- Modify: `frontend/src/components/ModelEvidence.vue`
- Modify: `frontend/src/components/StatusStrip.vue`
- Modify: `frontend/src/assets/main.css`
- Modify: `frontend/package.json`

- [ ] **Step 1: Write failing pure-JavaScript view-model tests**

```javascript
import { buildEvidenceRows, buildPriceBandRows } from './modelEvidence.js'

test('buildEvidenceRows separates CV and holdout metrics', () => {
  const rows = buildEvidenceRows(cardFixture)
  assert.deepEqual(rows[0], { label: '开发集 5 折', acc10: 0.56, scope: 'CV' })
  assert.deepEqual(rows[1], { label: '独立测试集', acc10: 0.52, scope: 'HOLDOUT' })
})

test('buildPriceBandRows keeps INR ranges and finite metrics', () => {
  assert.equal(buildPriceBandRows(cardFixture)[0].currency, 'INR')
})
```

- [ ] **Step 2: Add the test script and confirm RED**

Set `"test": "node --test src/*.test.js"` in `package.json`, then run:

```powershell
npm test
```

Expected: missing `modelEvidence.js`.

- [ ] **Step 3: Implement evidence transformations and UI**

Create defensive pure functions that return empty arrays for v2 metadata. Update the evidence panel to show the winner algorithm, CV versus holdout rows, unchanged gate threshold, and price-band diagnostics. Keep the existing RMSE comparison chart.

Add optional v3 technical inputs inside a native `<details class="technical-fields">` section. Before emitting, map each empty optional field with
`value === '' ? null : Number(value)`; omitted values therefore serialize as
`null`. Existing required fields and the default Honda example remain usable.
Use labels with units, selects for `seller_type`/`drivetrain`, and numeric inputs
for power, torque, RPM, dimensions, and tank capacity.

- [ ] **Step 4: Add responsive styles**

Use the existing neutral palette and border system. The technical section is unframed, grid columns collapse to one track below 720 px, tables scroll only inside their own evidence region, and no text overlaps at 360 px.

- [ ] **Step 5: Run tests/build and commit**

```powershell
npm test
npm run build
git add src/modelEvidence.js src/modelEvidence.test.js src/components/ValuationForm.vue src/components/ModelEvidence.vue src/components/StatusStrip.vue src/assets/main.css package.json package-lock.json
git commit -m "feat: show model competition evidence"
```

Expected: all frontend tests pass and Vite exits 0.

### Task 11: Documentation, Runtime Smoke Test, and GitHub Delivery

**Files:**
- Modify: `README.md`
- Modify: `backend/README.md`
- Modify: `backend/data/README.md`

- [ ] **Step 1: Update reproducibility documentation**

Document the v3 normalized columns, CatBoost dependency, outer-test sealing, five-fold ranking, quality gate, artifact list, optional API fields, and exact commands:

```powershell
cd backend
..\venv\Scripts\python.exe -m scripts.train_model --download
..\venv\Scripts\python.exe -m scripts.evaluate_model data\processed\normalized_training.csv
..\venv\Scripts\python.exe -m uvicorn main:app --reload
```

Keep the India/INR and no-calibrated-confidence limitations prominent.

- [ ] **Step 2: Run fresh backend and frontend verification**

```powershell
cd D:\car-valuation\backend
..\venv\Scripts\python.exe -m unittest discover -s tests -v
cd D:\car-valuation\frontend
npm test
npm run build
```

Expected: zero backend/frontend test failures and successful production build.

- [ ] **Step 3: Restart services and perform API smoke checks**

Start the backend and frontend on free local ports, then verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/model-health
Invoke-RestMethod http://127.0.0.1:8000/api/model-card
```

Expected: `quality_gate` is `pass`, `model_type` names the winner, and model-card evidence is non-empty. Submit the default Honda prediction and confirm a finite positive INR price with the same model version.

- [ ] **Step 4: Run browser QA**

Use the `agent-browser` skill to inspect 1440x900 and 390x844 viewports. Verify valuation submission, technical field expansion, model evidence, history, loading/error states, no horizontal page overflow, and no text overlap. Save screenshots under an ignored temporary directory.

- [ ] **Step 5: Verify repository scope and commit docs**

```powershell
git status --short
git ls-files | Select-String -Pattern '^car-valuation-submit/'
git diff --check
git add README.md backend/README.md backend/data/README.md
git commit -m "docs: document model quality gate v3"
```

Expected: no tracked submit path, no whitespace errors, and only intended files changed.

- [ ] **Step 6: Push and verify GitHub**

```powershell
git push origin main
git status --short --branch
git ls-remote origin refs/heads/main
```

Expected: local `main`, `origin/main`, and the remote SHA match; the worktree is clean.
