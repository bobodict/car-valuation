from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from services.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    enrich_features,
)
from services.model_competition import (
    CandidateConfig,
    ExtraTreesCandidateAdapter,
    MLPRegressor,
    _build_fold_preprocessor,
)
from services import model_runtime as model_runtime_module
from services.model_runtime import LegacyTorchRuntime, ModelRuntime, ModelRuntimeError


EXPECTED_PRICE = 505_000.0
LOG_EXPECTED_PRICE = math.log1p(EXPECTED_PRICE)


class SpoofedFittedComponent:
    def __init__(self, feature_width):
        self.fitted_ = False
        self.n_features_in_ = feature_width

    def fit(self, features, targets=None):
        return self

    def predict(self, features):
        return np.zeros(len(features), dtype=float)


def raw_vehicles(count=4):
    rows = []
    for index in range(count):
        rows.append(
            {
                "brand": "Honda" if index % 2 == 0 else "Toyota",
                "model": "Amaze S" if index % 2 == 0 else "Etios G",
                "year": 2018 + index % 2,
                "mileage": 50_000.0 + index * 1_000,
                "city": "Pune",
                "transmission": "Manual",
                "fuel_type": "Petrol",
                "color": "Grey",
                "seller_type": "Dealer",
                "drivetrain": "FWD",
                "displacement": 1.2,
                "seats": 5,
                "owner_count": 1,
                "max_power_bhp": 90.0,
                "power_rpm": 6_000.0,
                "max_torque_nm": 110.0,
                "torque_rpm": 4_000.0,
                "length_mm": 3_995.0,
                "width_mm": 1_695.0,
                "height_mm": 1_500.0,
                "fuel_tank_liter": 35.0,
            }
        )
    return pd.DataFrame(rows)


def canonical_features(count=4):
    return enrich_features(raw_vehicles(count), 2026).loc[:, MODEL_FEATURES]


def write_v3_contract(root, model_type, artifacts):
    manifest = {
        "artifact_version": "3.0.0",
        "model_contract_version": "3.0.0",
        "feature_version": "3.0.0",
        "model_version": "v3-runtime-test",
        "model_type": model_type,
        "model_artifacts": artifacts,
        "target_transform": "log1p",
        "collection_year": 2026,
    }
    feature_config = {
        "artifact_version": "3.0.0",
        "feature_version": "3.0.0",
        "model_version": "v3-runtime-test",
        "feature_cols": list(MODEL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_transform": "log1p",
        "collection_year": 2026,
    }
    (root / "model_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "feature_config.json").write_text(
        json.dumps(feature_config), encoding="utf-8"
    )
    return manifest, feature_config


def write_extra_trees_artifacts(root, log_target=LOG_EXPECTED_PRICE):
    features = canonical_features()
    preprocessor = _build_fold_preprocessor(scale_numeric=False)
    transformed = preprocessor.fit_transform(features)
    model = ExtraTreesRegressor(n_estimators=2, random_state=42)
    model.fit(transformed, np.full(len(features), log_target))
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "target_transform": "log1p",
        },
        root / "extra_trees.joblib",
    )
    return write_v3_contract(
        root, "extra_trees", {"bundle": "extra_trees.joblib"}
    )


def write_mlp_artifacts(root):
    features = canonical_features()
    preprocessor = _build_fold_preprocessor(scale_numeric=True)
    transformed = preprocessor.fit_transform(features)
    model = MLPRegressor(transformed.shape[1], hidden_dims=(2,), dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(LOG_EXPECTED_PRICE)
    joblib.dump(preprocessor, root / "mlp_preprocessor.joblib")
    torch.save(
        {
            "input_dim": transformed.shape[1],
            "hidden_dims": [2],
            "dropout": 0.0,
            "model_state": model.state_dict(),
            "target_transform": "log1p",
        },
        root / "mlp.pt",
    )
    return write_v3_contract(
        root,
        "mlp",
        {"preprocessor": "mlp_preprocessor.joblib", "model": "mlp.pt"},
    )


def write_catboost_artifacts(root):
    features = canonical_features()
    model = CatBoostRegressor(
        iterations=1,
        depth=1,
        learning_rate=1.0,
        loss_function="RMSE",
        allow_const_label=True,
        allow_writing_files=False,
        verbose=False,
        random_seed=42,
        cat_features=[MODEL_FEATURES.index(name) for name in CATEGORICAL_FEATURES],
    )
    model.fit(features, np.full(len(features), LOG_EXPECTED_PRICE), verbose=False)
    model.save_model(root / "catboost.cbm")
    return write_v3_contract(root, "catboost", {"model": "catboost.cbm"})


def write_legacy_artifacts(
    root,
    numeric_features=("mileage",),
    categorical_features=("brand",),
):
    feature_cols = [*numeric_features, *categorical_features]
    feature_config = {
        "artifact_version": "2.0.0",
        "feature_cols": feature_cols,
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
    }
    (root / "feature_config.json").write_text(
        json.dumps(feature_config), encoding="utf-8"
    )
    training_values = {
        **{name: [1.0, 2.0] for name in numeric_features},
        **{name: ["Honda", "Toyota"] for name in categorical_features},
    }
    training_frame = pd.DataFrame(training_values)
    transformers = []
    if numeric_features:
        transformers.append(
            ("numeric", StandardScaler(), list(numeric_features))
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            )
        )
    preprocessor = ColumnTransformer(transformers)
    transformed = preprocessor.fit_transform(training_frame)
    joblib.dump(preprocessor, root / "preprocess.joblib")
    model = MLPRegressor(transformed.shape[1], hidden_dims=(2,), dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(2.0)
    torch.save(
        {
            "input_dim": transformed.shape[1],
            "hidden_dims": [2],
            "dropout": 0.0,
            "model_state": model.state_dict(),
            "target_mean": 100.0,
            "target_std": 50.0,
        },
        root / "price_mlp.pt",
    )


def replace_with_symlink(test_case, link_path, target_path):
    link_path.unlink()
    try:
        os.symlink(target_path, link_path)
    except OSError as exc:
        test_case.skipTest(f"symlink creation is unavailable: {exc}")


def write_cache_v3_publication(
    root,
    model_version="v3-cache-first",
    artifact_bytes=b"artifact-v1",
    feature_marker="feature-v1",
):
    manifest = {
        "model_version": model_version,
        "model_type": "extra_trees",
        "model_artifacts": {"bundle": "winner.joblib"},
    }
    (root / "model_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    (root / "feature_config.json").write_text(
        json.dumps({"marker": feature_marker}, sort_keys=True), encoding="utf-8"
    )
    (root / ".publication-generation.json").write_text(
        json.dumps({"generation": "generation-v1"}, sort_keys=True), encoding="utf-8"
    )
    (root / "winner.joblib").write_bytes(artifact_bytes)


def write_cache_v3_formal_reports(root, marker="report-v1"):
    write_cache_v3_publication(root)
    for filename in (
        "metrics.json",
        "leaderboard.json",
        "error_analysis.json",
        "model_card.json",
    ):
        (root / filename).write_text(
            json.dumps({"marker": marker}, sort_keys=True), encoding="utf-8"
        )


def write_cache_legacy_publication(root, model_bytes=b"legacy-model-v1"):
    manifest_path = root / "model_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    v3_artifact = root / "winner.joblib"
    if v3_artifact.exists():
        v3_artifact.unlink()
    (root / "feature_config.json").write_text(
        json.dumps({"artifact_version": "2.0.0"}), encoding="utf-8"
    )
    (root / "preprocess.joblib").write_bytes(b"legacy-preprocessor")
    (root / "price_mlp.pt").write_bytes(model_bytes)


class ModelRuntimeLoadingTests(unittest.TestCase):
    def test_checked_in_legacy_bundle_loads_without_v3_manifest(self):
        source = Path(__file__).parents[1] / "models"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models"
            shutil.copytree(source, root)
            (root / "model_manifest.json").rename(root / "model_manifest.json.disabled")

            runtime = ModelRuntime.from_directory(root)

        self.assertIsInstance(runtime._implementation, LegacyTorchRuntime)

    def test_loads_bundle_written_by_production_extra_trees_adapter(self):
        source = raw_vehicles(4)
        training_frame = enrich_features(source, 2026).loc[:, MODEL_FEATURES]
        training_frame["price"] = [400_000.0, 450_000.0, 500_000.0, 550_000.0]
        adapter = ExtraTreesCandidateAdapter(
            CandidateConfig(
                "runtime-integration",
                "extra_trees",
                {
                    "n_estimators": 2,
                    "min_samples_leaf": 1,
                    "max_features": 1.0,
                    "n_jobs": 1,
                },
                1,
            ),
            seed=42,
        ).fit(training_frame)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = adapter.save(root)
            write_v3_contract(root, "extra_trees", metadata["artifacts"])
            runtime = ModelRuntime.from_directory(root)

            actual = runtime.predict(source)
            expected = adapter.predict(training_frame.loc[:, MODEL_FEATURES])

            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_catboost_manifest_loads_and_inverts_log_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_catboost_artifacts(root)

            runtime = ModelRuntime.from_directory(root)
            predictions = runtime.predict(raw_vehicles(2))

            self.assertEqual(predictions.shape, (2,))
            np.testing.assert_allclose(
                predictions, [EXPECTED_PRICE, EXPECTED_PRICE], rtol=1e-5
            )

    def test_extra_trees_bundle_loads_and_predicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_extra_trees_artifacts(root)

            runtime = ModelRuntime.from_directory(root)

            self.assertAlmostEqual(
                runtime.predict_one(raw_vehicles(1).iloc[0].to_dict()),
                EXPECTED_PRICE,
                places=4,
            )

    def test_v3_mlp_artifacts_load_on_cpu_and_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mlp_artifacts(root)

            runtime = ModelRuntime.from_directory(root)
            predictions = runtime.predict(raw_vehicles(2))

            np.testing.assert_allclose(
                predictions, [EXPECTED_PRICE, EXPECTED_PRICE], rtol=1e-5
            )

    def test_missing_manifest_uses_legacy_target_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_legacy_artifacts(root)

            runtime = ModelRuntime.from_directory(root)

            self.assertEqual(
                runtime.predict_one({"mileage": 10_000, "brand": "Honda"}),
                200.0,
            )

    def test_legacy_all_numeric_and_all_categorical_partitions_predict(self):
        cases = (
            (("mileage",), (), {"mileage": 10_000}),
            ((), ("brand",), {"brand": "Honda"}),
        )
        for numeric, categorical, vehicle in cases:
            with self.subTest(
                numeric=numeric,
                categorical=categorical,
            ):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write_legacy_artifacts(
                        root,
                        numeric_features=numeric,
                        categorical_features=categorical,
                    )

                    runtime = ModelRuntime.from_directory(root)

                    self.assertEqual(runtime.predict_one(vehicle), 200.0)

    def test_checked_in_legacy_artifacts_load_and_predict(self):
        models_dir = Path(__file__).resolve().parents[1] / "models"

        runtime = ModelRuntime.from_directory(models_dir)
        prediction = runtime.predict_one(raw_vehicles(1).iloc[0].to_dict())

        self.assertTrue(math.isfinite(prediction))

    def test_predict_one_requires_a_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_legacy_artifacts(root)
            runtime = ModelRuntime.from_directory(root)

            with self.assertRaisesRegex(ModelRuntimeError, "mapping"):
                runtime.predict_one([{"mileage": 10_000}])


class ModelRuntimeValidationTests(unittest.TestCase):
    def test_predict_does_not_rewrap_runtime_error_or_memory_error(self):
        implementation = Mock()
        runtime = ModelRuntime(implementation)
        errors = (ModelRuntimeError("existing"), MemoryError("memory"))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                implementation.predict.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    runtime.predict(pd.DataFrame([{"mileage": 1.0}]))
                self.assertIs(raised.exception, error)

    def test_load_does_not_rewrap_runtime_error_or_memory_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_legacy_artifacts(root)
            errors = (ModelRuntimeError("existing"), MemoryError("memory"))
            for error in errors:
                with self.subTest(error=type(error).__name__):
                    with patch.object(
                        model_runtime_module.LegacyTorchRuntime,
                        "from_directory",
                        side_effect=error,
                    ):
                        with self.assertRaises(type(error)) as raised:
                            ModelRuntime.from_directory(root)
                    self.assertIs(raised.exception, error)

    def test_rejects_finite_log_prediction_above_safe_downstream_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_extra_trees_artifacts(root, log_target=1_000_000.0)
            runtime = ModelRuntime.from_directory(root)

            with self.assertRaisesRegex(ModelRuntimeError, "safe|limit|large"):
                runtime.predict(raw_vehicles(1))

    def test_safe_log_boundary_leaves_two_times_float_headroom(self):
        safe_price = np.finfo(float).max / 4.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_extra_trees_artifacts(
                root, log_target=float(np.log1p(safe_price))
            )
            runtime = ModelRuntime.from_directory(root)

            prediction = runtime.predict(raw_vehicles(1))[0]

            self.assertTrue(math.isfinite(prediction * 2.0))

    def test_rejects_external_symlinked_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "models"
            root.mkdir()
            write_extra_trees_artifacts(root)
            external_manifest = parent / "external-manifest.json"
            external_manifest.write_bytes((root / "model_manifest.json").read_bytes())
            replace_with_symlink(
                self, root / "model_manifest.json", external_manifest
            )

            with self.assertRaisesRegex(ModelRuntimeError, "symlink|inside"):
                ModelRuntime.from_directory(root)

    def test_rejects_external_symlinked_v3_feature_config(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "models"
            root.mkdir()
            write_extra_trees_artifacts(root)
            external_config = parent / "external-feature-config.json"
            external_config.write_bytes((root / "feature_config.json").read_bytes())
            replace_with_symlink(
                self, root / "feature_config.json", external_config
            )

            with self.assertRaisesRegex(ModelRuntimeError, "symlink|inside"):
                ModelRuntime.from_directory(root)

    def test_rejects_external_legacy_preprocessor_before_joblib_load(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "models"
            root.mkdir()
            write_legacy_artifacts(root)
            external_preprocessor = parent / "external-preprocess.joblib"
            external_preprocessor.write_bytes(
                (root / "preprocess.joblib").read_bytes()
            )
            replace_with_symlink(
                self, root / "preprocess.joblib", external_preprocessor
            )

            with patch.object(
                model_runtime_module.joblib,
                "load",
                wraps=joblib.load,
            ) as load:
                with self.assertRaisesRegex(ModelRuntimeError, "symlink|inside"):
                    ModelRuntime.from_directory(root)

            load.assert_not_called()

    def test_audits_legacy_symlink_components_before_joblib_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_legacy_artifacts(root)
            original_is_symlink = Path.is_symlink

            def mark_preprocessor_as_symlink(path):
                return path.name == "preprocess.joblib" or original_is_symlink(path)

            with (
                patch.object(
                    Path,
                    "is_symlink",
                    autospec=True,
                    side_effect=mark_preprocessor_as_symlink,
                ),
                patch.object(
                    model_runtime_module.joblib,
                    "load",
                    wraps=joblib.load,
                ) as load,
            ):
                with self.assertRaisesRegex(ModelRuntimeError, "symlink"):
                    ModelRuntime.from_directory(root)

            load.assert_not_called()

    def test_rejects_nonpositive_legacy_target_std(self):
        for target_std in (0.0, -1.0):
            with self.subTest(target_std=target_std):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write_legacy_artifacts(root)
                    checkpoint = torch.load(
                        root / "price_mlp.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    checkpoint["target_std"] = target_std
                    torch.save(checkpoint, root / "price_mlp.pt")

                    with self.assertRaisesRegex(ModelRuntimeError, "target_std|positive"):
                        ModelRuntime.from_directory(root)

    def test_rejects_invalid_legacy_feature_partitions(self):
        mutations = (
            (
                "duplicate",
                lambda config: config.update(
                    feature_cols=[*config["feature_cols"], "mileage"]
                ),
            ),
            (
                "overlap",
                lambda config: config.update(
                    categorical_features=[
                        *config["categorical_features"],
                        "mileage",
                    ]
                ),
            ),
            (
                "omission",
                lambda config: config.update(feature_cols=["mileage"]),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write_legacy_artifacts(root)
                    config_path = root / "feature_config.json"
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    mutate(config)
                    config_path.write_text(json.dumps(config), encoding="utf-8")

                    with self.assertRaisesRegex(
                        ModelRuntimeError, "unique|overlap|compose|feature"
                    ):
                        ModelRuntime.from_directory(root)

    def test_rejects_legacy_preprocessor_checkpoint_width_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_legacy_artifacts(root)
            checkpoint = torch.load(
                root / "price_mlp.pt", map_location="cpu", weights_only=True
            )
            mismatched = MLPRegressor(
                checkpoint["input_dim"] + 1,
                tuple(checkpoint["hidden_dims"]),
                checkpoint["dropout"],
            )
            checkpoint["input_dim"] += 1
            checkpoint["model_state"] = mismatched.state_dict()
            torch.save(checkpoint, root / "price_mlp.pt")

            with self.assertRaisesRegex(ModelRuntimeError, "contract|width"):
                ModelRuntime.from_directory(root)

    def test_rejects_wrong_legacy_preprocessor_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_legacy_artifacts(root)
            wrong_preprocessor = StandardScaler().fit(
                pd.DataFrame({"mileage": [1.0, 2.0]})
            )
            joblib.dump(wrong_preprocessor, root / "preprocess.joblib")

            with self.assertRaisesRegex(ModelRuntimeError, "legacy|preprocessor|type"):
                ModelRuntime.from_directory(root)

    def test_rejects_malformed_or_unknown_manifest_without_legacy_fallback(self):
        invalid_payloads = ("not-json", "[]")
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write_legacy_artifacts(root)
                    (root / "model_manifest.json").write_text(
                        payload, encoding="utf-8"
                    )
                    with self.assertRaises(ModelRuntimeError):
                        ModelRuntime.from_directory(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unused.bin").write_bytes(b"unused")
            write_v3_contract(root, "random_forest", {"model": "unused.bin"})
            with self.assertRaisesRegex(ModelRuntimeError, "model_type"):
                ModelRuntime.from_directory(root)

    def test_rejects_missing_fields_and_contract_mismatches(self):
        mutations = (
            ("missing field", lambda manifest, config: manifest.pop("model_type")),
            (
                "artifact_version",
                lambda manifest, config: manifest.update(artifact_version="2.0.0"),
            ),
            (
                "model_contract_version",
                lambda manifest, config: manifest.update(
                    model_contract_version="4.0.0"
                ),
            ),
            (
                "feature_version",
                lambda manifest, config: config.update(feature_version="2.0.0"),
            ),
            (
                "model_version",
                lambda manifest, config: manifest.update(model_version="mlp-old"),
            ),
            (
                "target_transform",
                lambda manifest, config: config.update(target_transform="identity"),
            ),
            (
                "collection_year",
                lambda manifest, config: manifest.update(collection_year=1979),
            ),
            (
                "feature_cols",
                lambda manifest, config: config.update(
                    feature_cols=list(reversed(MODEL_FEATURES))
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest, config = write_extra_trees_artifacts(root)
                    mutate(manifest, config)
                    (root / "model_manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    (root / "feature_config.json").write_text(
                        json.dumps(config), encoding="utf-8"
                    )
                    with self.assertRaises(ModelRuntimeError):
                        ModelRuntime.from_directory(root)

    def test_rejects_artifact_traversal_missing_and_corrupt_artifacts(self):
        cases = (
            ("../outside.joblib", None),
            ("missing.joblib", None),
            ("corrupt.joblib", b"not-a-joblib"),
        )
        for artifact_path, contents in cases:
            with self.subTest(artifact_path=artifact_path):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    if contents is not None:
                        (root / artifact_path).write_bytes(contents)
                    write_v3_contract(
                        root, "extra_trees", {"bundle": artifact_path}
                    )
                    with self.assertRaises(ModelRuntimeError):
                        ModelRuntime.from_directory(root)

    def test_rejects_unfitted_or_structurally_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = canonical_features()
            preprocessor = _build_fold_preprocessor(scale_numeric=False)
            transformed = preprocessor.fit_transform(features)
            joblib.dump(
                {
                    "preprocessor": preprocessor,
                    "model": SpoofedFittedComponent(transformed.shape[1]),
                    "target_transform": "log1p",
                },
                root / "extra_trees.joblib",
            )
            write_v3_contract(
                root, "extra_trees", {"bundle": "extra_trees.joblib"}
            )

            with self.assertRaises(ModelRuntimeError):
                ModelRuntime.from_directory(root)

    def test_rejects_fitted_wrong_extra_trees_model_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = canonical_features()
            preprocessor = _build_fold_preprocessor(scale_numeric=False)
            transformed = preprocessor.fit_transform(features)
            model = RandomForestRegressor(n_estimators=1, random_state=42)
            model.fit(transformed, np.full(len(features), LOG_EXPECTED_PRICE))
            joblib.dump(
                {
                    "preprocessor": preprocessor,
                    "model": model,
                    "target_transform": "log1p",
                },
                root / "extra_trees.joblib",
            )
            write_v3_contract(
                root, "extra_trees", {"bundle": "extra_trees.joblib"}
            )

            with self.assertRaisesRegex(ModelRuntimeError, "ExtraTrees|family|type"):
                ModelRuntime.from_directory(root)

    def test_rejects_wrong_v3_mlp_preprocessor_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mlp_artifacts(root)
            wrong_preprocessor = StandardScaler().fit(
                pd.DataFrame({"mileage": [1.0, 2.0]})
            )
            joblib.dump(wrong_preprocessor, root / "mlp_preprocessor.joblib")

            with self.assertRaisesRegex(ModelRuntimeError, "MLP|preprocessor|type"):
                ModelRuntime.from_directory(root)

    def test_rejects_mlp_preprocessor_and_checkpoint_width_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mlp_artifacts(root)
            checkpoint = torch.load(
                root / "mlp.pt", map_location="cpu", weights_only=True
            )
            mismatched = MLPRegressor(
                checkpoint["input_dim"] + 1,
                tuple(checkpoint["hidden_dims"]),
                checkpoint["dropout"],
            )
            checkpoint["input_dim"] += 1
            checkpoint["model_state"] = mismatched.state_dict()
            torch.save(checkpoint, root / "mlp.pt")

            with self.assertRaisesRegex(ModelRuntimeError, "contract|width"):
                ModelRuntime.from_directory(root)

    def test_rejects_extra_trees_preprocessor_and_model_width_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = canonical_features()
            preprocessor = _build_fold_preprocessor(scale_numeric=False)
            transformed = preprocessor.fit_transform(features)
            model = ExtraTreesRegressor(n_estimators=1, random_state=42)
            model.fit(
                transformed[:, :-1],
                np.full(len(features), LOG_EXPECTED_PRICE),
            )
            joblib.dump(
                {
                    "preprocessor": preprocessor,
                    "model": model,
                    "target_transform": "log1p",
                },
                root / "extra_trees.joblib",
            )
            write_v3_contract(
                root, "extra_trees", {"bundle": "extra_trees.joblib"}
            )

            with self.assertRaisesRegex(ModelRuntimeError, "contract|width"):
                ModelRuntime.from_directory(root)

    def test_rejects_catboost_saved_feature_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = canonical_features().drop(columns=[MODEL_FEATURES[-1]])
            categorical_indices = [
                features.columns.get_loc(name)
                for name in CATEGORICAL_FEATURES
                if name in features
            ]
            model = CatBoostRegressor(
                iterations=1,
                depth=1,
                allow_const_label=True,
                allow_writing_files=False,
                verbose=False,
                cat_features=categorical_indices,
            )
            model.fit(
                features,
                np.full(len(features), LOG_EXPECTED_PRICE),
                verbose=False,
            )
            model.save_model(root / "catboost.cbm")
            write_v3_contract(root, "catboost", {"model": "catboost.cbm"})

            with self.assertRaisesRegex(ModelRuntimeError, "contract|feature"):
                ModelRuntime.from_directory(root)

    def test_rejects_catboost_without_canonical_categorical_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = canonical_features()
            for column in CATEGORICAL_FEATURES:
                features[column] = 1.0
            model = CatBoostRegressor(
                iterations=1,
                depth=1,
                allow_const_label=True,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(
                features,
                np.full(len(features), LOG_EXPECTED_PRICE),
                verbose=False,
            )
            model.save_model(root / "catboost.cbm")
            write_v3_contract(root, "catboost", {"model": "catboost.cbm"})

            with self.assertRaisesRegex(ModelRuntimeError, "categorical|indices"):
                ModelRuntime.from_directory(root)

    def test_rejects_nonfinite_wrong_shape_and_wrong_length_predictions(self):
        invalid_outputs = (
            np.array([np.nan, np.nan]),
            np.array([[LOG_EXPECTED_PRICE], [LOG_EXPECTED_PRICE]]),
            np.array([LOG_EXPECTED_PRICE]),
        )
        for output in invalid_outputs:
            with self.subTest(shape=output.shape):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write_extra_trees_artifacts(root)
                    runtime = ModelRuntime.from_directory(root)

                    with patch.object(
                        runtime._implementation.model,
                        "predict",
                        return_value=output,
                    ):
                        with self.assertRaises(ModelRuntimeError):
                            runtime.predict(raw_vehicles(2))


class PredictServiceRuntimeTests(unittest.TestCase):
    def tearDown(self):
        import predict_service

        predict_service.clear_model_runtime_cache()

    def test_import_does_not_load_artifacts(self):
        import predict_service
        from services import model_runtime

        with patch.object(
            model_runtime.ModelRuntime, "from_directory"
        ) as from_directory:
            importlib.reload(predict_service)

        from_directory.assert_not_called()

    def test_prediction_reuses_cached_runtime(self):
        import predict_service

        runtime = Mock()
        runtime.predict_one.return_value = 123.0
        predict_service.clear_model_runtime_cache()
        with patch.object(
            predict_service.ModelRuntime,
            "from_directory",
            return_value=runtime,
        ) as loader:
            first = predict_service.predict_price_one({"brand": "Honda"})
            second = predict_service.predict_price_one({"brand": "Toyota"})

        self.assertEqual((first, second), (123.0, 123.0))
        loader.assert_called_once_with(predict_service.settings.models_dir)

    def test_cache_clear_and_reload_load_a_new_runtime(self):
        import predict_service

        first_runtime = Mock()
        first_runtime.predict_one.return_value = 1.0
        second_runtime = Mock()
        second_runtime.predict_one.return_value = 2.0
        predict_service.clear_model_runtime_cache()
        with patch.object(
            predict_service.ModelRuntime,
            "from_directory",
            side_effect=[first_runtime, second_runtime],
        ) as loader:
            self.assertEqual(predict_service.predict_price_one({}), 1.0)
            reloaded = predict_service.reload_model_runtime()
            self.assertIs(reloaded, second_runtime)
            self.assertEqual(predict_service.predict_price_one({}), 2.0)

        self.assertEqual(loader.call_count, 2)

    def test_concurrent_cold_start_loads_runtime_once(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_publication(root)
            runtime = Mock()
            callers = 8
            start = threading.Barrier(callers)

            def load_once(_root):
                time.sleep(0.05)
                return runtime

            def get_runtime():
                start.wait(timeout=2)
                return predict_service.get_model_runtime()

            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
                experiment_path=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=load_once,
                ) as loader,
                ThreadPoolExecutor(max_workers=callers) as executor,
            ):
                runtimes = list(executor.map(lambda _: get_runtime(), range(callers)))

            self.assertTrue(all(value is runtime for value in runtimes))
            loader.assert_called_once_with(root)

    def test_manifest_content_replacement_automatically_reloads_runtime(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_publication(root, model_version="v3-first")
            first_runtime = Mock()
            second_runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
                experiment_path=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=[first_runtime, second_runtime],
                ) as loader,
            ):
                self.assertIs(predict_service.get_model_runtime(), first_runtime)
                write_cache_v3_publication(root, model_version="v3-second")
                self.assertIs(predict_service.get_model_runtime(), second_runtime)

            self.assertEqual(loader.call_count, 2)

    def test_formal_report_replacement_automatically_reloads_runtime(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_formal_reports(root)
            first_runtime = Mock()
            second_runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=[first_runtime, second_runtime],
                ) as loader,
            ):
                self.assertIs(predict_service.get_model_runtime(), first_runtime)
                (root / "leaderboard.json").write_text(
                    json.dumps({"marker": "report-v2"}, sort_keys=True),
                    encoding="utf-8",
                )
                self.assertIs(predict_service.get_model_runtime(), second_runtime)

            self.assertEqual(loader.call_count, 2)

    def test_formal_report_gap_uses_bounded_stale_runtime_fallback(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_formal_reports(root)
            manifest_path = root / "model_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_version"] = "3.0.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    return_value=runtime,
                ) as loader,
            ):
                self.assertIs(predict_service.get_model_runtime(), runtime)
                (root / "leaderboard.json").unlink()
                self.assertIs(predict_service.get_model_runtime(), runtime)
                with self.assertRaises(ModelRuntimeError):
                    predict_service.get_model_runtime()

            loader.assert_called_once_with(root)

    def test_manifest_gap_keeps_old_runtime_until_complete_identity_appears(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "model_manifest.json"
            write_cache_v3_publication(root, model_version="v3-first")
            first_runtime = Mock()
            second_runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
                experiment_path=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=[first_runtime, second_runtime],
                ) as loader,
            ):
                self.assertIs(predict_service.get_model_runtime(), first_runtime)
                manifest_path.unlink()
                self.assertIs(predict_service.get_model_runtime(), first_runtime)
                write_cache_v3_publication(root, model_version="v3-second")
                self.assertIs(predict_service.get_model_runtime(), second_runtime)

            self.assertEqual(loader.call_count, 2)

    def test_legacy_artifact_identity_change_automatically_reloads_runtime(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_legacy_publication(root)
            first_runtime = Mock()
            second_runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=[first_runtime, second_runtime],
                ) as loader,
            ):
                self.assertIs(predict_service.get_model_runtime(), first_runtime)
                (root / "price_mlp.pt").write_bytes(b"model-v2-longer")
                self.assertIs(predict_service.get_model_runtime(), second_runtime)

            self.assertEqual(loader.call_count, 2)

    def test_v3_artifact_replacement_reloads_with_unchanged_manifest(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_publication(
                root,
                artifact_bytes=b"a" * 100_000,
            )
            manifest_bytes = (root / "model_manifest.json").read_bytes()
            first_runtime = Mock()
            second_runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=[first_runtime, second_runtime],
                ) as loader,
            ):
                self.assertIs(predict_service.get_model_runtime(), first_runtime)
                (root / "winner.joblib").write_bytes(b"b" * 900_000)
                self.assertEqual(
                    (root / "model_manifest.json").read_bytes(), manifest_bytes
                )
                self.assertIs(predict_service.get_model_runtime(), second_runtime)

            self.assertEqual(loader.call_count, 2)

    def test_feature_config_change_reloads_and_malformed_json_is_explicit(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_publication(root)
            first_runtime = Mock()
            second_runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=[first_runtime, second_runtime],
                ) as loader,
            ):
                self.assertIs(predict_service.get_model_runtime(), first_runtime)
                (root / "feature_config.json").write_text(
                    json.dumps({"marker": "feature-v2"}), encoding="utf-8"
                )
                self.assertIs(predict_service.get_model_runtime(), second_runtime)
                (root / "feature_config.json").write_text("{broken", encoding="utf-8")
                with self.assertRaisesRegex(ModelRuntimeError, "JSON|identity|invalid"):
                    predict_service.get_model_runtime()

            self.assertEqual(loader.call_count, 2)

    def test_stable_v3_to_legacy_rollback_reloads_runtime(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_publication(root)
            first_runtime = Mock()
            legacy_runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=[first_runtime, legacy_runtime],
                ) as loader,
            ):
                self.assertIs(predict_service.get_model_runtime(), first_runtime)
                write_cache_legacy_publication(root)
                self.assertIs(predict_service.get_model_runtime(), legacy_runtime)

            self.assertEqual(loader.call_count, 2)

    def test_loader_discards_intermediate_publication_and_returns_final(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_publication(root, model_version="v3-intermediate")
            intermediate_runtime = Mock()
            final_runtime = Mock()
            load_count = 0

            def load_during_transition(_root):
                nonlocal load_count
                load_count += 1
                if load_count == 1:
                    write_cache_v3_publication(
                        root,
                        model_version="v3-final",
                        artifact_bytes=b"final-artifact",
                        feature_marker="feature-final",
                    )
                    return intermediate_runtime
                return final_runtime

            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    side_effect=load_during_transition,
                ) as loader,
            ):
                runtime = predict_service.get_model_runtime()

            self.assertIs(runtime, final_runtime)
            self.assertEqual(loader.call_count, 2)

    def test_repeated_identity_gap_has_bounded_stale_fallback(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_publication(root)
            runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    return_value=runtime,
                ),
            ):
                self.assertIs(predict_service.get_model_runtime(), runtime)
                with patch.object(
                    predict_service,
                    "_published_artifact_identity",
                    side_effect=OSError("publication gap"),
                ):
                    self.assertIs(predict_service.get_model_runtime(), runtime)
                    with self.assertRaisesRegex(ModelRuntimeError, "unavailable|gap"):
                        predict_service.get_model_runtime()

    def test_successful_identity_read_resets_gap_failure_budget(self):
        import predict_service

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_cache_v3_publication(root)
            runtime = Mock()
            fake_settings = SimpleNamespace(
                models_dir=root,
                published_models_dir=root,
            )
            predict_service.clear_model_runtime_cache()
            with (
                patch.object(predict_service, "settings", fake_settings),
                patch.object(
                    predict_service.ModelRuntime,
                    "from_directory",
                    return_value=runtime,
                ),
            ):
                self.assertIs(predict_service.get_model_runtime(), runtime)
                stable_identity = predict_service._cached_identity
                with patch.object(
                    predict_service,
                    "_published_artifact_identity",
                    side_effect=(
                        OSError("gap one"),
                        stable_identity,
                        OSError("gap two"),
                        OSError("gap three"),
                    ),
                ):
                    self.assertIs(predict_service.get_model_runtime(), runtime)
                    self.assertIs(predict_service.get_model_runtime(), runtime)
                    self.assertIs(predict_service.get_model_runtime(), runtime)
                    with self.assertRaises(ModelRuntimeError):
                        predict_service.get_model_runtime()


if __name__ == "__main__":
    unittest.main()
