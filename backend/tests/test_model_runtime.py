import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

from services.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    enrich_features,
)
from services.model_competition import MLPRegressor, _build_fold_preprocessor
from services.model_runtime import ModelRuntime, ModelRuntimeError


EXPECTED_PRICE = 505_000.0
LOG_EXPECTED_PRICE = math.log1p(EXPECTED_PRICE)


class FittedPassthroughPreprocessor:
    def __init__(self):
        self.fitted_ = True

    def transform(self, frame):
        return np.zeros((len(frame), 1), dtype=float)


class StaticPredictor:
    def __init__(self, output):
        self.output = output
        self.fitted_ = True
        self.n_features_in_ = 1

    def predict(self, features):
        return self.output


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


def write_extra_trees_artifacts(root):
    features = canonical_features()
    preprocessor = _build_fold_preprocessor(scale_numeric=False)
    transformed = preprocessor.fit_transform(features)
    model = ExtraTreesRegressor(n_estimators=2, random_state=42)
    model.fit(transformed, np.full(len(features), LOG_EXPECTED_PRICE))
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


def write_legacy_artifacts(root):
    feature_config = {
        "artifact_version": "2.0.0",
        "feature_cols": ["mileage"],
        "numeric_features": ["mileage"],
        "categorical_features": [],
    }
    (root / "feature_config.json").write_text(
        json.dumps(feature_config), encoding="utf-8"
    )
    # Legacy artifacts may use a non-v3 feature contract, so use a minimal
    # fitted transformer matching the saved feature list.
    from sklearn.preprocessing import StandardScaler

    preprocessor = StandardScaler().fit(pd.DataFrame({"mileage": [1.0, 2.0]}))
    joblib.dump(preprocessor, root / "preprocess.joblib")
    model = MLPRegressor(1, hidden_dims=(2,), dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(2.0)
    torch.save(
        {
            "input_dim": 1,
            "hidden_dims": [2],
            "dropout": 0.0,
            "model_state": model.state_dict(),
            "target_mean": 100.0,
            "target_std": 50.0,
        },
        root / "price_mlp.pt",
    )


class ModelRuntimeLoadingTests(unittest.TestCase):
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

            self.assertEqual(runtime.predict_one({"mileage": 10_000}), 200.0)

    def test_predict_one_requires_a_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_legacy_artifacts(root)
            runtime = ModelRuntime.from_directory(root)

            with self.assertRaisesRegex(ModelRuntimeError, "mapping"):
                runtime.predict_one([{"mileage": 10_000}])


class ModelRuntimeValidationTests(unittest.TestCase):
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
            joblib.dump(
                {
                    "preprocessor": FittedPassthroughPreprocessor(),
                    "model": object(),
                    "target_transform": "log1p",
                },
                root / "extra_trees.joblib",
            )
            write_v3_contract(
                root, "extra_trees", {"bundle": "extra_trees.joblib"}
            )

            with self.assertRaises(ModelRuntimeError):
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
                    joblib.dump(
                        {
                            "preprocessor": FittedPassthroughPreprocessor(),
                            "model": StaticPredictor(output),
                            "target_transform": "log1p",
                        },
                        root / "extra_trees.joblib",
                    )
                    write_v3_contract(
                        root, "extra_trees", {"bundle": "extra_trees.joblib"}
                    )
                    runtime = ModelRuntime.from_directory(root)

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
        loader.assert_called_once_with(predict_service.settings.experiment_path)

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


if __name__ == "__main__":
    unittest.main()
