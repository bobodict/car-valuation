import unittest
from unittest.mock import Mock, patch

import predict_service


class PredictionArtifactScalingTests(unittest.TestCase):
    def test_prediction_delegates_target_scaling_to_runtime(self):
        runtime = Mock()
        runtime.predict_one.return_value = 200.0
        vehicle = {"brand": "Honda", "year": 2018}

        with patch.object(
            predict_service, "get_model_runtime", return_value=runtime
        ):
            result = predict_service.predict_price_one(vehicle)

        self.assertEqual(result, 200.0)
        runtime.predict_one.assert_called_once_with(vehicle)


if __name__ == "__main__":
    unittest.main()
