import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import predict_service


class PredictionArtifactScalingTests(unittest.TestCase):
    def test_prediction_unscales_training_target_before_returning_price(self):
        fake_model = Mock(return_value=torch.tensor([[2.0]]))
        with patch.object(predict_service.preprocess, "transform", return_value=np.zeros((1, 3))):
            with patch.object(predict_service, "model", fake_model):
                old_mean = getattr(predict_service, "TARGET_MEAN", None)
                old_std = getattr(predict_service, "TARGET_STD", None)
                predict_service.TARGET_MEAN = 100.0
                predict_service.TARGET_STD = 50.0
                try:
                    result = predict_service.predict_price_one({"brand": "Honda", "year": 2018})
                finally:
                    predict_service.TARGET_MEAN = old_mean
                    predict_service.TARGET_STD = old_std

        self.assertEqual(result, 200.0)


if __name__ == "__main__":
    unittest.main()
