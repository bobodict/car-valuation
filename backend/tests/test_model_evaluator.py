import subprocess
import sys
import unittest
from pathlib import Path


class EvaluatorEntrypointTests(unittest.TestCase):
    def test_module_entrypoint_validates_dataset_before_loading_model(self):
        backend_dir = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "-m", "scripts.evaluate_model", "data/not-provided.csv"],
            cwd=backend_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileNotFoundError", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertNotIn("AttributeError", result.stderr)


if __name__ == "__main__":
    unittest.main()
