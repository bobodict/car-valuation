import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.download_public_dataset import DownloadError, download_dataset


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class DatasetDownloaderTests(unittest.TestCase):
    def test_download_writes_bytes_and_provenance_manifest(self):
        payload = b"Make,Model,Price\nHonda,Amaze,505000\n"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "raw" / "car-details-v4.csv"
            metadata = root / "raw" / "manifest.json"

            result = download_dataset(
                "https://example.test/car.csv",
                destination,
                metadata,
                opener=lambda url, timeout: FakeResponse(payload),
            )

            manifest = json.loads(metadata.read_text(encoding="utf-8"))
            expected_hash = hashlib.sha256(payload).hexdigest()

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(result["sha256"], expected_hash)
            self.assertEqual(manifest["source_url"], "https://example.test/car.csv")
            self.assertEqual(manifest["byte_count"], len(payload))
            self.assertEqual(manifest["sha256"], expected_hash)
            self.assertEqual(manifest["raw_path"], str(destination))
            self.assertIn("retrieved_at", manifest)

    def test_download_failure_keeps_existing_destination(self):
        original = b"old data"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "car-details-v4.csv"
            metadata = root / "manifest.json"
            destination.write_bytes(original)

            with self.assertRaises(DownloadError):
                download_dataset(
                    "https://example.test/car.csv",
                    destination,
                    metadata,
                    opener=lambda url, timeout: FakeResponse(b"new data", status=503),
                )

            self.assertEqual(destination.read_bytes(), original)
            self.assertFalse(metadata.exists())


if __name__ == "__main__":
    unittest.main()
