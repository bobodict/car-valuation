"""Download the public source dataset and record immutable provenance."""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


SOURCE_ID = "car-details-v4"
SOURCE_URL = "https://raw.githubusercontent.com/chandanverma07/DataSets/master/car%20details%20v4.csv"
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw" / "car-details-v4.csv"
DEFAULT_METADATA = DEFAULT_DESTINATION.with_name("manifest.json")


class DownloadError(RuntimeError):
    """Raised when a source cannot be downloaded or validated."""


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_json(destination: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(destination, payload)


def download_dataset(
    url: str,
    destination: str | Path,
    metadata_path: str | Path,
    opener=urlopen,
) -> dict:
    destination = Path(destination)
    metadata_path = Path(metadata_path)
    try:
        with opener(url, timeout=30) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise DownloadError(f"dataset download returned HTTP {status}")
            payload = response.read()
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"dataset download failed: {exc}") from exc

    if not payload:
        raise DownloadError("dataset download returned an empty body")

    manifest = {
        "source_id": SOURCE_ID,
        "source_url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "raw_path": str(destination),
    }
    _atomic_write_bytes(destination, payload)
    _atomic_write_json(metadata_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public car valuation dataset.")
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    manifest = download_dataset(args.url, args.destination, args.metadata)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
