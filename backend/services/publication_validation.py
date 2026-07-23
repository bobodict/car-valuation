"""Lazy validation for complete formal v3 model publications."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


V3_ARTIFACT_VERSION = "3.0.0"
V3_REPORT_FILES = (
    "model_manifest.json",
    "feature_config.json",
    "metrics.json",
    "leaderboard.json",
    "error_analysis.json",
    "model_card.json",
)


def _load_manifest(root: Path) -> Mapping[str, Any] | None:
    manifest_path = root / "model_manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model_manifest.json contains invalid JSON") from exc

    if not isinstance(manifest, Mapping):
        raise ValueError("model_manifest.json must contain a JSON object")
    return manifest


def is_formal_v3_manifest(root: str | Path) -> bool:
    """Return whether a publication is marked as a formal v3 release."""
    manifest = _load_manifest(Path(root))
    return manifest is not None and manifest.get("artifact_version") == V3_ARTIFACT_VERSION


def validate_formal_v3_reports(root: str | Path) -> dict[str, Any] | None:
    """Strictly validate a complete v3 report set when one is present.

    Lightweight v3 runtime fixtures and legacy publications intentionally do
    not carry the full training report set, so they retain their existing
    compatibility behavior.
    """
    publication_root = Path(root)
    manifest = _load_manifest(publication_root)
    if manifest is None:
        return None
    if manifest.get("artifact_version") != V3_ARTIFACT_VERSION:
        return None
    missing_reports = [
        filename
        for filename in V3_REPORT_FILES
        if not (publication_root / filename).is_file()
    ]
    if missing_reports:
        raise ValueError(
            "formal v3 publication is incomplete; missing required reports: "
            + ", ".join(missing_reports)
        )

    # Keep the training validator out of import-time API dependencies.
    from scripts.train_model import _validate_experiment_directory

    return _validate_experiment_directory(publication_root)
