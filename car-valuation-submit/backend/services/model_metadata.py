"""Validated access to the versioned model card."""

import json
from pathlib import Path

from config import settings


REQUIRED_KEYS = (
    "currency",
    "price_unit",
    "mileage_unit",
    "sample_count",
    "model_version",
    "split",
    "thresholds",
    "category_options",
)


def load_model_card(path: str | Path | None = None) -> dict:
    card_path = Path(path) if path else settings.models_dir / "model_card.json"
    with card_path.open("r", encoding="utf-8") as card_file:
        card = json.load(card_file)

    if "data_source" not in card and "source_id" in card:
        card["data_source"] = {
            key: card[key]
            for key in ("source_id", "source_url")
            if key in card
        }
    if "feature_version" not in card and "artifact_version" in card:
        card["feature_version"] = card["artifact_version"]

    missing = [key for key in REQUIRED_KEYS if key not in card]
    if missing:
        raise ValueError(f"model card missing keys: {', '.join(missing)}")
    if not card.get("currency") or not card.get("price_unit") or not card.get("mileage_unit"):
        raise ValueError("model card currency and units must be non-empty")
    if int(card["sample_count"]) <= 0:
        raise ValueError("model card sample_count must be positive")
    if not isinstance(card["data_source"], dict) or not card["data_source"].get("source_id"):
        raise ValueError("model card data_source.source_id is required")
    if not isinstance(card["split"], dict) or not {"train", "validation", "test"}.issubset(card["split"]):
        raise ValueError("model card split must include train, validation, and test")
    if not isinstance(card["thresholds"], dict):
        raise ValueError("model card thresholds must be an object")
    return card
