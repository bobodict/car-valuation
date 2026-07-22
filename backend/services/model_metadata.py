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


def _load_optional_object(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as evidence_file:
            value = json.load(evidence_file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} contains malformed JSON") from exc
    except OSError as exc:
        raise ValueError(f"{path.name} could not be read") from exc

    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_model_card(path: str | Path | None = None) -> dict:
    card_path = Path(path) if path else settings.models_dir / "model_card.json"
    with card_path.open("r", encoding="utf-8") as card_file:
        card = json.load(card_file)

    if not isinstance(card, dict):
        raise ValueError("model card must contain a JSON object")

    if "data_source" not in card and "source_id" in card:
        card["data_source"] = {
            key: card[key]
            for key in ("source_id", "source_url")
            if key in card
        }
    card.setdefault("feature_version", card.get("artifact_version", "2.0.0"))
    card.setdefault("model_type", "mlp")

    for field, filename in (
        ("leaderboard", "leaderboard.json"),
        ("error_analysis", "error_analysis.json"),
    ):
        if field in card:
            if not isinstance(card[field], dict):
                raise ValueError(f"model card {field} must be an object")
        else:
            card[field] = _load_optional_object(card_path.parent / filename)

    missing = [key for key in REQUIRED_KEYS if key not in card]
    if missing:
        raise ValueError(f"model card missing keys: {', '.join(missing)}")
    if not card.get("currency") or not card.get("price_unit") or not card.get("mileage_unit"):
        raise ValueError("model card currency and units must be non-empty")
    if int(card["sample_count"]) <= 0:
        raise ValueError("model card sample_count must be positive")
    if not isinstance(card["data_source"], dict) or not card["data_source"].get("source_id"):
        raise ValueError("model card data_source.source_id is required")
    if not isinstance(card["split"], dict):
        raise ValueError("model card split must be an object")
    legacy_split = {"train", "validation", "test"}.issubset(card["split"])
    v3_split = {"development", "test"}.issubset(card["split"])
    if not legacy_split and not v3_split:
        raise ValueError(
            "model card split must include train/validation/test or development/test"
        )
    if not isinstance(card["thresholds"], dict):
        raise ValueError("model card thresholds must be an object")
    return card
