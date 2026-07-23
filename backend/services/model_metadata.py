"""Validated access to the versioned model card."""

import json
from pathlib import Path

from config import settings
from predict_service import get_model_publication_state
from services.publication_validation import (
    validate_formal_v3_reports,
)


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


def _is_active_publication_root(root: Path) -> bool:
    published_root = getattr(settings, "published_models_dir", None)
    if published_root is None:
        return False
    return root.resolve() == Path(published_root).resolve()


def load_model_card(path: str | Path | None = None) -> dict:
    card_path = Path(path) if path else settings.models_dir / "model_card.json"
    expected_identity = None
    checks_publication = path is None or _is_active_publication_root(card_path.parent)
    if checks_publication:
        expected_identity, initial_is_stale = get_model_publication_state()
        if initial_is_stale:
            raise ValueError("model publication changed while model card was read")

    if checks_publication:
        formal_reports = validate_formal_v3_reports(card_path.parent)
    else:
        formal_reports = validate_formal_v3_reports(
            card_path.parent, require_generation=False
        )
    if formal_reports is None:
        with card_path.open("r", encoding="utf-8") as card_file:
            card = json.load(card_file)
    else:
        card = formal_reports["model_card.json"]

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
    v3_fold_split = {"development", "test", "folds"}.issubset(card["split"])
    v3_count_split = {"train", "validation", "development", "test"}.issubset(
        card["split"]
    )
    v3_split = v3_fold_split or v3_count_split
    if not legacy_split and not v3_split:
        raise ValueError(
            "model card split must include train/validation/test, "
            "development/test/folds, or train/validation/development/test"
        )
    if not isinstance(card["thresholds"], dict):
        raise ValueError("model card thresholds must be an object")
    if expected_identity is not None:
        final_identity, final_is_stale = get_model_publication_state()
        if final_identity != expected_identity or final_is_stale:
            raise ValueError("model publication changed while model card was read")
    return card
