from __future__ import annotations

import json
import random
from pathlib import Path
from threading import Lock
from typing import Generic, TypeVar

from pydantic import BaseModel

from app.models.schemas import ProcessedReview, Review, ReviewSource

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PATH = DATA_DIR / "reviews.json"
PROCESSED_PATH = DATA_DIR / "processed_reviews.json"

T = TypeVar("T", bound=BaseModel)

DATASET_ALIASES: dict[str, str | None] = {
    "all": None,
    "both": None,
    "nebula": "trustpilot",
    "asknebula": "trustpilot",
    "asknebula.com": "trustpilot",
    "trustpilot": "trustpilot",
    "nebula_appstore": "app_store",
    "nebula_ios": "app_store",
    "app_store": "app_store",
    "appstore": "app_store",
    "obrio": "dou",
    "dou": "dou",
}

DATASET_CHOICES = "nebula | nebula_appstore | obrio | all"


def resolve_dataset(dataset: str | None) -> str | None:
    """Return ReviewSource value to filter by, or None for all datasets."""
    if dataset is None or not str(dataset).strip():
        return None
    key = str(dataset).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"dataset must be one of: {DATASET_CHOICES}")
    return DATASET_ALIASES[key]


class JsonStore(Generic[T]):
    def __init__(self, path: Path, model: type[T]) -> None:
        self.path = path
        self.model = model
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[T]:
        with self._lock:
            if not self.path.exists():
                return []
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [self.model.model_validate(item) for item in raw]

    def save(self, items: list[T]) -> None:
        with self._lock:
            payload = [
                item.model_dump(mode="json", exclude_none=True) for item in items
            ]
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def replace(self, items: list[T]) -> list[T]:
        self.save(items)
        return items


class ReviewStore(JsonStore[Review]):
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        super().__init__(path, Review)

    def upsert_sources(self, reviews: list[Review]) -> list[Review]:
        """Replace reviews only for sources present in ``reviews``; keep others."""
        incoming_sources = {review.source for review in reviews}
        existing = [
            review for review in self.load() if review.source not in incoming_sources
        ]
        merged = existing + reviews
        self.save(merged)
        return merged

    def filter_dataset(self, dataset: str | None) -> list[Review]:
        source = resolve_dataset(dataset)
        reviews = self.load()
        if source is None:
            return reviews
        return [review for review in reviews if review.source.value == source]


class ProcessedReviewStore(JsonStore[ProcessedReview]):
    def __init__(self, path: Path = PROCESSED_PATH) -> None:
        super().__init__(path, ProcessedReview)

    def filter_dataset(self, dataset: str | None) -> list[ProcessedReview]:
        source = resolve_dataset(dataset)
        reviews = self.load()
        if source is None:
            return reviews
        return [review for review in reviews if review.source.value == source]


def sample_reviews(
    reviews: list[Review], count: int, random_sample: bool = True
) -> list[Review]:
    if count >= len(reviews):
        return list(reviews)
    if random_sample:
        return random.sample(reviews, count)
    return reviews[:count]


def sample_by_source(
    reviews: list[Review],
    count_per_source: int,
    random_sample: bool = True,
) -> list[Review]:
    """Sample up to N reviews from each source independently."""
    by_source: dict[ReviewSource, list[Review]] = {}
    for review in reviews:
        by_source.setdefault(review.source, []).append(review)

    sampled: list[Review] = []
    for source_reviews in by_source.values():
        sampled.extend(
            sample_reviews(
                source_reviews, count_per_source, random_sample=random_sample
            )
        )
    return sampled


store = ReviewStore()
processed_store = ProcessedReviewStore()
