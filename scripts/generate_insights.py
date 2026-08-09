#!/usr/bin/env python3
"""Generate NLP insights from processed (or raw) reviews.

- Sentiment: RoBERTa FT (EN) / spaCy+lexicon (UK), optional rating blend
- Common keywords/phrases in negative reviews
- Actionable improvement areas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.schemas import Review
from app.services.analyzer import build_insights
from app.storage import ProcessedReviewStore, ReviewStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NLP insights from reviews")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed_reviews.json",
        help="Processed reviews JSON (falls back to raw reviews.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "insights.json",
        help="Where to write insights JSON",
    )
    return parser.parse_args()


def load_reviews(path: Path) -> list[Review]:
    if not path.exists():
        fallback = ROOT / "data" / "reviews.json"
        if path != fallback and fallback.exists():
            path = fallback
        else:
            return []

    if path.name.startswith("processed"):
        items = ProcessedReviewStore(path).load()
        return [
            Review(
                id=item.id,
                source=item.source,
                title=item.title,
                text=item.text,
                rating=item.rating,
            )
            for item in items
        ]
    return ReviewStore(path).load()


def main() -> int:
    args = parse_args()
    reviews = load_reviews(args.input)
    if not reviews:
        print(
            f"No reviews found at {args.input}. "
            "Run scripts/run_trustpilot_pipeline.py first.",
            file=sys.stderr,
        )
        return 1

    insights = build_insights(reviews)
    payload = insights.model_dump(mode="json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote insights → {args.output}")
    print(f"Overall sentiment: {insights.overall_sentiment.value}")
    print(f"Summary: {insights.summary}")
    print("\nNegative keywords/phrases:")
    for item in insights.negative_keywords[:10]:
        print(f"  - {item.term} ({item.count}) [{item.type}]")
    print("\nImprovement areas:")
    for area in insights.improvement_areas or insights.recommended_actions:
        print(f"  - {area}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
