#!/usr/bin/env python3
"""Extract key fields and clean review text from stored raw reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.processing import process_reviews
from app.storage import ProcessedReviewStore, ReviewStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract title/text/rating and preprocess review text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "reviews.json",
        help="Raw reviews JSON path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed_reviews.json",
        help="Processed reviews JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reviews = ReviewStore(args.input).load()
    if not reviews:
        print(f"No raw reviews found in {args.input}", file=sys.stderr)
        return 1

    processed = process_reviews(reviews)
    ProcessedReviewStore(args.output).replace(processed)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "input_count": len(reviews),
                "processed": len(processed),
                "dropped": len(reviews) - len(processed),
                "fields": ["id", "source", "title", "text", "rating", "text_raw", "title_raw"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
