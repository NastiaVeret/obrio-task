#!/usr/bin/env python3
"""Parse Trustpilot HTML pages into structured review JSON.

Extracts key fields from each review:
  - title
  - text
  - rating (1-5 stars)
  - author, date, country, url

Default source: https://www.trustpilot.com/review/asknebula.com
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.collectors.trustpilot import (
    DEFAULT_DOMAIN,
    TrustpilotCollectError,
    parse_trustpilot_html,
)
from app.storage import ReviewStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse saved Trustpilot HTML into reviews.json"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "trustpilot",
        help="Directory with *.html pages from fetch_trustpilot.py",
    )
    parser.add_argument(
        "--domain",
        default=DEFAULT_DOMAIN,
        help="Trustpilot domain (used in review ids)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Max reviews to keep after parsing",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Keep first N reviews instead of a random sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "reviews.json",
        help="Parsed reviews output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    html_files = sorted(args.input_dir.glob("*.html"))
    if not html_files:
        print(
            f"No HTML files in {args.input_dir}. "
            "Run first: python scripts/fetch_trustpilot.py",
            file=sys.stderr,
        )
        return 1

    collected = []
    seen: set[str] = set()
    parse_errors: list[str] = []

    for path in html_files:
        try:
            html = path.read_text(encoding="utf-8")
            page_reviews = parse_trustpilot_html(html, domain=args.domain)
        except TrustpilotCollectError as exc:
            parse_errors.append(f"{path.name}: {exc}")
            continue

        for review in page_reviews:
            if review.id in seen:
                continue
            seen.add(review.id)
            collected.append(review)

    if not collected:
        print("No reviews parsed.", file=sys.stderr)
        for err in parse_errors:
            print(f" - {err}", file=sys.stderr)
        return 1

    if len(collected) > args.count:
        collected = (
            collected[: args.count]
            if args.no_random
            else random.sample(collected, args.count)
        )

    ReviewStore(args.output).replace(collected)

    with_rating = sum(1 for r in collected if r.rating is not None)
    with_title = sum(1 for r in collected if r.title)
    ratings = [int(r.rating) for r in collected if r.rating is not None]

    print(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "html_files": len(html_files),
                "parsed": len(collected),
                "with_title": with_title,
                "with_rating": with_rating,
                "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
                "output": str(args.output),
                "fields_extracted": [
                    "id",
                    "source",
                    "author",
                    "title",
                    "text",
                    "rating",
                    "date",
                    "country",
                    "url",
                ],
                "warnings": parse_errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
