#!/usr/bin/env python3
"""CLI collector for Nebula Trustpilot / OBRIO DOU / App Store reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.collectors.app_store import AppStoreCollectError, collect_app_store_reviews
from app.collectors.dou import DouCollectError, collect_dou_reviews
from app.collectors.trustpilot import TrustpilotCollectError, collect_trustpilot_reviews
from app.services.processing import process_reviews
from app.storage import ProcessedReviewStore, ReviewStore, sample_by_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Nebula Trustpilot reviews by default "
            "(https://www.trustpilot.com/review/asknebula.com)."
        )
    )
    parser.add_argument(
        "--source",
        choices=("trustpilot", "dou", "app_store", "both"),
        default="trustpilot",
        help="Review source (default: trustpilot)",
    )
    parser.add_argument(
        "--trustpilot-domain",
        default="asknebula.com",
        help="Trustpilot business domain",
    )
    parser.add_argument("--company-slug", default="obrio", help="DOU company slug")
    parser.add_argument(
        "--app-id",
        default="1459969523",
        help="Apple App Store numeric ID (default: Nebula)",
    )
    parser.add_argument("--country", default="us", help="App Store country code")
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of reviews to keep after sampling",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Disable random sampling (take first N)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "reviews.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collected = []
    errors: list[str] = []

    if args.source in {"trustpilot", "both"}:
        try:
            reviews = collect_trustpilot_reviews(
                domain=args.trustpilot_domain,
                max_reviews=max(args.count, 100),
                random_sample=not args.no_random,
            )
            print(
                f"Trustpilot: collected {len(reviews)} reviews for '{args.trustpilot_domain}'"
            )
            collected.extend(reviews)
        except TrustpilotCollectError as exc:
            errors.append(str(exc))
            print(f"Trustpilot error: {exc}", file=sys.stderr)

    if args.source in {"dou", "both"}:
        try:
            dou = collect_dou_reviews(company_slug=args.company_slug)
            print(f"DOU: collected {len(dou)} reviews for '{args.company_slug}'")
            collected.extend(dou)
        except DouCollectError as exc:
            errors.append(str(exc))
            print(f"DOU error: {exc}", file=sys.stderr)

    if args.source in {"app_store", "both"}:
        try:
            app_reviews = collect_app_store_reviews(
                app_id=args.app_id,
                country=args.country,
                max_reviews=max(args.count, 100),
            )
            print(f"App Store: collected {len(app_reviews)} reviews for app {args.app_id}")
            collected.extend(app_reviews)
        except AppStoreCollectError as exc:
            errors.append(str(exc))
            print(f"App Store error: {exc}", file=sys.stderr)

    if not collected:
        print("No reviews collected.", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        return 1

    sampled = sample_by_source(
        collected,
        count_per_source=args.count,
        random_sample=not args.no_random,
    )
    store = ReviewStore(args.output)
    stored = store.upsert_sources(sampled)
    processed_path = args.output.with_name("processed_reviews.json")
    processed = process_reviews(stored)
    ProcessedReviewStore(processed_path).replace(processed)

    by_source: dict[str, int] = {}
    for review in stored:
        by_source[review.source.value] = by_source.get(review.source.value, 0) + 1

    print(
        json.dumps(
            {
                "stored": len(stored),
                "processed": len(processed),
                "pool": len(collected),
                "sources": by_source,
                "output": str(args.output),
                "processed_output": str(processed_path),
                "warnings": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
