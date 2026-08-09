#!/usr/bin/env python3
"""Collect Nebula (Trustpilot + App Store) + Obrio (DOU) reviews into one store."""

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
        description="Collect Nebula Trustpilot + App Store + Obrio DOU reviews"
    )
    parser.add_argument("--trustpilot-domain", default="asknebula.com")
    parser.add_argument("--company-slug", default="obrio")
    parser.add_argument(
        "--app-id",
        default="1459969523",
        help="Nebula Apple App Store ID",
    )
    parser.add_argument("--country", default="us", help="App Store country code")
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Max reviews to keep per source",
    )
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument(
        "--skip-trustpilot",
        action="store_true",
        help="Skip Nebula/Trustpilot",
    )
    parser.add_argument(
        "--skip-app-store",
        action="store_true",
        help="Skip Nebula/App Store",
    )
    parser.add_argument(
        "--skip-dou",
        action="store_true",
        help="Skip Obrio/DOU",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collected = []
    errors: list[str] = []

    if not args.skip_trustpilot:
        try:
            reviews = collect_trustpilot_reviews(
                domain=args.trustpilot_domain,
                max_reviews=max(args.count, 100),
                random_sample=not args.no_random,
            )
            print(f"Nebula/Trustpilot: {len(reviews)} reviews")
            collected.extend(reviews)
        except TrustpilotCollectError as exc:
            errors.append(str(exc))
            print(f"Trustpilot error: {exc}", file=sys.stderr)

    if not args.skip_app_store:
        try:
            reviews = collect_app_store_reviews(
                app_id=args.app_id,
                country=args.country,
                max_reviews=max(args.count, 100),
            )
            print(f"Nebula/App Store: {len(reviews)} reviews")
            collected.extend(reviews)
        except AppStoreCollectError as exc:
            errors.append(str(exc))
            print(f"App Store error: {exc}", file=sys.stderr)

    if not args.skip_dou:
        try:
            reviews = collect_dou_reviews(company_slug=args.company_slug)
            print(f"Obrio/DOU: {len(reviews)} reviews")
            collected.extend(reviews)
        except DouCollectError as exc:
            errors.append(str(exc))
            print(f"DOU error: {exc}", file=sys.stderr)

    if not collected:
        print("No reviews collected.", file=sys.stderr)
        return 1

    sampled = sample_by_source(
        collected,
        count_per_source=args.count,
        random_sample=not args.no_random,
    )
    stored = ReviewStore().upsert_sources(sampled)
    processed = process_reviews(stored)
    ProcessedReviewStore().replace(processed)

    by_source: dict[str, int] = {}
    for review in stored:
        by_source[review.source.value] = by_source.get(review.source.value, 0) + 1

    print(
        json.dumps(
            {
                "stored_total": len(stored),
                "processed_total": len(processed),
                "by_source": by_source,
                "analyze_with": {
                    "nebula": "dataset=nebula (trustpilot)",
                    "nebula_appstore": "dataset=nebula_appstore (app_store)",
                    "obrio": "dataset=obrio (dou)",
                    "all": "dataset=all",
                },
                "warnings": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
