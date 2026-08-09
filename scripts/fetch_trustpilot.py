#!/usr/bin/env python3
"""Download Trustpilot review pages for a business domain (default: asknebula.com)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.collectors.trustpilot import (
    DEFAULT_DOMAIN,
    TRUSTPILOT_BASE,
    TrustpilotCollectError,
    fetch_trustpilot_pages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Trustpilot HTML pages for Nebula reviews. "
            f"Source: {TRUSTPILOT_BASE}/review/{DEFAULT_DOMAIN}"
        )
    )
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="Trustpilot domain")
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Number of review pages to fetch (~20 reviews/page)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "trustpilot",
        help="Directory for saved HTML pages",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        html_pages = fetch_trustpilot_pages(args.domain, max_pages=args.pages)
    except TrustpilotCollectError as exc:
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1

    saved = []
    for index, html in enumerate(html_pages, start=1):
        path = args.output_dir / f"{args.domain}_page_{index}.html"
        path.write_text(html, encoding="utf-8")
        saved.append(str(path))

    manifest = {
        "domain": args.domain,
        "source_url": f"{TRUSTPILOT_BASE}/review/{args.domain}",
        "pages_fetched": len(saved),
        "files": saved,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
