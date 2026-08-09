#!/usr/bin/env python3
"""End-to-end Trustpilot pipeline: fetch → parse → process."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch, parse, and process Nebula Trustpilot reviews"
    )
    parser.add_argument("--domain", default="asknebula.com")
    parser.add_argument("--pages", type=int, default=5, help="Pages to fetch (~20/page)")
    parser.add_argument("--count", type=int, default=100, help="Reviews to keep")
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Keep first N reviews instead of random sample",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = ROOT / "data" / "raw" / "trustpilot"

    run(
        [
            PYTHON,
            str(ROOT / "scripts" / "fetch_trustpilot.py"),
            "--domain",
            args.domain,
            "--pages",
            str(args.pages),
            "--output-dir",
            str(raw_dir),
        ]
    )

    parse_cmd = [
        PYTHON,
        str(ROOT / "scripts" / "parse_trustpilot.py"),
        "--input-dir",
        str(raw_dir),
        "--domain",
        args.domain,
        "--count",
        str(args.count),
        "--output",
        str(ROOT / "data" / "reviews.json"),
    ]
    if args.no_random:
        parse_cmd.append("--no-random")
    run(parse_cmd)

    run(
        [
            PYTHON,
            str(ROOT / "scripts" / "process_reviews.py"),
            "--input",
            str(ROOT / "data" / "reviews.json"),
            "--output",
            str(ROOT / "data" / "processed_reviews.json"),
        ]
    )

    run(
        [
            PYTHON,
            str(ROOT / "scripts" / "generate_insights.py"),
            "--input",
            str(ROOT / "data" / "processed_reviews.json"),
            "--output",
            str(ROOT / "data" / "insights.json"),
        ]
    )

    run(
        [
            PYTHON,
            str(ROOT / "scripts" / "visualize_insights.py"),
            "--input",
            str(ROOT / "data" / "processed_reviews.json"),
            "--output",
            str(ROOT / "data" / "charts" / "insights_overview.png"),
        ]
    )

    print("\nPipeline complete:")
    print(f"  raw HTML : {raw_dir}")
    print(f"  parsed   : {ROOT / 'data' / 'reviews.json'}")
    print(f"  processed: {ROOT / 'data' / 'processed_reviews.json'}")
    print(f"  insights : {ROOT / 'data' / 'insights.json'}")
    print(f"  charts   : {ROOT / 'data' / 'charts' / 'insights_overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
