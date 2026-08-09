#!/usr/bin/env python3
"""Smoke-test the REST API endpoints (server must already be running)."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def call(method: str, url: str, body: dict | None = None) -> tuple[int, object]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            payload: object
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"bytes": len(raw), "preview": raw[:120]}
            return response.status, payload
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        return exc.code, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Also call POST /api/v1/collect (slow; uses Playwright)",
    )
    args = parser.parse_args()
    base = args.base.rstrip("/")

    checks = [
        ("GET", f"{base}/health", None),
        ("GET", f"{base}/api/v1/metrics", None),
        ("GET", f"{base}/api/v1/insights", None),
        ("GET", f"{base}/api/v1/reviews?limit=2", None),
        ("GET", f"{base}/api/v1/reviews/download?format=json", None),
        ("GET", f"{base}/api/v1/reviews/download?format=csv", None),
    ]
    if args.collect:
        checks.insert(
            1,
            (
                "POST",
                f"{base}/api/v1/collect",
                {
                    "app": "asknebula.com",
                    "source": "trustpilot",
                    "count": 20,
                    "random_sample": True,
                },
            ),
        )

    failed = 0
    for method, url, body in checks:
        status, payload = call(method, url, body)
        ok = 200 <= status < 300
        failed += 0 if ok else 1
        summary = payload
        if isinstance(payload, dict):
            summary = {
                k: payload[k]
                for k in (
                    "status",
                    "total_reviews",
                    "stored",
                    "overall_sentiment",
                    "total",
                    "returned",
                    "bytes",
                    "average_rating",
                )
                if k in payload
            } or list(payload.keys())[:8]
        print(f"{'OK' if ok else 'FAIL'} {method} {url} -> {status} {summary}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
