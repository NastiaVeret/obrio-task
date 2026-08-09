from __future__ import annotations

import json
import logging
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.models.schemas import Review, ReviewSource

logger = logging.getLogger(__name__)

TRUSTPILOT_BASE = "https://www.trustpilot.com"
DEFAULT_DOMAIN = "asknebula.com"
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_BROWSER_INSTALL_HINT = (
    "Playwright Chromium is missing. Install with: "
    "python -m playwright install chromium "
    "(on Linux / Streamlit Cloud also add packages.txt system libs)"
)
_BROWSERS_INSTALLED = False


class TrustpilotCollectError(Exception):
    """Raised when Trustpilot review collection fails."""


def _project_browsers_path() -> Path:
    """Writable cache under the repo (works on Streamlit Cloud)."""
    root = Path(__file__).resolve().parents[2]
    return root / ".playwright-browsers"


def _configure_browsers_path() -> None:
    # Keep an explicit Docker/ms-playwright path if the host already set one.
    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        return
    path = _project_browsers_path()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(path)


def _missing_browser_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "executable doesn't exist" in message
        or "download new browsers" in message
        or "browser has been closed" in message
    )


def ensure_playwright_browsers(*, force: bool = False) -> None:
    """Download Chromium into the Playwright cache if needed.

    Safe to call from Streamlit startup / collect handlers.
    """
    global _BROWSERS_INSTALLED
    if _BROWSERS_INSTALLED and not force:
        return

    _configure_browsers_path()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise TrustpilotCollectError(
            "Playwright is required for Trustpilot collection. "
            "Install with: pip install playwright && python -m playwright install chromium"
        ) from exc

    needs_install = force
    if not needs_install:
        try:
            with sync_playwright() as playwright:
                exe = Path(playwright.chromium.executable_path)
                needs_install = not exe.exists()
        except Exception:  # noqa: BLE001
            needs_install = True

    if needs_install:
        logger.warning(
            "Installing Playwright Chromium into %s",
            os.environ.get("PLAYWRIGHT_BROWSERS_PATH"),
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
                env=os.environ.copy(),
            )
            if completed.stdout:
                logger.info(completed.stdout[-2000:])
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise TrustpilotCollectError(
                f"{_BROWSER_INSTALL_HINT}. Install failed: {detail}"
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise TrustpilotCollectError(f"{_BROWSER_INSTALL_HINT}. Details: {exc}") from exc

    _BROWSERS_INSTALLED = True


def _parse_next_data(html: str) -> dict[str, Any]:
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise TrustpilotCollectError(
            "Could not find Trustpilot review payload (__NEXT_DATA__). "
            "The page may be blocked or the layout changed."
        )
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise TrustpilotCollectError("Invalid Trustpilot __NEXT_DATA__ JSON") from exc


def parse_trustpilot_html(html: str, domain: str) -> list[Review]:
    data = _parse_next_data(html)
    page_props = (data.get("props") or {}).get("pageProps") or {}
    raw_reviews = page_props.get("reviews") or []
    if not isinstance(raw_reviews, list):
        raise TrustpilotCollectError("Unexpected Trustpilot reviews payload shape")

    reviews: list[Review] = []
    for item in raw_reviews:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        review_id = item.get("id") or str(len(reviews))
        consumer = item.get("consumer") or {}
        dates = item.get("dates") or {}
        rating = item.get("rating")
        try:
            rating_value = float(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating_value = None

        reviews.append(
            Review(
                id=f"trustpilot-{domain}-{review_id}",
                source=ReviewSource.TRUSTPILOT,
                author=consumer.get("displayName"),
                title=(item.get("title") or None),
                text=text,
                rating=rating_value,
                date=dates.get("publishedDate"),
                country=consumer.get("countryCode"),
                url=f"{TRUSTPILOT_BASE}/reviews/{review_id}",
                metadata={
                    "domain": domain,
                    "language": item.get("language"),
                    "likes": item.get("likes", 0),
                    "trustpilot_source": item.get("source"),
                },
            )
        )
    return reviews


def fetch_trustpilot_pages(domain: str, max_pages: int) -> list[str]:
    """Download Trustpilot review HTML pages using headless Chromium."""
    return _fetch_pages_with_playwright(domain, max_pages)


def _launch_chromium(playwright: Any):
    """Launch headless Chromium, installing browsers once if the cache is empty."""
    ensure_playwright_browsers()
    try:
        return playwright.chromium.launch(headless=True)
    except Exception as exc:  # noqa: BLE001
        if not _missing_browser_error(exc):
            raise
        ensure_playwright_browsers(force=True)
        try:
            return playwright.chromium.launch(headless=True)
        except Exception as retry_exc:  # noqa: BLE001
            raise TrustpilotCollectError(
                f"{_BROWSER_INSTALL_HINT}. Launch error: {retry_exc}"
            ) from retry_exc


def _fetch_pages_with_playwright(domain: str, max_pages: int) -> list[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise TrustpilotCollectError(
            "Playwright is required for Trustpilot collection. "
            "Install with: pip install playwright && python -m playwright install chromium"
        ) from exc

    _configure_browsers_path()
    pages_html: list[str] = []
    with sync_playwright() as playwright:
        browser = _launch_chromium(playwright)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()
        try:
            for page_num in range(1, max_pages + 1):
                url = f"{TRUSTPILOT_BASE}/review/{quote(domain)}"
                if page_num > 1:
                    url = f"{url}?page={page_num}"
                response = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                if response is not None and response.status == 404:
                    if page_num == 1:
                        raise TrustpilotCollectError(
                            f"Trustpilot business '{domain}' not found"
                        )
                    break
                # __NEXT_DATA__ is a hidden <script>; wait for attach, not visibility.
                page.wait_for_selector(
                    "#__NEXT_DATA__", state="attached", timeout=30_000
                )
                html = page.content()
                if "Verifying Connection" in html and "__NEXT_DATA__" not in html:
                    raise TrustpilotCollectError(
                        "Trustpilot blocked the request (bot verification)."
                    )
                pages_html.append(html)
        finally:
            context.close()
            browser.close()
    return pages_html


def collect_trustpilot_reviews(
    domain: str = DEFAULT_DOMAIN,
    max_reviews: int = 100,
    random_sample: bool = True,
) -> list[Review]:
    """Collect Nebula/AskNebula reviews from Trustpilot public pages."""
    if not domain or not domain.strip():
        raise TrustpilotCollectError("domain is required (e.g. asknebula.com)")
    domain = domain.strip().lower().removeprefix("https://").removeprefix("http://")
    domain = domain.removeprefix("www.").split("/")[0]
    if max_reviews < 1:
        raise TrustpilotCollectError("max_reviews must be >= 1")

    # Trustpilot serves ~20 reviews per page.
    max_pages = max(1, min(20, (max_reviews + 19) // 20 + (2 if random_sample else 0)))
    try:
        html_pages = _fetch_pages_with_playwright(domain, max_pages=max_pages)
    except TrustpilotCollectError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TrustpilotCollectError(f"Trustpilot browser fetch failed: {exc}") from exc

    collected: list[Review] = []
    seen: set[str] = set()
    for html in html_pages:
        page_reviews = parse_trustpilot_html(html, domain=domain)
        if not page_reviews:
            break
        for review in page_reviews:
            if review.id in seen:
                continue
            seen.add(review.id)
            collected.append(review)

    if not collected:
        raise TrustpilotCollectError(f"No Trustpilot reviews found for '{domain}'")

    if len(collected) > max_reviews:
        collected = (
            random.sample(collected, max_reviews)
            if random_sample
            else collected[:max_reviews]
        )

    logger.info("Collected %s Trustpilot reviews for %s", len(collected), domain)
    return collected
