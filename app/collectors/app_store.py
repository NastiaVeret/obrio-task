from __future__ import annotations

import logging
from typing import Any

import httpx

from app.models.schemas import Review, ReviewSource

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
MAX_PAGES = 10  # Apple RSS exposes up to 10 pages × ~50 reviews


class AppStoreCollectError(Exception):
    """Raised when App Store review collection fails."""


def _entry_label(entry: dict[str, Any], key: str, default: str | None = None) -> str | None:
    value = entry.get(key)
    if isinstance(value, dict):
        return value.get("label", default)
    if isinstance(value, str):
        return value
    return default


def _parse_entries(entries: list[dict[str, Any]], app_id: str, country: str) -> list[Review]:
    reviews: list[Review] = []
    for entry in entries:
        # First entry on page 1 is often app metadata, not a review.
        if "im:rating" not in entry:
            continue

        review_id = _entry_label(entry, "id")
        text = _entry_label(entry, "content") or ""
        if not text.strip():
            continue

        author = None
        author_block = entry.get("author")
        if isinstance(author_block, dict):
            author = _entry_label(author_block, "name")

        rating_raw = _entry_label(entry, "im:rating")
        try:
            rating = float(rating_raw) if rating_raw is not None else None
        except (TypeError, ValueError):
            rating = None

        reviews.append(
            Review(
                id=f"appstore-{app_id}-{review_id}",
                source=ReviewSource.APP_STORE,
                author=author,
                title=_entry_label(entry, "title"),
                text=text.strip(),
                rating=rating,
                date=_entry_label(entry, "updated"),
                app_id=app_id,
                country=country,
                version=_entry_label(entry, "im:version"),
                url=f"https://apps.apple.com/{country}/app/id{app_id}",
                metadata={"vote_count": _entry_label(entry, "im:voteCount")},
            )
        )
    return reviews


def collect_app_store_reviews(
    app_id: str = "1459969523",
    country: str = "us",
    max_reviews: int = 100,
    timeout: float = 30.0,
) -> list[Review]:
    """Fetch customer reviews from Apple's public RSS feed."""
    if not app_id or not str(app_id).isdigit():
        raise AppStoreCollectError("app_id must be a numeric Apple App Store ID")
    if max_reviews < 1:
        raise AppStoreCollectError("max_reviews must be >= 1")

    country = country.strip().lower()
    collected: list[Review] = []
    seen_ids: set[str] = set()

    try:
        with httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
        ) as client:
            for page in range(1, MAX_PAGES + 1):
                url = (
                    f"https://itunes.apple.com/{country}/rss/customerreviews/"
                    f"page={page}/id={app_id}/sortBy=mostRecent/json"
                )
                try:
                    response = client.get(url)
                except httpx.TimeoutException as exc:
                    raise AppStoreCollectError(
                        f"Timed out fetching App Store reviews (page {page})"
                    ) from exc
                except httpx.HTTPError as exc:
                    raise AppStoreCollectError(
                        f"Network error fetching App Store reviews: {exc}"
                    ) from exc

                if response.status_code == 404:
                    if page == 1:
                        raise AppStoreCollectError(
                            f"App '{app_id}' not found for country '{country}'"
                        )
                    break
                if response.status_code >= 400:
                    raise AppStoreCollectError(
                        f"App Store RSS returned HTTP {response.status_code} for page {page}"
                    )

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise AppStoreCollectError(
                        f"Invalid JSON from App Store RSS (page {page})"
                    ) from exc

                feed = payload.get("feed") or {}
                entries = feed.get("entry") or []
                if isinstance(entries, dict):
                    entries = [entries]
                if not entries:
                    break

                page_reviews = _parse_entries(entries, app_id=app_id, country=country)
                if not page_reviews:
                    break

                for review in page_reviews:
                    if review.id in seen_ids:
                        continue
                    seen_ids.add(review.id)
                    collected.append(review)
                    if len(collected) >= max_reviews:
                        logger.info(
                            "Collected %s App Store reviews for app %s",
                            len(collected),
                            app_id,
                        )
                        return collected

    except AppStoreCollectError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AppStoreCollectError(f"Unexpected App Store collection error: {exc}") from exc

    if not collected:
        raise AppStoreCollectError(
            f"No reviews found for app_id={app_id} country={country}"
        )

    logger.info("Collected %s App Store reviews for app %s", len(collected), app_id)
    return collected
