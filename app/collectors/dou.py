from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup

from app.models.schemas import Review, ReviewSource

logger = logging.getLogger(__name__)

DOU_BASE = "https://jobs.dou.ua"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class DouCollectError(Exception):
    """Raised when DOU review collection fails."""


def _parse_supports(comment_el: Any) -> int:
    text = comment_el.get_text(" ", strip=True)
    match = re.search(r"Підтримали:\s*(.+?)(?:Відповісти|$)", text)
    if not match:
        return 0
    names = [n.strip() for n in match.group(1).split(",") if n.strip()]
    return len(names)


def parse_dou_html(html: str, company_slug: str) -> list[Review]:
    soup = BeautifulSoup(html, "lxml")
    comments_list = soup.select_one("#commentsList")
    if comments_list is None:
        raise DouCollectError(
            "Could not find reviews list on page. The company slug may be invalid "
            "or DOU page structure changed."
        )

    reviews: list[Review] = []
    for block in comments_list.select(":scope > .b-comment"):
        comment = block.select_one(".comment")
        text_el = block.select_one(".l-text")
        if comment is None or text_el is None:
            continue

        text = text_el.get_text("\n", strip=True)
        if not text:
            continue

        comment_id = comment.get("id", "").replace("comment_", "") or None
        author_el = block.select_one(".b-post-author .avatar")
        role_el = block.select_one(".prof")
        date_el = block.select_one("a.comment-link")

        review_id = f"dou-{company_slug}-{comment_id or len(reviews)}"
        reviews.append(
            Review(
                id=review_id,
                source=ReviewSource.DOU,
                author=author_el.get_text(strip=True) if author_el else None,
                text=text,
                date=date_el.get_text(strip=True) if date_el else None,
                role=role_el.get_text(" ", strip=True) if role_el else None,
                url=f"{DOU_BASE}/companies/{company_slug}/reviews/#{comment_id}"
                if comment_id
                else f"{DOU_BASE}/companies/{company_slug}/reviews/",
                metadata={
                    "company_slug": company_slug,
                    "supports": _parse_supports(block),
                    "reply_count": len(block.select(".b-comment")),
                },
            )
        )

    if not reviews:
        raise DouCollectError(
            f"No reviews found for company '{company_slug}'. "
            "Check the slug or that the company has public reviews."
        )

    return reviews


def collect_dou_reviews(
    company_slug: str = "obrio",
    timeout: float = 30.0,
) -> list[Review]:
    """Fetch employer reviews from DOU for the given company slug."""
    if not company_slug or not company_slug.strip():
        raise DouCollectError("company_slug is required")

    slug = company_slug.strip().lower()
    url = f"{DOU_BASE}/companies/{slug}/reviews/"

    try:
        with httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
            },
            follow_redirects=True,
        ) as client:
            response = client.get(url)
    except httpx.TimeoutException as exc:
        raise DouCollectError(f"Timed out fetching DOU reviews from {url}") from exc
    except httpx.HTTPError as exc:
        raise DouCollectError(f"Network error fetching DOU reviews: {exc}") from exc

    if response.status_code == 404:
        raise DouCollectError(f"Company '{slug}' not found on DOU")
    if response.status_code >= 400:
        raise DouCollectError(
            f"DOU returned HTTP {response.status_code} for {url}"
        )

    reviews = parse_dou_html(response.text, slug)
    logger.info("Collected %s DOU reviews for %s", len(reviews), slug)
    return reviews
