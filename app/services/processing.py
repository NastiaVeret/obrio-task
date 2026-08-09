from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from app.models.schemas import ProcessedReview, Review, ReviewSource

# Zero-width / BOM characters that often appear in scraped text
_ZW_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")
# Any unicode whitespace (including NBSP, thin space, etc.)
_SPACE_RE = re.compile(r"[^\S\n]+", re.UNICODE)
_NL_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# DOU sometimes leaves escaped separators like "допомога\фідбек"
_ESCAPED_SEP_RE = re.compile(r"\\+")


def clean_text(value: str | None, *, lowercase: bool = False, strip_urls: bool = True) -> str:
    """Normalize review/title text for downstream analysis."""
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZW_RE.sub("", text)
    text = _ESCAPED_SEP_RE.sub(" ", text)

    if strip_urls:
        text = _URL_RE.sub(" ", text)

    # Keep paragraph breaks, normalize other whitespace.
    text = _SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL_RE.sub("\n\n", text).strip()

    if lowercase:
        text = text.lower()

    return text


def extract_key_fields(review: Review) -> dict[str, Any]:
    """Pull the analysis-relevant fields from a raw source review."""
    return {
        "id": review.id,
        "source": review.source,
        "title": review.title,
        "text": review.text,
        "rating": review.rating,
    }


def process_review(review: Review) -> ProcessedReview | None:
    """Extract key fields and clean text. Returns None if text is empty after cleaning."""
    fields = extract_key_fields(review)
    text_raw = fields["text"] or ""
    title_raw = fields["title"]

    text = clean_text(text_raw)
    title = clean_text(title_raw) or None

    if not text:
        return None

    rating = fields["rating"]
    if rating is not None:
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            rating = None
        else:
            if rating < 1 or rating > 5:
                rating = None

    return ProcessedReview(
        id=fields["id"],
        source=ReviewSource(fields["source"]),
        title=title,
        text=text,
        rating=rating,
        text_raw=text_raw,
        title_raw=title_raw,
    )


def process_reviews(reviews: list[Review]) -> list[ProcessedReview]:
    """Process a list of raw reviews, dropping rows with empty cleaned text."""
    processed: list[ProcessedReview] = []
    for review in reviews:
        item = process_review(review)
        if item is not None:
            processed.append(item)
    return processed
