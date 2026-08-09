from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from app.models.schemas import RatingBreakdown, Review
from app.services.sentiment import analyze_text

RatingBasis = Literal["native", "sentiment_proxy", "mixed", "none"]


@dataclass
class RatingMetrics:
    total_reviews: int
    reviews_with_native_rating: int
    reviews_used_for_rating: int
    average_rating: float | None
    rating_distribution: list[RatingBreakdown]
    rating_basis: RatingBasis
    note: str


def sentiment_score_to_stars(score: float) -> int:
    """Map sentiment score in [-1, 1] to a 1–5 star scale."""
    stars = int(round((score + 1.0) * 2.0 + 1.0))
    return max(1, min(5, stars))


def resolve_star_rating(
    review: Review,
    *,
    allow_sentiment_proxy: bool = True,
) -> tuple[int | None, Literal["native", "sentiment_proxy", "none"]]:
    """Return a 1–5 rating from the review, or a sentiment proxy when missing."""
    if review.rating is not None:
        try:
            stars = int(round(float(review.rating)))
        except (TypeError, ValueError):
            stars = None
        else:
            if 1 <= stars <= 5:
                return stars, "native"

    if not allow_sentiment_proxy:
        return None, "none"

    _, score, _ = analyze_text(review.text, rating=None)
    return sentiment_score_to_stars(score), "sentiment_proxy"


def calculate_rating_metrics(
    reviews: list[Review],
    *,
    allow_sentiment_proxy: bool = True,
) -> RatingMetrics:
    """Calculate average rating and 1–5 star distribution.

    DOU employer reviews usually have no native star rating. When
    ``allow_sentiment_proxy`` is True, missing ratings are inferred from
    text sentiment so the requested stats can still be computed.
    """
    if not reviews:
        return RatingMetrics(
            total_reviews=0,
            reviews_with_native_rating=0,
            reviews_used_for_rating=0,
            average_rating=None,
            rating_distribution=[
                RatingBreakdown(rating=star, count=0, percentage=0.0) for star in range(1, 6)
            ],
            rating_basis="none",
            note="No reviews available.",
        )

    resolved: list[int] = []
    bases: list[str] = []
    native_count = 0

    for review in reviews:
        stars, basis = resolve_star_rating(
            review, allow_sentiment_proxy=allow_sentiment_proxy
        )
        if basis == "native":
            native_count += 1
        if stars is None:
            continue
        resolved.append(stars)
        bases.append(basis)

    if not resolved:
        return RatingMetrics(
            total_reviews=len(reviews),
            reviews_with_native_rating=native_count,
            reviews_used_for_rating=0,
            average_rating=None,
            rating_distribution=[
                RatingBreakdown(rating=star, count=0, percentage=0.0) for star in range(1, 6)
            ],
            rating_basis="none",
            note=(
                "No star ratings available. DOU reviews do not include 1–5 stars; "
                "enable sentiment proxy or use a rated source."
            ),
        )

    counts = Counter(resolved)
    total_rated = len(resolved)
    distribution = [
        RatingBreakdown(
            rating=star,
            count=counts.get(star, 0),
            percentage=round(100.0 * counts.get(star, 0) / total_rated, 1),
        )
        for star in range(1, 6)
    ]

    unique_bases = set(bases)
    if unique_bases == {"native"}:
        basis: RatingBasis = "native"
        note = "Average and distribution use native star ratings from the source."
    elif unique_bases == {"sentiment_proxy"}:
        basis = "sentiment_proxy"
        note = (
            "DOU reviews have no native stars. Ratings were inferred from text "
            "sentiment (1–5 proxy) for metrics visualization."
        )
    else:
        basis = "mixed"
        note = (
            "Mixed basis: native stars where available, sentiment proxy otherwise."
        )

    return RatingMetrics(
        total_reviews=len(reviews),
        reviews_with_native_rating=native_count,
        reviews_used_for_rating=total_rated,
        average_rating=round(mean(resolved), 2),
        rating_distribution=distribution,
        rating_basis=basis,
        note=note,
    )


def rating_metrics_to_dict(metrics: RatingMetrics) -> dict:
    return {
        "total_reviews": metrics.total_reviews,
        "reviews_with_native_rating": metrics.reviews_with_native_rating,
        "reviews_used_for_rating": metrics.reviews_used_for_rating,
        "average_rating": metrics.average_rating,
        "rating_basis": metrics.rating_basis,
        "note": metrics.note,
        "rating_distribution": [
            item.model_dump() for item in metrics.rating_distribution
        ],
    }
