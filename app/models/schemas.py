from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ReviewSource(str, Enum):
    DOU = "dou"
    APP_STORE = "app_store"
    TRUSTPILOT = "trustpilot"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Review(BaseModel):
    """Raw review fields from the source site only (no derived analysis)."""

    id: str
    source: ReviewSource
    author: str | None = None
    title: str | None = None
    text: str
    rating: float | None = None
    date: str | None = None
    role: str | None = None
    app_id: str | None = None
    country: str | None = None
    version: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedReview(BaseModel):
    """Key fields extracted from a raw review, with cleaned text for analysis."""

    id: str
    source: ReviewSource
    title: str | None = None
    text: str
    rating: float | None = None
    text_raw: str
    title_raw: str | None = None


class CollectRequest(BaseModel):
    """Collect reviews for a specified app/product.

    Defaults to Nebula reviews on Trustpilot (`asknebula.com`).
    """

    app: str = Field(
        default="asknebula.com",
        description=(
            "App identifier: Trustpilot domain (asknebula.com), "
            "App Store numeric id, or DOU company slug"
        ),
    )
    source: ReviewSource | str = Field(
        default="both",
        description="trustpilot | app_store | dou | both (all three)",
    )
    company_slug: str | None = Field(
        default=None,
        description="DOU company slug (defaults from app when source=dou)",
    )
    trustpilot_domain: str | None = Field(
        default=None,
        description="Trustpilot domain (defaults from app when source=trustpilot)",
    )
    app_id: str | None = Field(
        default=None,
        description="Apple App Store numeric ID (defaults from app when source=app_store)",
    )
    country: str = Field(default="us", min_length=2, max_length=5)
    count: int = Field(default=100, ge=1, le=500)
    random_sample: bool = True

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"both", "all"}:
                return "both"
            if value in {"tp", "trust"}:
                return "trustpilot"
        return value

    @field_validator("app", mode="before")
    @classmethod
    def normalize_app(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        cleaned = value.strip().lower()
        aliases = {
            "nebula": "asknebula.com",
            "asknebula": "asknebula.com",
            "obrio": "obrio",
        }
        return aliases.get(cleaned, cleaned)

    def resolved_trustpilot_domain(self) -> str:
        value = self.trustpilot_domain or self.app
        cleaned = value.strip().lower()
        cleaned = cleaned.removeprefix("https://").removeprefix("http://")
        cleaned = cleaned.removeprefix("www.").split("/")[0]
        if cleaned in {"nebula", "asknebula"}:
            cleaned = "asknebula.com"
        if not cleaned or "." not in cleaned:
            raise ValueError("Trustpilot app/domain must look like asknebula.com")
        return cleaned

    def resolved_app_id(self) -> str:
        value = self.app_id or self.app
        # Convenience alias for Nebula App Store listing
        if value in {"nebula", "asknebula.com", "asknebula"}:
            value = "1459969523"
        if not str(value).isdigit():
            raise ValueError("app_id/app must be a numeric Apple App Store ID")
        return str(value)

    def resolved_company_slug(self) -> str:
        value = (self.company_slug or self.app).strip().lower()
        if value in {"asknebula.com", "asknebula", "nebula"}:
            value = "obrio"
        if not value.replace("-", "").isalnum():
            raise ValueError("company_slug/app must be alphanumeric (hyphens allowed)")
        return value


class CollectResponse(BaseModel):
    collected: int
    stored: int
    processed: int = 0
    sources: dict[str, int]
    message: str
    collected_at: datetime


class ProcessResponse(BaseModel):
    input_count: int
    processed: int
    dropped: int
    message: str


class ReviewsResponse(BaseModel):
    total: int
    returned: int
    reviews: list[Review]


class ProcessedReviewsResponse(BaseModel):
    total: int
    returned: int
    reviews: list[ProcessedReview]


class RatingBreakdown(BaseModel):
    rating: int
    count: int
    percentage: float


class SentimentBreakdown(BaseModel):
    label: SentimentLabel
    count: int
    percentage: float


class ThemeStat(BaseModel):
    theme: str
    count: int
    avg_sentiment_score: float


class RatingMetricsResponse(BaseModel):
    total_reviews: int
    reviews_with_native_rating: int
    reviews_used_for_rating: int
    average_rating: float | None
    rating_distribution: list[RatingBreakdown]
    rating_basis: str
    note: str


class MetricsResponse(BaseModel):
    total_reviews: int
    sources: dict[str, int]
    average_rating: float | None
    rating_distribution: list[RatingBreakdown]
    rating_basis: str = "none"
    rating_note: str = ""
    sentiment_distribution: list[SentimentBreakdown]
    average_sentiment_score: float
    top_themes: list[ThemeStat]
    average_text_length: float
    reviews_with_rating: int


class KeywordStat(BaseModel):
    term: str
    count: int
    type: str  # keyword | phrase


class Insight(BaseModel):
    priority: str
    category: str
    title: str
    detail: str
    evidence_count: int
    sample_quotes: list[str] = Field(default_factory=list)


class InsightsResponse(BaseModel):
    summary: str
    overall_sentiment: SentimentLabel
    sentiment_distribution: list[SentimentBreakdown] = Field(default_factory=list)
    negative_keywords: list[KeywordStat] = Field(default_factory=list)
    insights: list[Insight]
    recommended_actions: list[str]
    improvement_areas: list[str] = Field(default_factory=list)
