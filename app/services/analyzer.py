from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

from app.models.schemas import (
    Insight,
    InsightsResponse,
    KeywordStat,
    MetricsResponse,
    Review,
    ReviewSource,
    SentimentBreakdown,
    SentimentLabel,
    ThemeStat,
)
from app.services.keywords import extract_negative_keywords
from app.services.metrics import calculate_rating_metrics
from app.services.sentiment import analyze_text

IMPROVEMENT_RULES: list[tuple[tuple[str, ...], str]] = [
    (
        ("refund", "money back", "charged", "charge", "billing", "subscription", "cancel", "cancelled"),
        "Billing & cancellations: make cancel/refund flows clearer and confirm charges stop immediately.",
    ),
    (
        ("scam", "fraud", "steal", "stolen", "ripoff", "rip off", "шахрай", "обман"),
        "Trust & transparency: publish pricing upfront and respond publicly to scam/fraud accusations.",
    ),
    (
        ("advisor", "psychic", "reading", "generic", "vague", "copy paste", "template"),
        "Advisor quality: raise reading specificity standards and reduce generic/template responses.",
    ),
    (
        ("credits", "credit", "minutes", "paywall", "expensive", "overpriced", "cost"),
        "Credits/pricing UX: clarify credit burn rate before chat and reduce surprise upsells.",
    ),
    (
        (
            "customer service",
            "support",
            "no response",
            "ignored",
            "reply",
            "email",
            "підтрим",
            "ігнор",
            "відповід",
        ),
        "Support / communication: set clear SLA for replies and close the loop on open issues.",
    ),
    (
        ("bug", "crash", "glitch", "error", "broken", "login"),
        "Product reliability: triage crash/login bugs by version and publish fix notes.",
    ),
    (
        ("зарплат", "оклад", "bonus", "бонус", "компенсац"),
        "Compensation clarity: document salary bands, bonus rules, and review cadence.",
    ),
    (
        ("овертайм", "переработ", "вигоран", "burnout", "overtime", "workload", "навантаж"),
        "Workload balance: audit overtime expectations and protect sustainable delivery pace.",
    ),
    (
        ("токсич", "toxic", "менеджмент", "керівниц", "звільн", "скороч"),
        "People management: investigate management/communication friction raised in reviews.",
    ),
]


def _uses_rating(review: Review) -> bool:
    """True when a native 1–5 star rating should influence metrics/sentiment."""
    return review.source != ReviewSource.DOU and review.rating is not None


@dataclass
class AnalyzedReview:
    review: Review
    sentiment: SentimentLabel
    sentiment_score: float
    themes: list[str]


def analyze_reviews(reviews: list[Review]) -> list[AnalyzedReview]:
    analyzed: list[AnalyzedReview] = []
    for review in reviews:
        rating = review.rating if _uses_rating(review) else None
        label, score, themes = analyze_text(review.text, rating)
        analyzed.append(
            AnalyzedReview(
                review=review,
                sentiment=label,
                sentiment_score=score,
                themes=themes,
            )
        )
    return analyzed


def compute_metrics(reviews: list[Review]) -> MetricsResponse:
    if not reviews:
        return MetricsResponse(
            total_reviews=0,
            sources={},
            average_rating=None,
            rating_distribution=[],
            rating_basis="none",
            rating_note="No reviews available.",
            sentiment_distribution=[],
            average_sentiment_score=0.0,
            top_themes=[],
            average_text_length=0.0,
            reviews_with_rating=0,
        )

    analyzed = analyze_reviews(reviews)
    sources = Counter(item.review.source.value for item in analyzed)
    rated_reviews = [item.review for item in analyzed if _uses_rating(item.review)]
    rating_metrics = calculate_rating_metrics(
        rated_reviews, allow_sentiment_proxy=False
    )
    if not rated_reviews:
        rating_metrics.note = (
            "No star ratings for this dataset (Obrio/DOU). "
            "Use sentiment + keywords insights instead."
        )

    sentiments = Counter(item.sentiment.value for item in analyzed)
    theme_scores: dict[str, list[float]] = defaultdict(list)
    for item in analyzed:
        for theme in item.themes:
            theme_scores[theme].append(item.sentiment_score)

    total = len(analyzed)
    sentiment_distribution = [
        SentimentBreakdown(
            label=SentimentLabel(label),
            count=sentiments.get(label, 0),
            percentage=round(100 * sentiments.get(label, 0) / total, 1),
        )
        for label in (
            SentimentLabel.POSITIVE,
            SentimentLabel.NEUTRAL,
            SentimentLabel.NEGATIVE,
        )
    ]
    top_themes = sorted(
        [
            ThemeStat(
                theme=theme,
                count=len(scores),
                avg_sentiment_score=round(mean(scores), 3),
            )
            for theme, scores in theme_scores.items()
        ],
        key=lambda item: (-item.count, item.theme),
    )[:10]

    return MetricsResponse(
        total_reviews=total,
        sources=dict(sources),
        average_rating=rating_metrics.average_rating,
        rating_distribution=rating_metrics.rating_distribution,
        rating_basis=rating_metrics.rating_basis,
        rating_note=rating_metrics.note,
        sentiment_distribution=sentiment_distribution,
        average_sentiment_score=round(
            mean([item.sentiment_score for item in analyzed]), 3
        ),
        top_themes=top_themes,
        average_text_length=round(mean(len(item.review.text) for item in analyzed), 1),
        reviews_with_rating=rating_metrics.reviews_used_for_rating,
    )


def _quotes_for(
    items: list[AnalyzedReview],
    *,
    theme: str | None = None,
    sentiment: SentimentLabel | None = None,
    contains: str | None = None,
    limit: int = 2,
) -> list[str]:
    selected: list[str] = []
    for item in items:
        if theme and theme not in item.themes:
            continue
        if sentiment and item.sentiment != sentiment:
            continue
        blob = f"{item.review.title or ''} {item.review.text}".lower()
        if contains and contains.lower() not in blob:
            continue
        snippet = item.review.text.replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        if snippet and snippet not in selected:
            selected.append(snippet)
        if len(selected) >= limit:
            break
    return selected


def _improvement_areas(negative_terms: list[str]) -> list[str]:
    joined = " | ".join(negative_terms).lower()
    areas: list[str] = []
    for patterns, action in IMPROVEMENT_RULES:
        if any(pattern in joined for pattern in patterns):
            areas.append(action)
    return areas


def build_insights(reviews: list[Review]) -> InsightsResponse:
    if not reviews:
        return InsightsResponse(
            summary="No reviews available. Run collection/parsing first.",
            overall_sentiment=SentimentLabel.NEUTRAL,
            sentiment_distribution=[],
            negative_keywords=[],
            insights=[],
            recommended_actions=[
                "Run: python scripts/run_trustpilot_pipeline.py --count 100"
            ],
            improvement_areas=[],
        )

    analyzed = analyze_reviews(reviews)
    metrics = compute_metrics(reviews)

    positive = next(
        s.count for s in metrics.sentiment_distribution if s.label == SentimentLabel.POSITIVE
    )
    negative = next(
        s.count for s in metrics.sentiment_distribution if s.label == SentimentLabel.NEGATIVE
    )
    neutral = next(
        s.count for s in metrics.sentiment_distribution if s.label == SentimentLabel.NEUTRAL
    )

    overall = SentimentLabel.POSITIVE
    if negative > positive:
        overall = SentimentLabel.NEGATIVE
    elif abs(positive - negative) <= max(1, len(reviews) * 0.08) and neutral >= positive:
        overall = SentimentLabel.NEUTRAL

    negative_items = [
        item for item in analyzed if item.sentiment == SentimentLabel.NEGATIVE
    ]
    positive_items = [
        item for item in analyzed if item.sentiment == SentimentLabel.POSITIVE
    ]
    negative_texts = [
        f"{item.review.title or ''} {item.review.text}".strip() for item in negative_items
    ]
    positive_texts = [
        f"{item.review.title or ''} {item.review.text}".strip() for item in positive_items
    ]
    negative_keyword_rows = extract_negative_keywords(
        negative_texts, positive_texts, top_n=15
    )
    negative_keywords = [KeywordStat.model_validate(row) for row in negative_keyword_rows]
    negative_terms = [row["term"] for row in negative_keyword_rows]

    insights: list[Insight] = []
    theme_map = {t.theme: t for t in metrics.top_themes}

    if negative_keywords:
        top_terms = ", ".join(k.term for k in negative_keywords[:5])
        insights.append(
            Insight(
                priority="high",
                category="risk",
                title="Common language in negative reviews",
                detail=(
                    f"From {len(negative_items)} negative-sentiment reviews, "
                    f"top terms/phrases are: {top_terms}."
                ),
                evidence_count=len(negative_items),
                sample_quotes=_quotes_for(
                    negative_items, sentiment=SentimentLabel.NEGATIVE, limit=2
                )
                or _quotes_for(negative_items, limit=2),
            )
        )
    elif negative_items:
        insights.append(
            Insight(
                priority="high",
                category="risk",
                title="Negative sentiment detected",
                detail=(
                    f"{len(negative_items)} reviews are negative by NLP sentiment. "
                    "Review quotes below for concrete issues."
                ),
                evidence_count=len(negative_items),
                sample_quotes=_quotes_for(negative_items, limit=3),
            )
        )

    if "pricing_billing" in theme_map:
        theme = theme_map["pricing_billing"]
        insights.append(
            Insight(
                priority="high" if theme.avg_sentiment_score < 0 else "medium",
                category="risk" if theme.avg_sentiment_score < 0 else "observation",
                title="Subscription / billing is a recurring topic",
                detail=(
                    f"{theme.count} reviews mention pricing/subscription "
                    f"(avg sentiment {theme.avg_sentiment_score:+.2f})."
                ),
                evidence_count=theme.count,
                sample_quotes=_quotes_for(analyzed, theme="pricing_billing", limit=2),
            )
        )

    if "customer_support" in theme_map and theme_map["customer_support"].avg_sentiment_score < 0.1:
        theme = theme_map["customer_support"]
        insights.append(
            Insight(
                priority="high",
                category="risk",
                title="Support experience needs attention",
                detail=(
                    f"{theme.count} reviews mention support/customer service "
                    f"(avg sentiment {theme.avg_sentiment_score:+.2f})."
                ),
                evidence_count=theme.count,
                sample_quotes=_quotes_for(
                    analyzed, theme="customer_support", sentiment=SentimentLabel.NEGATIVE
                ),
            )
        )

    if "product_quality" in theme_map:
        theme = theme_map["product_quality"]
        tone = "strength" if theme.avg_sentiment_score >= 0.15 else "risk"
        insights.append(
            Insight(
                priority="medium",
                category=tone,
                title="Product/reading quality is frequently discussed",
                detail=(
                    f"{theme.count} reviews mention product/reading quality "
                    f"(avg sentiment {theme.avg_sentiment_score:+.2f})."
                ),
                evidence_count=theme.count,
                sample_quotes=_quotes_for(analyzed, theme="product_quality", limit=2),
            )
        )

    low_rated = [
        item
        for item in analyzed
        if _uses_rating(item.review) and item.review.rating is not None and item.review.rating <= 2
    ]
    if low_rated:
        insights.append(
            Insight(
                priority="high",
                category="risk",
                title="Low star ratings need response playbooks",
                detail=f"{len(low_rated)} reviews are rated 1–2 stars in this sample.",
                evidence_count=len(low_rated),
                sample_quotes=_quotes_for(low_rated, limit=2),
            )
        )

    if positive / max(len(reviews), 1) >= 0.6:
        insights.append(
            Insight(
                priority="medium",
                category="strength",
                title="Majority sentiment is positive",
                detail=(
                    f"{positive} of {len(reviews)} reviews are positive "
                    f"({100 * positive / len(reviews):.0f}%). Amplify these themes in marketing."
                ),
                evidence_count=positive,
                sample_quotes=_quotes_for(
                    analyzed, sentiment=SentimentLabel.POSITIVE, limit=2
                ),
            )
        )

    if not insights:
        insights.append(
            Insight(
                priority="medium",
                category="observation",
                title="No dominant risk pattern detected",
                detail="Continue monitoring weekly sentiment and keyword shifts.",
                evidence_count=len(reviews),
            )
        )

    improvement_areas = _improvement_areas(negative_terms)
    if negative_texts:
        improvement_areas.extend(
            area
            for area in _improvement_areas(negative_texts)
            if area not in improvement_areas
        )
    if low_rated and not any("Response ops" in a for a in improvement_areas):
        improvement_areas.append(
            "Response ops: reply to every new 1–2★ Trustpilot/App Store review within 24–48 hours."
        )

    actions: list[str] = list(improvement_areas)
    if low_rated and metrics.average_rating is not None and metrics.average_rating < 4.0:
        actions.append(
            "Run a weekly triage: classify negative / low-rated reviews into theme buckets."
        )
    if not actions:
        actions.append(
            "Keep a weekly review digest and track sentiment + negative-keyword deltas."
        )

    rating_part = ""
    if metrics.average_rating is not None and metrics.rating_basis == "native":
        rating_part = f" Average native rating: {metrics.average_rating}."
    summary = (
        f"NLP analysis of {metrics.total_reviews} reviews "
        f"({', '.join(f'{k}={v}' for k, v in metrics.sources.items())}). "
        f"Sentiment — positive: {positive}, neutral: {neutral}, negative: {negative}. "
        f"Overall tone: {overall.value} "
        f"(avg sentiment {metrics.average_sentiment_score:+.2f})."
        f"{rating_part}"
    )

    return InsightsResponse(
        summary=summary,
        overall_sentiment=overall,
        sentiment_distribution=metrics.sentiment_distribution,
        negative_keywords=negative_keywords,
        insights=insights[:8],
        recommended_actions=actions[:8],
        improvement_areas=improvement_areas[:8],
    )
