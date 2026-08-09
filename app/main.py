from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.collectors.app_store import AppStoreCollectError, collect_app_store_reviews
from app.collectors.dou import DouCollectError, collect_dou_reviews
from app.collectors.trustpilot import TrustpilotCollectError, collect_trustpilot_reviews
from app.models.schemas import (
    CollectRequest,
    CollectResponse,
    InsightsResponse,
    MetricsResponse,
    ProcessResponse,
    ProcessedReviewsResponse,
    RatingMetricsResponse,
    Review,
    ReviewSource,
    ReviewsResponse,
)
from app.services.analyzer import build_insights, compute_metrics
from app.services.metrics import calculate_rating_metrics, rating_metrics_to_dict
from app.services.processing import process_reviews
from app.storage import (
    processed_store,
    resolve_dataset,
    sample_by_source,
    store,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nebula / Obrio Review Analysis API",
    description=(
        "Collect and analyze reviews for **Nebula** (Trustpilot + App Store) "
        "and **Obrio** (DOU). Choose with `dataset`: "
        "`nebula`, `nebula_appstore`, `obrio`, or `all`."
    ),
    version="1.2.0",
)

DatasetQuery = Query(
    default="all",
    description="nebula | nebula_appstore | obrio | trustpilot | app_store | dou | all",
    examples=["nebula", "nebula_appstore", "obrio", "all"],
)


def _parse_dataset(dataset: str | None) -> str | None:
    try:
        return resolve_dataset(dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _reviews_for_analysis(dataset: str | None = "all") -> list[Review]:
    """Prefer cleaned processed reviews for the selected dataset."""
    source = _parse_dataset(dataset)
    processed = processed_store.filter_dataset(dataset)
    if processed:
        return [
            Review(
                id=item.id,
                source=item.source,
                title=item.title,
                text=item.text,
                rating=item.rating,
            )
            for item in processed
            if source is None or item.source.value == source
        ]
    return store.filter_dataset(dataset)


def _require_reviews(dataset: str | None = "all") -> list[Review]:
    reviews = _reviews_for_analysis(dataset)
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No reviews stored for dataset='{dataset or 'all'}'. "
                "Collect with POST /api/v1/collect "
                "(source=trustpilot|app_store|dou|both)."
            ),
        )
    return reviews


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/datasets", tags=["system"], summary="List available datasets")
def list_datasets() -> dict[str, object]:
    reviews = store.load()
    counts: dict[str, int] = {}
    for review in reviews:
        counts[review.source.value] = counts.get(review.source.value, 0) + 1
    return {
        "available": {
            "nebula": {
                "source": "trustpilot",
                "label": "Nebula (Trustpilot)",
                "url": "https://www.trustpilot.com/review/asknebula.com",
                "count": counts.get("trustpilot", 0),
            },
            "nebula_appstore": {
                "source": "app_store",
                "label": "Nebula (Apple App Store)",
                "url": "https://apps.apple.com/us/app/id1459969523",
                "app_id": "1459969523",
                "count": counts.get("app_store", 0),
            },
            "obrio": {
                "source": "dou",
                "label": "Obrio (DOU employer reviews)",
                "url": "https://jobs.dou.ua/companies/obrio/reviews/",
                "count": counts.get("dou", 0),
            },
        },
        "total": len(reviews),
        "choose_with": "dataset=nebula|nebula_appstore|obrio|all",
    }


@app.post(
    "/api/v1/collect",
    response_model=CollectResponse,
    tags=["collection"],
    summary="Collect Nebula and/or Obrio reviews",
)
def collect_reviews(request: CollectRequest) -> CollectResponse:
    """Collect reviews. Use source=both for Trustpilot + App Store + DOU."""
    source = request.source
    collected: list[Review] = []
    source_counts: dict[str, int] = {}
    errors: list[str] = []

    want_trustpilot = source in ("both", "all", ReviewSource.TRUSTPILOT, "trustpilot")
    want_dou = source in ("both", "all", ReviewSource.DOU, "dou")
    want_app_store = source in (
        "both",
        "all",
        ReviewSource.APP_STORE,
        "app_store",
    )

    if not want_trustpilot and not want_dou and not want_app_store:
        raise HTTPException(
            status_code=400,
            detail="source must be one of: trustpilot, dou, app_store, both",
        )

    if want_trustpilot:
        try:
            domain = request.resolved_trustpilot_domain()
            tp_reviews = collect_trustpilot_reviews(
                domain=domain,
                max_reviews=max(request.count, 100),
                random_sample=request.random_sample,
            )
            source_counts["trustpilot"] = len(tp_reviews)
            collected.extend(tp_reviews)
        except (TrustpilotCollectError, ValueError) as exc:
            errors.append(str(exc))
            logger.exception("Trustpilot collection failed")

    if want_dou:
        try:
            slug = request.resolved_company_slug()
            dou_reviews = collect_dou_reviews(company_slug=slug)
            source_counts["dou"] = len(dou_reviews)
            collected.extend(dou_reviews)
        except (DouCollectError, ValueError) as exc:
            errors.append(str(exc))
            logger.exception("DOU collection failed")

    if want_app_store:
        try:
            app_id = request.resolved_app_id()
            app_reviews = collect_app_store_reviews(
                app_id=app_id,
                country=request.country,
                max_reviews=max(request.count, 100),
            )
            source_counts["app_store"] = len(app_reviews)
            collected.extend(app_reviews)
        except (AppStoreCollectError, ValueError) as exc:
            errors.append(str(exc))
            logger.exception("App Store collection failed")

    if not collected:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Failed to collect reviews from all requested sources",
                "errors": errors or ["No reviews returned"],
            },
        )

    sampled = sample_by_source(
        collected,
        count_per_source=request.count,
        random_sample=request.random_sample,
    )
    stored = store.upsert_sources(sampled)
    processed = processed_store.replace(process_reviews(stored))

    message = (
        f"Upserted sources={list(source_counts)}. "
        f"Store now has {len(stored)} raw / {len(processed)} processed reviews. "
        "Analyze with dataset=nebula|nebula_appstore|obrio|all."
    )
    if errors:
        message += f" Warnings: {'; '.join(errors)}"

    return CollectResponse(
        collected=len(collected),
        stored=len(stored),
        processed=len(processed),
        sources=source_counts,
        message=message,
        collected_at=datetime.now(timezone.utc),
    )


@app.post(
    "/api/v1/process",
    response_model=ProcessResponse,
    tags=["collection"],
    summary="Re-process stored raw reviews",
)
def process_stored_reviews() -> ProcessResponse:
    reviews = store.load()
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail="No raw reviews stored. Call POST /api/v1/collect first.",
        )

    processed = process_reviews(reviews)
    processed_store.replace(processed)
    dropped = len(reviews) - len(processed)

    return ProcessResponse(
        input_count=len(reviews),
        processed=len(processed),
        dropped=dropped,
        message=(
            f"Extracted title/text/rating and cleaned text for {len(processed)} reviews"
            + (f" (dropped {dropped} empty)" if dropped else "")
        ),
    )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    tags=["analysis"],
    summary="Return calculated metrics for a dataset",
)
def metrics(dataset: str = DatasetQuery) -> MetricsResponse:
    return compute_metrics(_require_reviews(dataset))


@app.get(
    "/api/v1/metrics/ratings",
    response_model=RatingMetricsResponse,
    tags=["analysis"],
    summary="Average rating and star distribution",
)
def rating_metrics(dataset: str = DatasetQuery) -> RatingMetricsResponse:
    reviews = [
        review
        for review in _require_reviews(dataset)
        if review.source.value != "dou" and review.rating is not None
    ]
    result = calculate_rating_metrics(reviews, allow_sentiment_proxy=False)
    if not reviews:
        result.note = (
            "No native star ratings for this dataset "
            "(Obrio/DOU). Use /insights for NLP sentiment."
        )
    return RatingMetricsResponse.model_validate(rating_metrics_to_dict(result))


@app.get(
    "/api/v1/insights",
    response_model=InsightsResponse,
    tags=["analysis"],
    summary="Return NLP insights for a dataset",
)
def insights(dataset: str = DatasetQuery) -> InsightsResponse:
    return build_insights(_require_reviews(dataset))


@app.get(
    "/api/v1/analysis",
    tags=["analysis"],
    summary="Return metrics and insights for a dataset",
)
def analysis(dataset: str = DatasetQuery) -> dict[str, object]:
    reviews = _require_reviews(dataset)
    return {
        "dataset": dataset,
        "metrics": compute_metrics(reviews).model_dump(mode="json"),
        "insights": build_insights(reviews).model_dump(mode="json"),
    }


@app.get(
    "/api/v1/reviews",
    response_model=ReviewsResponse,
    tags=["reviews"],
    summary="List raw review data",
)
def list_reviews(
    dataset: str = DatasetQuery,
    source: str | None = Query(
        default=None,
        description="Deprecated alias of dataset: trustpilot | dou | app_store",
    ),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> ReviewsResponse:
    key = source or dataset
    try:
        resolve_dataset(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    reviews = store.filter_dataset(key)
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=f"No raw reviews for dataset='{key}'. Call POST /api/v1/collect.",
        )

    total = len(reviews)
    page = reviews[offset : offset + limit]
    return ReviewsResponse(total=total, returned=len(page), reviews=page)


@app.get(
    "/api/v1/reviews/download",
    tags=["reviews"],
    summary="Download raw review data",
)
def download_reviews(
    dataset: str = DatasetQuery,
    format: str = Query(default="json", pattern="^(json|csv)$"),
) -> StreamingResponse:
    try:
        resolve_dataset(dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    reviews = store.filter_dataset(dataset)
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=f"No raw reviews for dataset='{dataset}'.",
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = [review.model_dump(mode="json", exclude_none=True) for review in reviews]
    safe_name = (dataset or "all").replace(".", "_")

    if format == "csv":
        fieldnames = [
            "id",
            "source",
            "author",
            "title",
            "text",
            "rating",
            "date",
            "country",
            "url",
            "app_id",
            "version",
            "role",
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        payload = buffer.getvalue()
        filename = f"reviews_{safe_name}_{timestamp}.csv"
        media_type = "text/csv"
    else:
        payload = json.dumps(rows, ensure_ascii=False, indent=2)
        filename = f"reviews_{safe_name}_{timestamp}.json"
        media_type = "application/json"

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(
        iter([payload]),
        media_type=media_type,
        headers=headers,
    )


@app.get(
    "/api/v1/processed",
    response_model=ProcessedReviewsResponse,
    tags=["reviews"],
    summary="List processed review data",
)
def list_processed_reviews(
    dataset: str = DatasetQuery,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> ProcessedReviewsResponse:
    try:
        resolve_dataset(dataset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    reviews = processed_store.filter_dataset(dataset)
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=f"No processed reviews for dataset='{dataset}'.",
        )

    total = len(reviews)
    page = reviews[offset : offset + limit]
    return ProcessedReviewsResponse(total=total, returned=len(page), reviews=page)


@app.get("/", tags=["system"])
def root() -> dict[str, object]:
    return {
        "name": "Nebula / Obrio Review Analysis API",
        "docs": "/docs",
        "datasets": {
            "nebula": "Trustpilot reviews for asknebula.com",
            "obrio": "DOU employer reviews for OBRIO",
            "all": "Both datasets combined",
        },
        "endpoints": {
            "datasets": "GET /api/v1/datasets",
            "collect": "POST /api/v1/collect",
            "metrics": "GET /api/v1/metrics?dataset=nebula|obrio|all",
            "insights": "GET /api/v1/insights?dataset=nebula|obrio|all",
            "analysis": "GET /api/v1/analysis?dataset=nebula|obrio|all",
            "reviews": "GET /api/v1/reviews?dataset=nebula|obrio|all",
            "download": "GET /api/v1/reviews/download?dataset=nebula&format=json|csv",
        },
    }
