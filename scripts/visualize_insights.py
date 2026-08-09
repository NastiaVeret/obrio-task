#!/usr/bin/env python3
"""Visualize sentiment and rating distributions for a chosen dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplconfig"),
)

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.schemas import Review
from app.services.analyzer import build_insights, compute_metrics
from app.storage import ProcessedReviewStore, ReviewStore, resolve_dataset

SENTIMENT_COLORS = {
    "positive": "#2a9d8f",
    "neutral": "#bdbdbd",
    "negative": "#e76f51",
}
RATING_COLORS = ["#e76f51", "#f4a261", "#e9c46a", "#90be6d", "#2a9d8f"]


def load_reviews(path: Path, dataset: str) -> list[Review]:
    source = resolve_dataset(dataset)
    if path.name.startswith("processed") and path.exists():
        items = ProcessedReviewStore(path).load()
        reviews = [
            Review(
                id=item.id,
                source=item.source,
                title=item.title,
                text=item.text,
                rating=item.rating,
            )
            for item in items
        ]
    else:
        reviews = ReviewStore(
            path if path.exists() else ROOT / "data" / "reviews.json"
        ).load()

    if source is None:
        return reviews
    return [r for r in reviews if r.source.value == source]


def make_overview_chart(
    sentiment_df: pd.DataFrame,
    rating_df: pd.DataFrame,
    *,
    title: str,
    average_rating: float | None,
    overall_sentiment: str,
    output: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"{title}  |  overall={overall_sentiment}"
        + (f"  |  avg rating={average_rating:.2f}" if average_rating is not None else ""),
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0, 0]
    colors = [SENTIMENT_COLORS.get(label, "#999") for label in sentiment_df["label"]]
    ax.bar(sentiment_df["label"], sentiment_df["percentage"], color=colors)
    ax.set_title("Sentiment distribution (%)")
    ax.set_ylabel("% of reviews")
    ax.set_ylim(0, max(100, float(sentiment_df["percentage"].max()) + 15))
    for i, row in sentiment_df.iterrows():
        ax.text(
            i,
            row["percentage"] + 1.5,
            f"{row['percentage']}%\n({int(row['count'])})",
            ha="center",
            fontsize=9,
        )

    ax = axes[0, 1]
    pie_sent = sentiment_df[sentiment_df["count"] > 0]
    ax.pie(
        pie_sent["count"],
        labels=pie_sent["label"],
        autopct="%1.1f%%",
        colors=[SENTIMENT_COLORS.get(label, "#999") for label in pie_sent["label"]],
        startangle=90,
    )
    ax.set_title("Sentiment share")

    ax = axes[1, 0]
    ax.bar(
        rating_df["rating"].astype(str) + "★",
        rating_df["percentage"],
        color=RATING_COLORS,
    )
    ax.set_title("Rating distribution (%)")
    ax.set_ylabel("% of reviews")
    ax.set_ylim(0, max(100, float(rating_df["percentage"].max()) + 15))
    for i, row in rating_df.iterrows():
        ax.text(
            i,
            row["percentage"] + 1.5,
            f"{row['percentage']}%\n({int(row['count'])})",
            ha="center",
            fontsize=9,
        )

    ax = axes[1, 1]
    pie_rating = rating_df[rating_df["count"] > 0]
    if pie_rating.empty:
        ax.text(0.5, 0.5, "No star ratings", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.pie(
            pie_rating["count"],
            labels=pie_rating["rating"].astype(str) + "★",
            autopct="%1.1f%%",
            colors=[RATING_COLORS[int(r) - 1] for r in pie_rating["rating"]],
            startangle=90,
        )
    ax.set_title("Rating share")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def make_keywords_chart(keywords_df: pd.DataFrame, output: Path) -> Path | None:
    if keywords_df.empty:
        return None
    plot_df = keywords_df.sort_values("count", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["term"], plot_df["count"], color="#e76f51")
    ax.set_title("Top terms in negative reviews")
    ax.set_xlabel("Mentions")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sentiment/rating distribution charts"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed_reviews.json",
    )
    parser.add_argument(
        "--dataset",
        default="nebula",
        help="nebula | nebula_appstore | obrio | all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Overview chart path (default: data/charts/<dataset>_insights_overview.png)",
    )
    parser.add_argument(
        "--insights-json",
        type=Path,
        default=None,
        help="Insights JSON path (default: data/insights_<dataset>.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    safe = dataset.replace("/", "_")
    output = args.output or (ROOT / "data" / "charts" / f"{safe}_insights_overview.png")
    insights_json = args.insights_json or (ROOT / "data" / f"insights_{safe}.json")
    keywords_path = output.with_name(f"{safe}_negative_keywords.png")

    reviews = load_reviews(args.input, dataset)
    if not reviews:
        print(f"No reviews found for dataset={dataset!r} at {args.input}", file=sys.stderr)
        return 1

    metrics = compute_metrics(reviews)
    insights = build_insights(reviews)

    sentiment_df = pd.DataFrame(
        [item.model_dump() for item in insights.sentiment_distribution]
    )
    rating_df = pd.DataFrame(
        [item.model_dump() for item in metrics.rating_distribution]
    )
    keywords_df = pd.DataFrame(
        [item.model_dump() for item in insights.negative_keywords]
    )

    insights_json.parent.mkdir(parents=True, exist_ok=True)
    insights_json.write_text(
        json.dumps(insights.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    titles = {
        "nebula": "Nebula (Trustpilot)",
        "nebula_appstore": "Nebula (App Store)",
        "obrio": "Obrio (DOU)",
        "all": "All sources",
    }
    chart_path = make_overview_chart(
        sentiment_df,
        rating_df,
        title=titles.get(dataset, dataset),
        average_rating=metrics.average_rating,
        overall_sentiment=insights.overall_sentiment.value,
        output=output,
    )
    kw_path = make_keywords_chart(keywords_df, keywords_path)

    print(
        json.dumps(
            {
                "dataset": dataset,
                "chart": str(chart_path),
                "keywords_chart": str(kw_path) if kw_path else None,
                "insights_json": str(insights_json),
                "average_rating": metrics.average_rating,
                "overall_sentiment": insights.overall_sentiment.value,
                "summary": insights.summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
