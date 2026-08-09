#!/usr/bin/env python3
"""Rewrite notebooks: imports only in first code cell, strip noise."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_eda() -> None:
    cells = [
        md(
            """# Nebula / Obrio Review Analysis

Datasets: `nebula` (Trustpilot), `nebula_appstore`, `obrio` (DOU), or `all`.

```bash
python scripts/collect_both.py --count 100
```
"""
        ),
        md("## 0. Setup & load data"),
        code(
            """import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.schemas import Review
from app.services.analyzer import build_insights, compute_metrics
from app.services.metrics import calculate_rating_metrics, rating_metrics_to_dict
from app.services.sentiment import analyze_text
from app.storage import resolve_dataset

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 5)

DATASET = "nebula"  # nebula | nebula_appstore | obrio | all
RAW_PATH = ROOT / "data" / "reviews.json"
PROCESSED_PATH = ROOT / "data" / "processed_reviews.json"
source_filter = resolve_dataset(DATASET)

raw_all = pd.read_json(RAW_PATH)
df_all = pd.read_json(PROCESSED_PATH)
for frame in (raw_all, df_all):
    for col, default in [("title", None), ("rating", None), ("title_raw", None), ("text_raw", "")]:
        if col not in frame.columns:
            frame[col] = default

if source_filter is None:
    raw_df, df = raw_all.copy(), df_all.copy()
else:
    raw_df = raw_all[raw_all["source"] == source_filter].copy()
    df = df_all[df_all["source"] == source_filter].copy()

if df.empty:
    raise ValueError(f"No reviews for DATASET={DATASET!r}. Run: python scripts/collect_both.py")

reviews = [
    Review(
        id=row["id"],
        source=row["source"],
        title=row["title"] if pd.notna(row.get("title")) else None,
        text=row["text"],
        rating=float(row["rating"]) if pd.notna(row.get("rating")) else None,
    )
    for _, row in df.iterrows()
]

print(df_all["source"].value_counts().to_dict())
print(f"DATASET={DATASET!r}: {len(df)} processed / {len(raw_df)} raw")
display(df.head())
"""
        ),
        md("## 1. Key fields"),
        code(
            """cols = [c for c in ["id", "source", "title", "text", "rating"] if c in df.columns]
key_fields = df[cols].copy()
if "title" not in key_fields.columns:
    key_fields["title"] = None
if "rating" not in key_fields.columns:
    key_fields["rating"] = pd.NA

key_fields["has_title"] = key_fields["title"].notna() & (key_fields["title"].astype(str).str.len() > 0)
key_fields["has_rating"] = key_fields["rating"].notna()
key_fields["text_len"] = key_fields["text"].fillna("").str.len()

summary = (
    key_fields.groupby("source")
    .agg(
        reviews=("id", "count"),
        with_title=("has_title", "sum"),
        with_rating=("has_rating", "sum"),
        avg_text_len=("text_len", "mean"),
        avg_rating=("rating", "mean"),
    )
    .round(2)
)
display(summary)
display(key_fields.head(10))
"""
        ),
        md("## 2. Source mix"),
        code(
            """source_counts = df["source"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ["#2a6f97", "#e76f51", "#2a9d8f"]

source_counts.plot(kind="bar", ax=axes[0], color=colors[: len(source_counts)], rot=0)
axes[0].set_title("Reviews by source")
axes[0].set_xlabel("")
axes[0].set_ylabel("Count")

axes[1].pie(
    source_counts,
    labels=source_counts.index,
    autopct="%1.1f%%",
    colors=colors[: len(source_counts)],
    startangle=90,
)
axes[1].set_title("Source share")
plt.tight_layout()
plt.show()
"""
        ),
        md("## 3. Rating metrics"),
        code(
            """rating_metrics = calculate_rating_metrics(reviews, allow_sentiment_proxy=True)
metrics_dict = rating_metrics_to_dict(rating_metrics)
dist_df = pd.DataFrame(metrics_dict["rating_distribution"])
dist_df["stars"] = dist_df["rating"].astype(str) + "★"

print(f"Average rating: {metrics_dict['average_rating']} ({metrics_dict['rating_basis']})")
print(metrics_dict["note"])
display(dist_df[["stars", "count", "percentage"]])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
star_colors = ["#e76f51", "#f4a261", "#e9c46a", "#90be6d", "#2a9d8f"]
avg = metrics_dict["average_rating"] or 0
axes[0].bar(["Average rating"], [avg], color="#2a6f97", width=0.5)
axes[0].set_ylim(0, 5)
axes[0].set_title(f"Average rating = {avg:.2f}")

axes[1].bar(dist_df["stars"], dist_df["percentage"], color=star_colors)
axes[1].set_ylabel("% of reviews")
axes[1].set_title("Rating distribution (%)")
for i, row in dist_df.iterrows():
    axes[1].text(i, row["percentage"] + 1, f"{row['percentage']}%", ha="center", fontsize=9)
plt.tight_layout()
plt.show()
"""
        ),
        md("## 4. Text length"),
        code(
            """df["text_len"] = df["text"].fillna("").str.len()
if "text_raw" in df.columns:
    df["chars_removed"] = df["text_raw"].fillna("").str.len() - df["text_len"]
    display(df["chars_removed"].describe().to_frame().T)

fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=df, x="source", y="text_len", ax=ax, palette="Set2")
ax.set_title("Cleaned text length by source")
plt.tight_layout()
plt.show()
"""
        ),
        md("## 5. Sentiment & themes"),
        code(
            """def enrich_row(row):
    rating = row["rating"] if "rating" in row.index and pd.notna(row["rating"]) else None
    label, score, themes = analyze_text(row["text"], rating)
    return pd.Series({
        "sentiment": label.value,
        "sentiment_score": score,
        "themes": themes,
    })

enriched = df.join(df.apply(enrich_row, axis=1))
palette = {"positive": "#2a9d8f", "neutral": "#bdbdbd", "negative": "#e76f51"}
order = ["positive", "neutral", "negative"]

display(enriched[["source", "title", "rating", "sentiment", "sentiment_score", "themes"]].head(10))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sent_counts = enriched["sentiment"].value_counts().reindex(order, fill_value=0)
sns.barplot(x=sent_counts.index, y=sent_counts.values, ax=axes[0], palette=palette)
axes[0].set_title("Sentiment distribution")
sns.boxplot(data=enriched, x="source", y="sentiment_score", ax=axes[1], palette="pastel")
axes[1].axhline(0, color="gray", ls="--", lw=1)
axes[1].set_title("Sentiment score by source")
plt.tight_layout()
plt.show()
"""
        ),
        code(
            """theme_rows = enriched.explode("themes").dropna(subset=["themes"])
if theme_rows.empty:
    print("No themes detected.")
else:
    theme_counts = theme_rows["themes"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    theme_counts.plot(kind="barh", ax=ax, color="#264653")
    ax.set_title("Theme mentions")
    plt.tight_layout()
    plt.show()
"""
        ),
        code(
            """rated = enriched[enriched["rating"].notna()].copy() if "rating" in enriched.columns else pd.DataFrame()
if rated.empty:
    print("No native star ratings to plot.")
else:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=rated, x="rating", y="sentiment_score", hue="sentiment",
        palette=palette, s=70, ax=ax,
    )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("Star rating vs sentiment score")
    plt.tight_layout()
    plt.show()
"""
        ),
        md("## 6. Sample reviews"),
        code(
            """preview = enriched[["source", "title", "rating", "sentiment", "sentiment_score", "text"]].copy()
preview["text"] = preview["text"].str.replace("\\n", " ", regex=False).str.slice(0, 120) + "..."
display(preview.sort_values("sentiment_score", ascending=False).head(5))
display(preview.sort_values("sentiment_score", ascending=True).head(5))
"""
        ),
        md("## 7. Insights"),
        code(
            """insights = build_insights(reviews)
metrics = compute_metrics(reviews)
sent_df = pd.DataFrame([s.model_dump() for s in insights.sentiment_distribution])
rating_df = pd.DataFrame([r.model_dump() for r in metrics.rating_distribution])
neg_df = pd.DataFrame([k.model_dump() for k in insights.negative_keywords])

print(insights.summary)
display(sent_df)
display(neg_df.head(15) if not neg_df.empty else neg_df)

for item in insights.insights:
    print(f"\\n[{item.priority} | {item.category}] {item.title}")
    print(f"  {item.detail}")

print("\\nImprovement areas:")
for area in insights.improvement_areas or insights.recommended_actions:
    print(f"- {area}")

out = ROOT / "data" / "insights.json"
out.write_text(json.dumps(insights.model_dump(mode="json"), ensure_ascii=False, indent=2))
print(f"Saved {out}")
"""
        ),
        code(
            """sentiment_colors = {"positive": "#2a9d8f", "neutral": "#bdbdbd", "negative": "#e76f51"}
rating_colors = ["#e76f51", "#f4a261", "#e9c46a", "#90be6d", "#2a9d8f"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"Insights | {insights.overall_sentiment.value} | avg rating={metrics.average_rating}",
    fontsize=13,
    fontweight="bold",
)

axes[0, 0].bar(sent_df["label"], sent_df["percentage"], color=[sentiment_colors[x] for x in sent_df["label"]])
axes[0, 0].set_title("Sentiment %")
pie_sent = sent_df[sent_df["count"] > 0]
axes[0, 1].pie(
    pie_sent["count"], labels=pie_sent["label"], autopct="%1.1f%%",
    colors=[sentiment_colors[x] for x in pie_sent["label"]], startangle=90,
)
axes[0, 1].set_title("Sentiment share")

axes[1, 0].bar(rating_df["rating"].astype(str) + "★", rating_df["percentage"], color=rating_colors)
axes[1, 0].set_title("Rating %")
pie_rating = rating_df[rating_df["count"] > 0]
axes[1, 1].pie(
    pie_rating["count"],
    labels=pie_rating["rating"].astype(str) + "★",
    autopct="%1.1f%%",
    colors=[rating_colors[int(r) - 1] for r in pie_rating["rating"]],
    startangle=90,
)
axes[1, 1].set_title("Rating share")
plt.tight_layout()

charts_dir = ROOT / "data" / "charts"
charts_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(charts_dir / "insights_overview.png", dpi=150, bbox_inches="tight")
plt.show()

if not neg_df.empty:
    plot_df = neg_df.sort_values("count", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["term"], plot_df["count"], color="#e76f51")
    ax.set_title("Negative keywords / phrases")
    plt.tight_layout()
    fig.savefig(charts_dir / "negative_keywords.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
        ),
        md("## 8. Live API (optional)"),
        code(
            """API = "http://127.0.0.1:8000"
try:
    with urlopen(f"{API}/api/v1/metrics?dataset={DATASET}", timeout=3) as resp:
        payload = json.load(resp)
    print(json.dumps({
        "total_reviews": payload["total_reviews"],
        "sources": payload["sources"],
        "average_rating": payload["average_rating"],
        "average_sentiment_score": payload["average_sentiment_score"],
    }, indent=2))
except URLError as exc:
    print(f"API not reachable at {API}: {exc}")
"""
        ),
    ]
    path = ROOT / "notebooks" / "review_analysis_eda.ipynb"
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {path} ({len(cells)} cells)")


def write_nlp_compare() -> None:
    import runpy
    runpy.run_path(str(ROOT / "scripts" / "build_nlp_comparison_nb.py"))


if __name__ == "__main__":
    write_eda()
    write_nlp_compare()
