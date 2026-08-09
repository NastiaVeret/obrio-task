"""Streamlit UI for Nebula / Obrio review analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.schemas import Review
from app.services.analyzer import build_insights, compute_metrics
from app.services.metrics import calculate_rating_metrics, rating_metrics_to_dict
from app.storage import processed_store, resolve_dataset, store


def _install_playwright_chromium() -> None:
    """Download Chromium into Playwright's default cache (needed on Streamlit Cloud)."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            if Path(playwright.chromium.executable_path).exists():
                return
    except Exception:  # noqa: BLE001
        pass

    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=False,
        timeout=600,
    )

DATASETS = {
    "nebula": "Nebula · Trustpilot",
    "nebula_appstore": "Nebula · App Store",
    "obrio": "Obrio · DOU",
    "all": "All sources",
}

SENTIMENT_COLORS = {
    "positive": "#2a9d8f",
    "neutral": "#8a8f98",
    "negative": "#c45c26",
}
STAR_COLORS = ["#c45c26", "#d4894a", "#c4a35a", "#6f9e6f", "#2a9d8f"]


st.set_page_config(
    page_title="Nebula / Obrio Reviews",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600&display=swap');
      :root {
        --ink: #1c2430;
        --muted: #5c6673;
        --bg: #f3efe6;
        --panel: #fffdf8;
        --line: #d9d2c5;
        --accent: #1f6f6a;
        --warm: #c45c26;
      }
      html, body, [class*="css"] {
        font-family: "Source Sans 3", sans-serif;
        color: var(--ink);
      }
      .stApp {
        background:
          radial-gradient(1000px 480px at 10% -10%, #e7f0ea 0%, transparent 55%),
          radial-gradient(900px 420px at 100% 0%, #f3e4d4 0%, transparent 50%),
          linear-gradient(180deg, #f7f3eb 0%, #efe8db 100%);
      }
      h1, h2, h3, .brand {
        font-family: "Fraunces", Georgia, serif !important;
        letter-spacing: -0.02em;
      }
      .brand {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 0.2rem 0;
        color: var(--ink);
      }
      .lede {
        color: var(--muted);
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
      }
      div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.75rem 1rem;
      }
      div[data-testid="stMetric"] label { color: var(--muted); }
      .insight-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
      }
      .insight-card.risk { border-left-color: var(--warm); }
      .insight-card .meta {
        font-size: 0.8rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .quote {
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0.35rem 0 0 0;
        padding-left: 0.6rem;
        border-left: 2px solid var(--line);
      }
      section[data-testid="stSidebar"] {
        background: #f8f4ec;
        border-right: 1px solid var(--line);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def dataset_counts() -> dict[str, int]:
    counts: dict[str, int] = {"trustpilot": 0, "app_store": 0, "dou": 0}
    for review in store.load():
        counts[review.source.value] = counts.get(review.source.value, 0) + 1
    return counts


def reviews_for(dataset: str) -> list[Review]:
    source = resolve_dataset(dataset)
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


@st.cache_data(show_spinner="Analyzing reviews…")
def analyze(dataset: str) -> dict:
    reviews = reviews_for(dataset)
    if not reviews:
        return {"empty": True, "count": 0}

    metrics = compute_metrics(reviews)
    insights = build_insights(reviews)
    rated = [
        r for r in reviews
        if getattr(r.source, "value", r.source) != "dou" and r.rating is not None
    ]
    rating = rating_metrics_to_dict(
        calculate_rating_metrics(rated, allow_sentiment_proxy=False)
    )
    if not rated:
        rating["note"] = "No star ratings (Obrio/DOU). Insights use NLP sentiment only."
    rows = [
        {
            "id": r.id,
            "source": r.source.value if hasattr(r.source, "value") else r.source,
            "title": r.title,
            "text": r.text,
            "rating": r.rating,
        }
        for r in reviews
    ]
    return {
        "empty": False,
        "count": len(reviews),
        "metrics": metrics.model_dump(mode="json"),
        "insights": insights.model_dump(mode="json"),
        "rating": rating,
        "reviews": rows,
    }


with st.sidebar:
    st.markdown("### Dataset")
    dataset = st.selectbox(
        "Analyze",
        options=list(DATASETS.keys()),
        format_func=lambda key: DATASETS[key],
        index=0,
    )
    counts = dataset_counts()
    st.caption(
        f"Stored — Trustpilot {counts.get('trustpilot', 0)} · "
        f"App Store {counts.get('app_store', 0)} · "
        f"DOU {counts.get('dou', 0)}"
    )
    if st.button("Refresh analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Collect")
    collect_source = st.selectbox(
        "Source",
        options=["both", "trustpilot", "app_store", "dou"],
        help="Upserts into the local JSON store",
    )
    collect_count = st.slider("Max per source", 20, 200, 100, 10)
    if st.button("Collect reviews", use_container_width=True, type="primary"):
        with st.spinner("Collecting…"):
            try:
                from app.collectors.app_store import collect_app_store_reviews
                from app.collectors.dou import collect_dou_reviews
                from app.collectors.trustpilot import collect_trustpilot_reviews
                from app.services.processing import process_reviews
                from app.storage import sample_by_source

                gathered: list[Review] = []
                if collect_source in {"both", "trustpilot"}:
                    with st.spinner(
                        "Installing Chromium for Trustpilot (first run can take 1–2 min)…"
                    ):
                        _install_playwright_chromium()
                    gathered.extend(
                        collect_trustpilot_reviews(
                            domain="asknebula.com",
                            max_reviews=max(collect_count, 100),
                        )
                    )
                if collect_source in {"both", "app_store"}:
                    gathered.extend(
                        collect_app_store_reviews(
                            app_id="1459969523",
                            max_reviews=max(collect_count, 100),
                        )
                    )
                if collect_source in {"both", "dou"}:
                    gathered.extend(collect_dou_reviews(company_slug="obrio"))

                sampled = sample_by_source(
                    gathered, count_per_source=collect_count, random_sample=True
                )
                stored = store.upsert_sources(sampled)
                processed_store.replace(process_reviews(stored))
                st.cache_data.clear()
                st.success(f"Store now has {len(stored)} reviews")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    st.markdown("---")
    st.caption("API docs: http://127.0.0.1:8000/docs")


st.markdown('<p class="brand">Nebula / Obrio</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="lede">Review metrics and NLP insights · {DATASETS[dataset]}</p>',
    unsafe_allow_html=True,
)

payload = analyze(dataset)
if payload["empty"]:
    st.warning("No reviews for this dataset. Collect from the sidebar first.")
    st.stop()

metrics = payload["metrics"]
insights = payload["insights"]
rating = payload["rating"]
reviews_df = pd.DataFrame(payload["reviews"])

has_ratings = rating.get("average_rating") is not None
if has_ratings:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reviews", payload["count"])
    m2.metric("Avg rating", f"{rating['average_rating']:.2f}")
    m3.metric("Sentiment", insights["overall_sentiment"])
    m4.metric("Avg sentiment", f"{metrics['average_sentiment_score']:+.2f}")
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Reviews", payload["count"])
    m2.metric("Sentiment", insights["overall_sentiment"])
    m3.metric("Avg sentiment", f"{metrics['average_sentiment_score']:+.2f}")
st.caption(rating.get("note") or metrics.get("rating_note") or "")

tab_overview, tab_insights, tab_reviews = st.tabs(
    ["Overview", "Insights", "Reviews"]
)

with tab_overview:
    left, right = st.columns(2)

    sent_df = pd.DataFrame(insights["sentiment_distribution"])
    if not sent_df.empty:
        fig = px.bar(
            sent_df,
            x="label",
            y="percentage",
            color="label",
            color_discrete_map=SENTIMENT_COLORS,
            text="percentage",
            title="Sentiment distribution (%)",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            showlegend=False,
            yaxis_title="% of reviews",
            xaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=48, b=24),
        )
        left.plotly_chart(fig, use_container_width=True)

    rating_df = pd.DataFrame(rating.get("rating_distribution") or [])
    rated_rows = rating_df[rating_df["count"] > 0] if not rating_df.empty else rating_df
    if not rated_rows.empty:
        rating_df = rated_rows
        rating_df["stars"] = rating_df["rating"].astype(str) + "★"
        color_map = {
            row["stars"]: STAR_COLORS[int(row["rating"]) - 1] for _, row in rating_df.iterrows()
        }
        fig = px.bar(
            rating_df,
            x="stars",
            y="percentage",
            color="stars",
            color_discrete_map=color_map,
            text="percentage",
            title="Rating distribution (%)",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            showlegend=False,
            yaxis_title="% of reviews",
            xaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=48, b=24),
        )
        right.plotly_chart(fig, use_container_width=True)
    else:
        right.info("No star ratings for this dataset. Obrio insights are NLP-only (sentiment + keywords).")

    if metrics.get("top_themes"):
        theme_df = pd.DataFrame(metrics["top_themes"])
        fig = px.bar(
            theme_df.sort_values("count"),
            x="count",
            y="theme",
            orientation="h",
            title="Top themes",
            color_discrete_sequence=["#1f6f6a"],
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Mentions",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    neg = insights.get("negative_keywords") or []
    if neg:
        neg_df = pd.DataFrame(neg).sort_values("count").tail(12)
        fig = px.bar(
            neg_df,
            x="count",
            y="term",
            orientation="h",
            title="Keywords in negative reviews",
            color_discrete_sequence=["#c45c26"],
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Mentions",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_insights:
    st.write(insights.get("summary", ""))
    for item in insights.get("insights") or []:
        css = "insight-card risk" if item.get("category") == "risk" else "insight-card"
        quotes_html = "".join(
            f'<p class="quote">“{q}”</p>' for q in (item.get("sample_quotes") or [])[:2]
        )
        st.markdown(
            f"""
            <div class="{css}">
              <div class="meta">{item.get("priority", "")} · {item.get("category", "")}</div>
              <h4 style="margin:0.25rem 0 0.35rem 0;">{item.get("title", "")}</h4>
              <div>{item.get("detail", "")}</div>
              {quotes_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    areas = insights.get("improvement_areas") or insights.get("recommended_actions") or []
    if areas:
        st.subheader("Improvement areas")
        for area in areas:
            st.markdown(f"- {area}")

with tab_reviews:
    sources = sorted(reviews_df["source"].dropna().unique().tolist())
    source_filter = st.multiselect("Filter source", sources, default=sources)
    q = st.text_input("Search text", placeholder="payment, recruiter, crash…")
    view = reviews_df[reviews_df["source"].isin(source_filter)].copy()
    if q.strip():
        mask = view["text"].fillna("").str.contains(q.strip(), case=False, na=False)
        if "title" in view.columns:
            mask = mask | view["title"].fillna("").str.contains(q.strip(), case=False, na=False)
        view = view[mask]
    st.dataframe(
        view[["source", "rating", "title", "text"]],
        use_container_width=True,
        height=480,
    )
    st.download_button(
        "Download filtered CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name=f"reviews_{dataset}.csv",
        mime="text/csv",
    )
