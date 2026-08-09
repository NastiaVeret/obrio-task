# Nebula / Obrio Review Analysis API

Collect, clean, and analyze public reviews for:

| Dataset | Source | URL |
|---------|--------|-----|
| `nebula` | Trustpilot | https://www.trustpilot.com/review/asknebula.com |
| `nebula_appstore` | Apple App Store | https://apps.apple.com/us/app/id1459969523 |
| `obrio` | DOU employer reviews | https://jobs.dou.ua/companies/obrio/reviews/ |

This README covers **running the API**, the **analysis approach**, and a **sample insights report** (with charts).

---

## 1. Run the API locally

### Prerequisites

- Python 3.11+ recommended
- Chromium for Trustpilot collection (Playwright)

### Install

```bash
cd /path/to/new-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Optional (Obrio Ukrainian spaCy model):

```bash
python -m spacy download uk_core_news_sm
```

Fine-tune English RoBERTa (after collecting rated Nebula reviews):

```bash
python scripts/train_roberta_sentiment.py
# → data/models/roberta_sentiment
```

Optional LLM notebook comparisons: set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env`.

### Start the server

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

- Interactive docs: http://127.0.0.1:8000/docs  
- Health check: http://127.0.0.1:8000/health  
- Smoke test: `python scripts/test_api.py`

### Collect reviews (if `data/` is empty)

```bash
# Trustpilot + App Store + DOU (up to 100 per source)
curl -s -X POST http://127.0.0.1:8000/api/v1/collect \
  -H 'Content-Type: application/json' \
  -d '{"source":"both","count":100}'
```

Or from the CLI:

```bash
python scripts/collect_both.py --count 100
```

### Query metrics & insights

```bash
# Nebula (Trustpilot)
curl -s "http://127.0.0.1:8000/api/v1/metrics?dataset=nebula" | python3 -m json.tool
curl -s "http://127.0.0.1:8000/api/v1/insights?dataset=nebula" | python3 -m json.tool
curl -s "http://127.0.0.1:8000/api/v1/analysis?dataset=nebula" | python3 -m json.tool

# Obrio (DOU) — text/sentiment only (no native stars)
curl -s "http://127.0.0.1:8000/api/v1/insights?dataset=obrio" | python3 -m json.tool
```

### Main endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/collect` | Collect reviews (`trustpilot` / `app_store` / `dou` / `both`) |
| `GET` | `/api/v1/metrics` | Counts, rating + sentiment distributions, themes |
| `GET` | `/api/v1/insights` | Summary, negative keywords, actions |
| `GET` | `/api/v1/analysis` | Metrics + insights together |
| `GET` | `/api/v1/reviews` | List raw reviews |
| `GET` | `/api/v1/reviews/download` | Download `json` or `csv` |
| `GET` | `/api/v1/processed` | List cleaned reviews |
| `GET` | `/api/v1/metrics/ratings` | Average rating + star % |
| `POST` | `/api/v1/process` | Re-run text cleaning |
| `GET` | `/api/v1/datasets` | Available datasets + counts |
| `GET` | `/health` | Health check |

All analysis routes accept `dataset=nebula|nebula_appstore|obrio|all`.

### UI alternatives

```bash
streamlit run streamlit_app.py
jupyter notebook notebooks/review_analysis_eda.ipynb
```

---

## 2. Approach & design decisions

### Pipeline

```text
Collectors (Trustpilot / App Store / DOU)
        ↓
  data/reviews.json          (raw, upserted by source)
        ↓
  text cleaning              (title / text / rating)
        ↓
  data/processed_reviews.json
        ↓
  metrics + NLP insights     (API / scripts / notebooks)
        ↓
  data/insights_*.json + charts
```

### Key decisions

1. **Multi-source, one store**  
   Reviews are upserted **per source**, so Nebula Trustpilot, Nebula App Store, and Obrio DOU can coexist. Callers pick a dataset with `?dataset=…`.

2. **Native stars vs text-only**  
   Trustpilot and App Store expose 1–5 ratings; DOU does not. Rating metrics never invent stars for Obrio — use sentiment + keywords instead. Optional sentiment→star proxy exists only for exploratory notebook charts.

3. **Sentiment stack (chosen from notebook bake-offs)**  
   - **Nebula (EN):** fine-tuned **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest` → `data/models/roberta_sentiment`), best accuracy in the NLP methods comparison. Falls back to VADER+lexicon if the checkpoint is missing.  
   - **Obrio (UK):** **spaCy `uk_core_news_sm` lemmas + lexicon** — highest agreement among non-trivial UA methods (0.93), keeps negative detections (UKR-RoBERTa FT collapsed to 0% negative).  
   - When a native rating exists (Trustpilot / App Store), it is lightly blended with the text score.

4. **Insights from negative language**  
   Negative-sentiment reviews are mined for keywords/phrases; rule maps turn recurring terms into concrete improvement areas (billing, trust, support, etc.).

5. **FastAPI surface**  
   Collection, metrics, insights, and download are separate endpoints so the same backend powers curl, Streamlit, and notebooks without duplicating logic.

6. **Notebooks are experiments; production uses the winners**  
   Method comparisons live in notebooks. Production calls the winning pipelines above. Retrain EN RoBERTa after collecting new rated reviews:

   ```bash
   python scripts/train_roberta_sentiment.py
   python -m spacy download uk_core_news_sm   # once, for Obrio
   ```

### Project layout

```text
app/
  collectors/     Trustpilot, App Store, DOU
  services/       processing, sentiment, metrics, keywords, analyzer
  models/         Pydantic schemas
  main.py         FastAPI app
scripts/          CLI collect / process / insights / visualize
notebooks/        EDA + NLP method comparisons
docs/             Sample report
data/             reviews, insights, charts
streamlit_app.py  Local dashboard
```

---

## 3. Sample report (Nebula)

Full write-up with charts: **[docs/sample_report_nebula.md](docs/sample_report_nebula.md)**

**Chosen app:** Nebula on Trustpilot (`dataset=nebula`, n=100)

| Metric | Value |
|--------|------:|
| Overall sentiment | positive |
| Avg sentiment score | +0.66 (RoBERTa FT) |
| Avg native rating | 4.24 / 5 |
| Positive / neutral / negative | 83% / 2% / 15% |
| 5★ share | 63% |
| 1–2★ reviews | 10 |

**Highlights**

- Strength: product/reading quality mentioned often (avg sentiment +0.33).  
- Risk: billing / “money” / “fraudulently took” language dominates negative reviews.  
- Action: clarify cancel/refund flows and reply to every new 1–2★ review within 24–48h.

### Visualizations

Sentiment + rating distribution:

![Nebula insights overview](data/charts/nebula_insights_overview.png)

Top terms in negative reviews:

![Nebula negative keywords](data/charts/nebula_negative_keywords.png)

Regenerate:

```bash
python scripts/visualize_insights.py --dataset nebula
# → data/insights_nebula.json
# → data/charts/nebula_insights_overview.png
# → data/charts/nebula_negative_keywords.png
```

---

## 4. Extra scripts & notebooks

```bash
# Insights JSON only
python scripts/generate_insights.py

# Charts for another dataset
python scripts/visualize_insights.py --dataset obrio
python scripts/visualize_insights.py --dataset all

# NLP method bake-offs
jupyter notebook notebooks/nlp_methods_comparison.ipynb
jupyter notebook notebooks/uk_obrio_sentiment_comparison.ipynb
```

---

## Notes

- Trustpilot blocks plain HTTP scrapers; collection uses headless Chromium (Playwright).
- Rating endpoints use **native stars only** for product metrics; DOU analysis is text/NLP-first.
- Sample data may already be present under `data/` so the API works before a fresh collect.
