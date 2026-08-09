#!/usr/bin/env python3
"""Build notebooks/uk_obrio_sentiment_comparison.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "uk_obrio_sentiment_comparison.ipynb"


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


cells = [
    md(
        """# Ukrainian Obrio reviews — model comparison

Focus: **Obrio DOU** reviews (Ukrainian). DOU has no star ratings, so we compare methods via:

- label **distributions**
- **pairwise agreement**
- small **Gemini** reference sample
- weak **lexicon** labels (honest baseline, not gold)

| Method | Model |
|--------|--------|
| Lexicon | project UK/EN term lists |
| spaCy + lexicon | `uk_core_news_sm` lemmas |
| UKR-RoBERTa (FT) | `youscan/ukr-roberta-base` fine-tuned on weak labels |
| mBERT / UkrSenti-style (FT) | `bert-base-multilingual-cased` fine-tuned the same way |
| XLM-RoBERTa | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| mDeBERTa-v3 | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` zero-shot |
| Gemini | LLM prompting (small sample) |
"""
    ),
    md("## 0. Imports & config"),
    code(
        """import os
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 4)

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from app.services.sentiment import analyze_text
from app.storage import resolve_dataset

DATASET = "obrio"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

UKR_ROBERTA = "youscan/ukr-roberta-base"
MBERT = "bert-base-multilingual-cased"
XLM_R = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MDEBERTA = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
FT_EPOCHS = 3
MAX_GEMINI = 5
GEMINI_SLEEP_S = 4.0
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RANDOM_STATE = 42
"""
    ),
    md("## 1. Load Obrio Ukrainian reviews"),
    code(
        """source_filter = resolve_dataset(DATASET)
df = pd.read_json(ROOT / "data" / "processed_reviews.json")
for col, default in [("title", None), ("rating", None), ("text", "")]:
    if col not in df.columns:
        df[col] = default
df = df[df["source"] == source_filter].copy() if source_filter else df
df["text"] = df["text"].fillna("").astype(str)
df["full_text"] = (
    df["title"].fillna("").astype(str).str.strip() + " " + df["text"].str.strip()
).str.strip()
df = df[df["full_text"].str.len() > 0].copy()

# Weak labels from project lexicon (for FT train split + optional metrics)
weak = df["full_text"].map(lambda t: analyze_text(t, rating=None)[0].value)
df["weak_label"] = weak
df["y"] = df["weak_label"].map(LABEL2ID)

print(f"Obrio reviews: {len(df)}")
print("Weak lexicon labels:", df["weak_label"].value_counts().to_dict())
display(df[["id", "weak_label", "full_text"]].head(3))
"""
    ),
    md("## 2. Shared helpers"),
    code(
        """preds: dict[str, list[str]] = {}
timings: dict[str, float] = {}


def record(name: str, labels: list[str], seconds: float):
    assert len(labels) == len(df)
    preds[name] = labels
    timings[name] = seconds
    print(name, pd.Series(labels).value_counts().to_dict(), f"{seconds:.2f}s")


def fine_tune_predict(model_id: str, name: str) -> list[str]:
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=RANDOM_STATE, stratify=df["weak_label"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], truncation=True, max_length=256)
            enc["labels"] = int(self.labels[idx])
            return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    args = TrainingArguments(
        output_dir=str(ROOT / "data" / "models" / name.replace(" ", "_")),
        num_train_epochs=FT_EPOCHS,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=RANDOM_STATE,
        use_cpu=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=DS(train_df["full_text"], train_df["y"]),
        data_collator=collator,
    )
    t0 = time.perf_counter()
    trainer.train()
    train_secs = time.perf_counter() - t0

    # Predict on ALL Obrio rows for fair method comparison
    full_ds = DS(df["full_text"], df["y"])
    t0 = time.perf_counter()
    logits = trainer.predict(full_ds).predictions
    labels = [ID2LABEL[int(i)] for i in np.argmax(logits, axis=1)]
    infer_secs = time.perf_counter() - t0

    # Holdout score vs weak labels (diagnostic only)
    test_logits = trainer.predict(DS(test_df["full_text"], test_df["y"])).predictions
    test_pred = [ID2LABEL[int(i)] for i in np.argmax(test_logits, axis=1)]
    print(
        f"  holdout vs weak lexicon: "
        f"acc={accuracy_score(test_df['weak_label'], test_pred):.3f} "
        f"macroF1={f1_score(test_df['weak_label'], test_pred, average='macro'):.3f} "
        f"(train {train_secs:.1f}s / infer {infer_secs:.1f}s)"
    )
    record(name, labels, train_secs + infer_secs)
    return labels
"""
    ),
    md("## 3. Lexicon (project)"),
    code(
        """t0 = time.perf_counter()
lex_pred = [analyze_text(t, rating=None)[0].value for t in df["full_text"]]
record("Lexicon (project)", lex_pred, time.perf_counter() - t0)
"""
    ),
    md("## 4. spaCy `uk_core_news_sm` + lexicon on lemmas"),
    code(
        """nlp = spacy.load("uk_core_news_sm")


def spacy_lexicon_label(text: str) -> str:
    doc = nlp(text or "")
    lemmas = " ".join(tok.lemma_ for tok in doc if not tok.is_space)
    # Reuse project analyzer on lemmatized Ukrainian text
    return analyze_text(lemmas or text, rating=None)[0].value


t0 = time.perf_counter()
spacy_pred = [spacy_lexicon_label(t) for t in df["full_text"]]
record("spaCy uk_core_news_sm + lexicon", spacy_pred, time.perf_counter() - t0)

# Linguistic peek
sample = df["full_text"].iloc[0]
doc = nlp(sample)
print("Sample lemmas/POS:")
print([(tok.text, tok.lemma_, tok.pos_) for tok in doc[:20]])
"""
    ),
    md(
        """## 5. UKR-RoBERTa (fine-tuned)

`youscan/ukr-roberta-base` fine-tuned on Obrio **weak lexicon labels** (no DOU stars available).
"""
    ),
    code(
        """fine_tune_predict(UKR_ROBERTA, "UKR-RoBERTa (FT)")
"""
    ),
    md(
        """## 6. mBERT / UkrSentiBERT-style (fine-tuned)

Public **UkrSentiBERT** checkpoints are scarce/incomplete on HF, so we fine-tune
`bert-base-multilingual-cased` the same way (common Ukrainian sentiment baseline in papers).
"""
    ),
    code(
        """fine_tune_predict(MBERT, "mBERT UkrSenti-style (FT)")
"""
    ),
    md("## 7. XLM-RoBERTa (pretrained multilingual sentiment)"),
    code(
        """xlm = pipeline(
    "sentiment-analysis",
    model=XLM_R,
    tokenizer=XLM_R,
    truncation=True,
    max_length=512,
    device=-1,
)
t0 = time.perf_counter()
xlm_out = xlm(df["full_text"].tolist(), batch_size=8)
xlm_pred = [str(item["label"]).lower() for item in xlm_out]
record("XLM-RoBERTa (pretrained)", xlm_pred, time.perf_counter() - t0)
"""
    ),
    md("## 8. mDeBERTa-v3 (zero-shot NLI)"),
    code(
        """zs = pipeline(
    "zero-shot-classification",
    model=MDEBERTA,
    device=-1,
)
t0 = time.perf_counter()
mdeb_pred = []
for text in df["full_text"].tolist():
    out = zs(text[:1500], candidate_labels=LABELS, multi_label=False)
    mdeb_pred.append(out["labels"][0])
record("mDeBERTa-v3 zero-shot", mdeb_pred, time.perf_counter() - t0)
"""
    ),
    md(
        """## 9. Gemini LLM (small sample)

Scores only `MAX_GEMINI` reviews to stay under free-tier quota.
"""
    ),
    code(
        """api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None


def parse_label(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return "neutral"
    first = re.sub(r"[^a-z]+", "", text.split()[0])
    if first in LABELS:
        return first
    for label in ("positive", "negative", "neutral"):
        if re.search(rf"\\b{label}\\b", text):
            return label
    return "neutral"


GEMINI_FEW_SHOT = '''Ти строгий класифікатор тональності відгуків.
Поверни рівно одне англійське слово: positive, neutral, або negative.
Без пояснень. Якщо є чітка похвала чи критика — не став neutral.

Приклади:
Відгук: "Дуже дякую команді, чудовий досвід і підтримка на кожному кроці!"
Label: positive

Відгук: "Токсичний менеджмент, ігнорують і вигораєш від переробок."
Label: negative

Відгук: "Звичайна компанія, нічого особливого."
Label: neutral

Відгук: "Шахрайство з офером, після співбесіди тиша."
Label: negative

Відгук: "Класна атмосфера, росту професійно, рекомендую."
Label: positive
'''


gemini_rows = []
if gemini_client is None:
    print("Gemini skipped — set GEMINI_API_KEY in .env")
else:
    # Stratified tiny sample
    parts = []
    for _, g in df.groupby("weak_label"):
        parts.append(g.sample(n=min(2, len(g)), random_state=RANDOM_STATE))
    sample = pd.concat(parts).head(MAX_GEMINI).reset_index(drop=True)
    print(f"Gemini on {len(sample)} reviews via {GEMINI_MODEL}")
    t0 = time.perf_counter()
    for i, row in sample.iterrows():
        prompt = (
            f"{GEMINI_FEW_SHOT}\\n"
            f"Відгук: {row['full_text'][:800]!r}\\n"
            "Label:"
        )
        try:
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=8),
            )
            label = parse_label(resp.text or "")
        except genai_errors.ClientError as exc:
            print(f"Gemini stopped at {i}: {exc}")
            break
        gemini_rows.append({
            "id": row.get("id"),
            "weak_label": row["weak_label"],
            "gemini": label,
            "text": row["full_text"][:200],
        })
        print(f"  [{len(gemini_rows)}/{len(sample)}] {label}")
        if GEMINI_SLEEP_S and len(gemini_rows) < len(sample):
            time.sleep(GEMINI_SLEEP_S)
    print(f"done in {time.perf_counter() - t0:.1f}s")
    if gemini_rows:
        gemini_df = pd.DataFrame(gemini_rows)
        display(gemini_df)
        print(
            classification_report(
                gemini_df["weak_label"], gemini_df["gemini"],
                labels=LABELS, digits=3, zero_division=0,
            )
        )
"""
    ),
    md("## 10. Combined comparison"),
    code(
        """# Distributions
dist = pd.DataFrame({name: pd.Series(vals).value_counts(normalize=True) for name, vals in preds.items()})
dist = dist.reindex(LABELS).fillna(0.0)
display((dist * 100).round(1))

fig, ax = plt.subplots(figsize=(11, 4))
dist.T.plot(kind="bar", ax=ax, color=["#c45c26", "#8a8f98", "#2a9d8f"])
ax.set_ylabel("Share")
ax.set_title("Predicted label distribution by method")
ax.legend(title="label")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()

# Pairwise agreement
methods = list(preds.keys())
agree = pd.DataFrame(index=methods, columns=methods, dtype=float)
for a in methods:
    for b in methods:
        agree.loc[a, b] = np.mean(np.array(preds[a]) == np.array(preds[b]))

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(agree.astype(float), annot=True, fmt=".2f", cmap="Greens", ax=ax, vmin=0.3, vmax=1)
ax.set_title("Inter-method agreement (Obrio UA)")
plt.tight_layout()
charts = ROOT / "data" / "charts"
charts.mkdir(parents=True, exist_ok=True)
fig.savefig(charts / "uk_obrio_agreement.png", dpi=150, bbox_inches="tight")
plt.show()

# vs weak lexicon
rows = []
for name, labels in preds.items():
    rows.append({
        "method": name,
        "agree_with_lexicon": round(float(np.mean(np.array(labels) == np.array(lex_pred))), 3),
        "seconds": round(timings[name], 2),
        "pos_share": round(labels.count("positive") / len(labels), 3),
        "neg_share": round(labels.count("negative") / len(labels), 3),
    })
summary = pd.DataFrame(rows).sort_values("agree_with_lexicon", ascending=False)
display(summary)
summary.to_csv(charts / "uk_obrio_methods_summary.csv", index=False)
print("Saved", charts / "uk_obrio_methods_summary.csv")
"""
    ),
    code(
        """# Side-by-side predictions sample
view = df[["full_text", "weak_label"]].copy()
for name, labels in preds.items():
    view[name] = labels
view["text_short"] = view["full_text"].str.replace("\\n", " ", regex=False).str.slice(0, 120)
cols = ["text_short", "weak_label"] + list(preds.keys())
display(view[cols].head(12))
"""
    ),
    md(
        """## Notes

- Obrio DOU reviews are Ukrainian and **unrated**; do not treat lexicon / FT holdout scores as true accuracy.
- Prefer **XLM-RoBERTa**, **mDeBERTa zero-shot**, and **Gemini** for zero-shot UA sentiment; use **UKR-RoBERTa FT** when you can collect a small labeled set.
- `uk_core_news_sm` helps with lemmas/POS; polarity still comes from the lexicon layer here.
- Install spaCy model once: `python -m spacy download uk_core_news_sm`
"""
    ),
]

OUT.write_text(
    json.dumps(
        {
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
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUT} ({len(cells)} cells)")
