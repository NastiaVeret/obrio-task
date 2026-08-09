#!/usr/bin/env python3
"""Build notebooks/nlp_methods_comparison.ipynb — four sentiment methods."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "nlp_methods_comparison.ipynb"


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
        """# Sentiment Methods Comparison

Compare four approaches on Nebula / Obrio reviews (star ratings as labels):

| Method | Type |
|--------|------|
| **VADER** | Lexicon-based |
| **TF-IDF + Logistic Regression** | Classic ML |
| **RoBERTa** | Fine-tuned transformer (trained on our train split) |
| **Gemini** | LLM prompting (`GOOGLE_API_KEY` / `GEMINI_API_KEY`) |

All methods are scored on the **same test set**. Metrics: accuracy, macro-F1, per-class F1, confusion matrices, runtime.
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
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 4)

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from app.storage import resolve_dataset

DATASET = "all"  # nebula | nebula_appstore | obrio | all
TEST_SIZE = 0.25
RANDOM_STATE = 42
MAX_GEMINI = 5  # small sample to stay under free-tier quota
GEMINI_SLEEP_S = 4.0  # pause between calls
GEMINI_MAX_RETRIES = 2
ROBERTA_BASE = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_EPOCHS = 3
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Avoid gemini-2.0-flash — often shows free-tier limit: 0 when exhausted
GEMINI_FALLBACKS = [
    GEMINI_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b",
]
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
"""
    ),
    md("## 1. Load data & train/test split"),
    code(
        """def stars_to_label(rating) -> str | None:
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return None
    stars = int(round(float(rating)))
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


source_filter = resolve_dataset(DATASET)
df = pd.read_json(ROOT / "data" / "processed_reviews.json")
for col, default in [("title", None), ("rating", None), ("text", "")]:
    if col not in df.columns:
        df[col] = default
if source_filter is not None:
    df = df[df["source"] == source_filter].copy()

df["text"] = df["text"].fillna("").astype(str)
df["full_text"] = (
    df["title"].fillna("").astype(str).str.strip() + " " + df["text"].str.strip()
).str.strip()
df["label"] = df["rating"].map(stars_to_label)
data = df[df["label"].notna() & (df["full_text"].str.len() > 0)].copy()
data["y"] = data["label"].map(LABEL2ID)

train_df, test_df = train_test_split(
    data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=data["label"],
)

print(f"DATASET={DATASET!r}  total rated={len(data)}")
print("label counts:", data["label"].value_counts().to_dict())
print(f"train={len(train_df)}  test={len(test_df)}")
display(test_df[["source", "rating", "label", "full_text"]].head(3))
"""
    ),
    md(
        """## 2. Shared evaluation helpers

Every method returns a list of labels for `test_df`. Results go into `results`.
"""
    ),
    code(
        """results: dict[str, dict] = {}


def evaluate(name: str, y_pred: list[str], seconds: float) -> pd.Series:
    y_true = test_df["label"].tolist()
    assert len(y_pred) == len(y_true)
    row = {
        "method": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 3),
        "seconds": round(seconds, 3),
        "reviews_per_sec": round(len(y_true) / max(seconds, 1e-9), 1),
    }
    _, _, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    for label, f1 in zip(LABELS, f1s):
        row[f"f1_{label}"] = round(float(f1), 3)
    results[name] = {"y_pred": y_pred, **row}
    print(name, row)
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
    return pd.Series(row)


def metrics_table() -> pd.DataFrame:
    cols = [
        "method", "accuracy", "macro_f1", "weighted_f1",
        "f1_negative", "f1_neutral", "f1_positive",
        "seconds", "reviews_per_sec",
    ]
    return pd.DataFrame([results[k] for k in results])[cols].sort_values(
        ["macro_f1", "accuracy"], ascending=False
    )
"""
    ),
    md("## 3. Lexicon — VADER"),
    code(
        """vader = SentimentIntensityAnalyzer()


def predict_vader(texts: list[str]) -> list[str]:
    out = []
    for text in texts:
        score = vader.polarity_scores(text or "")["compound"]
        if score >= 0.05:
            out.append("positive")
        elif score <= -0.05:
            out.append("negative")
        else:
            out.append("neutral")
    return out


t0 = time.perf_counter()
vader_pred = predict_vader(test_df["full_text"].tolist())
evaluate("VADER (lexicon)", vader_pred, time.perf_counter() - t0)
"""
    ),
    md("## 4. Classic ML — TF-IDF + Logistic Regression"),
    code(
        """ml_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )),
])

t0 = time.perf_counter()
ml_pipe.fit(train_df["full_text"], train_df["label"])
train_secs = time.perf_counter() - t0

t0 = time.perf_counter()
ml_pred = ml_pipe.predict(test_df["full_text"]).tolist()
infer_secs = time.perf_counter() - t0

print(f"train {train_secs:.2f}s | infer {infer_secs:.2f}s")
evaluate("TF-IDF + LogReg (classic ML)", ml_pred, train_secs + infer_secs)
"""
    ),
    md(
        """## 5. Fine-tuned transformer — RoBERTa

Starts from a sentiment-pretrained RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and fine-tunes it on the **train** split, then predicts the **test** split.
"""
    ),
    code(
        """tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE)
collator = DataCollatorWithPadding(tokenizer=tokenizer)


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], truncation=True, max_length=256)
        enc["labels"] = int(self.labels[idx])
        return enc


train_ds = ReviewDataset(train_df["full_text"], train_df["y"])
test_ds = ReviewDataset(test_df["full_text"], test_df["y"])

model = AutoModelForSequenceClassification.from_pretrained(
    ROBERTA_BASE,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

training_args = TrainingArguments(
    output_dir=str(ROOT / "data" / "models" / "roberta_sentiment"),
    num_train_epochs=ROBERTA_EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
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
    args=training_args,
    train_dataset=train_ds,
    data_collator=collator,
)

t0 = time.perf_counter()
trainer.train()
train_secs = time.perf_counter() - t0

t0 = time.perf_counter()
logits = trainer.predict(test_ds).predictions
roberta_pred = [ID2LABEL[int(i)] for i in np.argmax(logits, axis=1)]
infer_secs = time.perf_counter() - t0

print(f"train {train_secs:.1f}s | infer {infer_secs:.1f}s")
evaluate("RoBERTa fine-tuned", roberta_pred, train_secs + infer_secs)
"""
    ),
    md(
        """## 6. LLM prompting — Google Gemini

Uses `GOOGLE_API_KEY` / `GEMINI_API_KEY` from `.env`.  
Handles **429 quota** with backoff, model fallbacks, and partial results (won't crash the notebook).
"""
    ),
    code(
        """api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None
active_gemini_model = None


def _response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    parts = []
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "\\n".join(parts)


def _parse_label(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return "neutral"
    # Handle "label: positive" / "answer: negative"
    text = re.sub(r"^(label|answer|sentiment)\\s*[:=-]\\s*", "", text).strip()
    first = re.sub(r"[^a-z]+", "", text.split()[0])
    if first in LABELS:
        return first
    # Prefer polar labels before neutral.
    for label in ("positive", "negative", "neutral"):
        if re.search(r"\\b" + label + r"\\b", text):
            return label
    return "neutral"


GEMINI_FEW_SHOT = "\\n".join([
    "You are a strict sentiment classifier for app/customer reviews.",
    "Reply with ONLY one word: positive OR negative OR neutral.",
    "No punctuation, no Label: prefix, no explanation.",
    "If the user is happy/satisfied -> positive.",
    "If the user complains/angry/scammed -> negative.",
    "Use neutral ONLY when the review is truly mixed or bland.",
    "",
    "Examples:",
    "Review: Amazing app, the advisors are so helpful and I love the readings!",
    "positive",
    "",
    "Review: Scam. They charged me and refused a refund. Worst experience ever.",
    "negative",
    "",
    "Review: It works okay. Nothing special, nothing terrible.",
    "neutral",
    "",
    "Review: Cancelled after one chat - vague answers and pushy upsells.",
    "negative",
    "",
    "Review: Super accurate reading, worth every credit. Highly recommend!",
    "positive",
])


def predict_gemini_one(text: str, model: str, *, debug: bool = False) -> str:
    review = (text or "").replace("\\n", " ").strip()[:800]
    prompt = GEMINI_FEW_SHOT + "\\n\\nReview: " + review + "\\n"
    delay = 20.0
    last_err = None
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=16,
                ),
            )
            raw = _response_text(response)
            label = _parse_label(raw)
            if debug:
                print(f"    raw={raw!r} -> {label}")
            return label
        except genai_errors.ClientError as exc:
            last_err = exc
            msg = str(exc)
            # Daily/free-tier hard stop — don't keep retrying this model
            if "limit: 0" in msg:
                raise
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                print(
                    f"  rate-limit on {model} "
                    f"(attempt {attempt+1}/{GEMINI_MAX_RETRIES}); sleep {delay:.0f}s"
                )
                time.sleep(delay)
                continue
            raise
    raise last_err


def pick_gemini_model() -> str | None:
    seen = []
    for model in GEMINI_FALLBACKS:
        if not model or model in seen:
            continue
        seen.append(model)
        try:
            pred = predict_gemini_one(
                "Amazing app, the advisors are so helpful and I love it!",
                model,
                debug=True,
            )
            print(f"Gemini model OK: {model} → {pred}")
            if pred != "positive":
                print("  warning: few-shot probe expected positive; check prompt/model")
            return model
        except Exception as exc:
            print(f"Gemini model failed ({model}): {type(exc).__name__}: {str(exc)[:160]}")
    return None


def gemini_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    # Small stratified sample (not only 5-star reviews).
    if n is None or len(frame) <= n:
        return frame.copy()
    parts = []
    per = max(1, n // frame["label"].nunique())
    for _, group in frame.groupby("label"):
        parts.append(group.sample(n=min(per, len(group)), random_state=RANDOM_STATE))
    sample = pd.concat(parts)
    if len(sample) < n:
        rest = frame.drop(index=sample.index)
        need = min(n - len(sample), len(rest))
        if need:
            sample = pd.concat(
                [sample, rest.sample(n=need, random_state=RANDOM_STATE)]
            )
    return sample.head(n).reset_index(drop=True)


def score_gemini(y_true, y_pred, model: str, secs: float, note: str):
    y_true = [str(x).strip().lower() for x in y_true]
    y_pred = [str(x).strip().lower() for x in y_pred]
    print("DEBUG y_true:", y_true)
    print("DEBUG y_pred:", y_pred)
    row = {
        "method": f"Gemini ({model})",
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 3),
        "seconds": round(secs, 3),
        "reviews_per_sec": round(len(y_pred) / max(secs, 1e-9), 1),
        "note": note,
    }
    _, _, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    for label, f1 in zip(LABELS, f1s):
        row[f"f1_{label}"] = round(float(f1), 3)
    results[row["method"]] = {"y_pred": y_pred, "y_true_subset": y_true, **row}
    print(row["method"], row)
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))


if gemini_client is None:
    print("Gemini skipped — set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
else:
    try:
        active_gemini_model = pick_gemini_model()
        if active_gemini_model is None:
            print("Gemini unavailable (quota/billing). Continuing without it.")
        else:
            gemini_df = (
                test_df
                if MAX_GEMINI is None
                else gemini_sample(test_df, MAX_GEMINI)
            )
            print(
                f"Gemini scoring {len(gemini_df)} reviews "
                f"(of {len(test_df)} test) on {active_gemini_model}"
            )
            texts = gemini_df["full_text"].tolist()
            y_true_sub = gemini_df["label"].tolist()
            gemini_pred = []
            t0 = time.perf_counter()
            for i, text in enumerate(texts):
                try:
                    gemini_pred.append(
                        predict_gemini_one(text, active_gemini_model, debug=True)
                    )
                    print(
                        f"  [{i+1}/{len(texts)}] → {gemini_pred[-1]}  "
                        f"(true={y_true_sub[i]})"
                    )
                except Exception as exc:
                    print(
                        f"Stopping Gemini early at {i}/{len(texts)}: "
                        f"{type(exc).__name__}: {str(exc)[:180]}"
                    )
                    break
                if GEMINI_SLEEP_S and i + 1 < len(texts):
                    time.sleep(GEMINI_SLEEP_S)
            secs = time.perf_counter() - t0
            if not gemini_pred:
                print("Gemini produced no predictions.")
            else:
                score_gemini(
                    y_true_sub[: len(gemini_pred)],
                    gemini_pred,
                    active_gemini_model,
                    secs,
                    note=f"scored {len(gemini_pred)}/{len(test_df)} test rows",
                )
    except Exception as exc:
        # Keep traceback short so it does not paint over later HTML tables in VS Code.
        print(f"Gemini section failed: {type(exc).__name__}: {str(exc)[:240]}")
"""
    ),
    md("## 7. Combined metrics & visualizations"),
    code(
        """compare = metrics_table()
# Plain text avoids VS Code/Jupyter HTML-table + traceback overlap glitches.
print(compare.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=compare, x="method", y="macro_f1", ax=axes[0], color="#3d7ea6")
axes[0].set_title("Macro-F1")
axes[0].tick_params(axis="x", rotation=25)
axes[0].set_ylim(0, 1)

sns.barplot(data=compare, x="method", y="accuracy", ax=axes[1], color="#2f6f4e")
axes[1].set_title("Accuracy")
axes[1].tick_params(axis="x", rotation=25)
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """f1_long = compare.melt(
    id_vars=["method"],
    value_vars=["f1_negative", "f1_neutral", "f1_positive"],
    var_name="class",
    value_name="f1",
)
f1_long["class"] = f1_long["class"].str.replace("f1_", "", regex=False)

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=f1_long, x="method", y="f1", hue="class", ax=ax)
ax.set_title("Per-class F1")
ax.tick_params(axis="x", rotation=25)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=compare, x="method", y="seconds", ax=ax, color="#c45c26")
ax.set_title("Total time (train + infer where applicable)")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """n = len(results)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6))
if n == 1:
    axes = [axes]

for ax, (name, payload) in zip(axes, results.items()):
    y_true = payload.get("y_true_subset", test_df["label"].tolist())
    y_pred = payload["y_pred"]
    # If Gemini used a subset, align lengths
    if len(y_pred) != len(y_true):
        y_true = y_true[: len(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar=False,
    )
    ax.set_title(name)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (stars)")

plt.suptitle("Confusion matrices", y=1.02)
plt.tight_layout()

charts = ROOT / "data" / "charts"
charts.mkdir(parents=True, exist_ok=True)
fig.savefig(charts / "nlp_methods_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved", charts / "nlp_methods_confusion.png")
"""
    ),
    code(
        """methods = [m for m in results if "y_true_subset" not in results[m]]
if len(methods) >= 2:
    agree = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            agree.loc[a, b] = np.mean(
                np.array(results[a]["y_pred"]) == np.array(results[b]["y_pred"])
            )
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(agree.astype(float), annot=True, fmt=".2f", cmap="Greens", ax=ax, vmin=0.4, vmax=1)
    ax.set_title("Agreement on full test set")
    plt.tight_layout()
    fig.savefig(charts / "nlp_methods_agreement.png", dpi=150, bbox_inches="tight")
    plt.show()

compare.to_csv(charts / "nlp_methods_metrics.csv", index=False)
print("Saved metrics →", charts / "nlp_methods_metrics.csv")
display(compare)
"""
    ),
    md(
        """## Notes

- **Labels** come from stars: 1–2 → negative, 3 → neutral, 4–5 → positive (Trustpilot / App Store only).
- **VADER / Gemini** do not use the train split; **TF-IDF+LR** and **RoBERTa** are fit on train.
- Class imbalance is strong (many 5★). Macro-F1 matters more than accuracy.
- Gemini: put `GEMINI_API_KEY=...` in `.env`. On 429, the cell retries / falls back / keeps partial scores.
- If free-tier quota for a model is `limit: 0`, switch `GEMINI_MODEL` or enable billing in Google AI Studio.
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
