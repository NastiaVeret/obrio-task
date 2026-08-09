#!/usr/bin/env python3
"""Fine-tune RoBERTa on rated Nebula reviews and save for production inference.

Uses star ratings as labels (1–2 negative, 3 neutral, 4–5 positive), matching
the NLP comparison notebook. Writes ``data/models/roberta_sentiment``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
ROBERTA_BASE = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_OUT = ROOT / "data" / "models" / "roberta_sentiment"


def stars_to_label(rating) -> str | None:
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return None
    stars = int(round(float(rating)))
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=256)
        enc["labels"] = int(self.labels[idx])
        return enc


def load_rated_frame(path: Path) -> pd.DataFrame:
    df = pd.read_json(path)
    for col, default in [("title", None), ("rating", None), ("text", ""), ("source", "")]:
        if col not in df.columns:
            df[col] = default
    # Rated product sources only (Trustpilot / App Store)
    df = df[df["source"].isin(["trustpilot", "app_store"])].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["full_text"] = (
        df["title"].fillna("").astype(str).str.strip() + " " + df["text"].str.strip()
    ).str.strip()
    df["label"] = df["rating"].map(stars_to_label)
    data = df[df["label"].notna() & (df["full_text"].str.len() > 0)].copy()
    data["y"] = data["label"].map(LABEL2ID)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa sentiment model")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed_reviews.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--base-model", default=ROBERTA_BASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_rated_frame(args.input)
    if len(data) < 20:
        print(f"Need more rated reviews (found {len(data)}).", file=sys.stderr)
        return 1

    train_df, test_df = train_test_split(
        data,
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )
    print(f"Rated reviews: {len(data)}  train={len(train_df)}  test={len(test_df)}")
    print("labels:", data["label"].value_counts().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args.output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output / "trainer_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=42,
        use_cpu=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ReviewDataset(tokenizer, train_df["full_text"], train_df["y"]),
        data_collator=collator,
    )
    trainer.train()

    logits = trainer.predict(
        ReviewDataset(tokenizer, test_df["full_text"], test_df["y"])
    ).predictions
    pred = [ID2LABEL[int(i)] for i in np.argmax(logits, axis=1)]
    y_true = test_df["label"].tolist()
    acc = float(np.mean(np.array(pred) == np.array(y_true)))
    print(f"Holdout accuracy vs star labels: {acc:.3f}")

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    meta = {
        "base_model": args.base_model,
        "labels": LABELS,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "holdout_accuracy": round(acc, 3),
        "epochs": args.epochs,
    }
    (args.output / "training_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Saved fine-tuned model → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
