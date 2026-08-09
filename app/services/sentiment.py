from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from app.models.schemas import SentimentLabel

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ROBERTA_DIR = ROOT / "data" / "models" / "roberta_sentiment"
ROBERTA_BASE = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]

POSITIVE_TERMS = {
    "great",
    "love",
    "amazing",
    "excellent",
    "awesome",
    "helpful",
    "fantastic",
    "wonderful",
    "best",
    "recommend",
    "perfect",
    "good",
    "easy",
    "happy",
    "supportive",
    "подобається",
    "подобалось",
    "подоба",
    "чудов",
    "прекрас",
    "прекраст",
    "круто",
    "дякую",
    "вдячний",
    "вдячна",
    "професійн",
    "комфорт",
    "підтрим",
    "рекоменд",
    "позитив",
    "супер",
    "класно",
    "радію",
    "радий",
    "рада",
    "довіря",
    "цікав",
    "розвит",
    "развив",
    "семью",
    "сімʼя",
}

NEGATIVE_TERMS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "scam",
    "fraud",
    "refund",
    "waste",
    "broken",
    "bug",
    "crash",
    "expensive",
    "cancel",
    "disappointed",
    "worst",
    "spam",
    "шахрай",
    "обман",
    "поган",
    "жах",
    "негатив",
    "звільн",
    "скороч",
    "ігнор",
    "игнор",
    "розчарув",
    "разочаров",
    "проблема",
    "проблем",
    "токсич",
    "переработ",
    "вигоран",
    "выгоран",
    "не рекоменду",
    "позов",
    "скарг",
    "жалоб",
    "суддя",
    "судовий",
    "lawsuit",
}

THEME_PATTERNS: dict[str, tuple[str, ...]] = {
    "hiring_process": (
        "рекрут",
        "hiring",
        "interview",
        "співбесід",
        "офер",
        "offer",
        "фідбек",
        "feedback",
        "кандидат",
    ),
    "team_culture": (
        "команд",
        "team",
        "культур",
        "атмосфер",
        "довіра",
        "підтрим",
        "вайб",
        "colleagues",
    ),
    "growth_learning": (
        "розвит",
        "навчан",
        "growth",
        "learn",
        "курс",
        "скіл",
        "skill",
        "ментор",
    ),
    "work_life_balance": (
        "баланс",
        "overtime",
        "вигоран",
        "burnout",
        "переработ",
        "flexible",
        "гнучк",
        "remote",
        "віддален",
    ),
    "product_quality": (
        "product",
        "продукт",
        "bug",
        "crash",
        "feature",
        "app",
        "nebula",
        "функц",
    ),
    "pricing_billing": (
        "price",
        "pricing",
        "subscription",
        "refund",
        "expensive",
        "paywall",
        "billing",
        "підписк",
        "кошт",
    ),
    "customer_support": (
        "support",
        "customer service",
        "підтримк",
        "help",
        "response",
    ),
    "management": (
        "менеджмент",
        "management",
        "leadership",
        "керівництв",
        "bureaucracy",
        "бюрократ",
    ),
}

_VADER = None
_ROBERTA = None  # (tokenizer, model, device) | False if unavailable
_SPACY_UK = None  # nlp | False if unavailable


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ']+", text.lower()))


def _term_matches(term: str, lowered: str, tokens: set[str]) -> bool:
    """Match lexicon terms on tokens/stems — avoid ``суд`` hitting ``судьба``."""
    if " " in term:
        return term in lowered
    if term in tokens:
        return True
    # Prefix stems only when long enough to be discriminative.
    if len(term) >= 4:
        return any(tok.startswith(term) for tok in tokens)
    return False


def _score_from_terms(text: str) -> float:
    lowered = text.lower()
    tokens = _tokenize(text)
    score = 0.0
    for term in POSITIVE_TERMS:
        if _term_matches(term, lowered, tokens):
            score += 1.0
    for term in NEGATIVE_TERMS:
        if _term_matches(term, lowered, tokens):
            score -= 1.2
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 6.0))


def detect_themes(text: str) -> list[str]:
    lowered = text.lower()
    themes: list[str] = []
    for theme, patterns in THEME_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            themes.append(theme)
    return themes


def score_to_label(score: float) -> SentimentLabel:
    if score >= 0.15:
        return SentimentLabel.POSITIVE
    if score <= -0.15:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[а-яА-ЯіїєґІЇЄҐ]", text or ""))


def _vader_score(text: str) -> float | None:
    global _VADER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return None
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return float(_VADER.polarity_scores(text or "")["compound"])


def _load_roberta():
    """Lazy-load fine-tuned RoBERTa (preferred) or cardiffnlp base."""
    global _ROBERTA
    if _ROBERTA is not None:
        return _ROBERTA if _ROBERTA is not False else None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.warning("transformers/torch unavailable; falling back from RoBERTa")
        _ROBERTA = False
        return None

    model_id = str(ROBERTA_DIR) if (ROBERTA_DIR / "config.json").exists() else ROBERTA_BASE
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        _ROBERTA = (tokenizer, model, device, model_id)
        logger.info("Loaded RoBERTa sentiment model from %s", model_id)
        return _ROBERTA
    except Exception as exc:  # noqa: BLE001 — keep API up if model download fails
        logger.warning("RoBERTa load failed (%s); using lexicon/VADER fallback", exc)
        _ROBERTA = False
        return None


def _roberta_score(text: str) -> float | None:
    loaded = _load_roberta()
    if not loaded:
        return None
    tokenizer, model, device, _ = loaded
    import torch
    import torch.nn.functional as F

    enc = tokenizer(
        text or "",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    id2label = {int(k): v for k, v in getattr(model.config, "id2label", {}).items()}
    if not id2label:
        id2label = {i: label for i, label in enumerate(LABELS)}

    # Normalize label names (cardiffnlp may use LABEL_0 / Negative / etc.)
    def canon(name: str) -> str:
        n = str(name).lower()
        if "neg" in n or n.endswith("0"):
            return "negative"
        if "neu" in n or n.endswith("1"):
            return "neutral"
        if "pos" in n or n.endswith("2"):
            return "positive"
        return n

    by_label = {canon(id2label[i]): float(probs[i]) for i in range(len(probs))}
    pos = by_label.get("positive", 0.0)
    neg = by_label.get("negative", 0.0)
    return float(np.clip(pos - neg, -1.0, 1.0))


def _load_spacy_uk():
    global _SPACY_UK
    if _SPACY_UK is not None:
        return _SPACY_UK if _SPACY_UK is not False else None
    try:
        import spacy

        _SPACY_UK = spacy.load("uk_core_news_sm")
        return _SPACY_UK
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "spaCy uk_core_news_sm unavailable (%s); using raw-text lexicon", exc
        )
        _SPACY_UK = False
        return None


def _uk_spacy_lexicon_score(text: str) -> float:
    """Obrio path: spaCy lemmas + project lexicon (best UA method in comparison)."""
    nlp = _load_spacy_uk()
    if nlp is None:
        return _score_from_terms(text)
    doc = nlp(text or "")
    lemmas = " ".join(tok.lemma_ for tok in doc if not tok.is_space)
    return _score_from_terms(lemmas or text)


def _english_text_score(text: str) -> float:
    """Nebula path: fine-tuned RoBERTa, with VADER+lexicon fallback."""
    roberta = _roberta_score(text)
    if roberta is not None:
        return roberta
    lexical = _score_from_terms(text)
    vader = _vader_score(text)
    if vader is not None:
        return 0.75 * vader + 0.25 * lexical
    return lexical


def analyze_text(
    text: str, rating: float | None = None
) -> tuple[SentimentLabel, float, list[str]]:
    """Sentiment for a review.

    - English / Latin (Nebula): fine-tuned RoBERTa
    - Ukrainian / Cyrillic (Obrio): spaCy ``uk_core_news_sm`` lemmas + lexicon
    - Optional native star blend for product reviews
    """
    if _has_cyrillic(text):
        text_score = _uk_spacy_lexicon_score(text)
    else:
        text_score = _english_text_score(text)

    if rating is not None:
        rating_score = (float(rating) - 3.0) / 2.0
        score = 0.55 * text_score + 0.45 * rating_score
    else:
        score = text_score

    score = max(-1.0, min(1.0, score))
    return score_to_label(score), round(score, 3), detect_themes(text)
