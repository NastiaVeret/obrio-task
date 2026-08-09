from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ']+")

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "could",
    "of",
    "as",
    "also",
    "get",
    "got",
    "im",
    "ive",
    "dont",
    "didnt",
    "cant",
    "wasnt",
    "werent",
    "isnt",
    "arent",
    "nebula",
    "obrio",
    "app",
    "apps",
    "review",
    "reviews",
    "one",
    "two",
    "really",
    "even",
    "still",
    "much",
    "many",
    "lot",
    "like",
    "know",
    "want",
    "time",
    "back",
}

# Ukrainian + Russian function words / generic employer-review noise
UK_STOPWORDS = {
    "і",
    "й",
    "та",
    "але",
    "або",
    "чи",
    "не",
    "ні",
    "так",
    "як",
    "що",
    "це",
    "той",
    "ця",
    "ці",
    "для",
    "про",
    "при",
    "над",
    "під",
    "без",
    "від",
    "до",
    "зі",
    "із",
    "на",
    "у",
    "в",
    "з",
    "по",
    "за",
    "ще",
    "вже",
    "уже",
    "був",
    "була",
    "було",
    "були",
    "є",
    "бути",
    "мене",
    "мені",
    "ми",
    "ви",
    "вони",
    "они",
    "він",
    "вона",
    "воно",
    "його",
    "її",
    "їх",
    "дуже",
    "також",
    "тому",
    "томущо",
    "коли",
    "якщо",
    "там",
    "тут",
    "все",
    "всі",
    "сам",
    "себе",
    "свої",
    "свій",
    "моя",
    "мій",
    "наша",
    "наш",
    "тим",
    "теж",
    "обидві",
    "обидва",
    "обидвох",
    "року",
    "рік",
    "років",
    "місяць",
    "місяців",
    "місяці",
    "день",
    "дні",
    "днів",
    "год",
    "года",
    "лет",
    "всем",
    "привіт",
    "привет",
    "компанія",
    "компанії",
    "компанію",
    "компанією",
    "компании",
    "компания",
    "робот",
    "робота",
    "роботі",
    "роботу",
    "работа",
    "работы",
    "позицию",
    "позицію",
    "позиції",
    "вакансію",
    "вакансія",
    "вакансії",
    "досвід",
    "досвіду",
    "досвідом",
    "опыт",
    "опыта",
    "етап",
    "етапу",
    "етапі",
    "відбору",
    "відбір",
    "людина",
    "люди",
    "людей",
    "спеціаліст",
    "спеціалісти",
    "специалист",
    "специалисты",
    "можливість",
    "возможность",
    "який",
    "яка",
    "які",
    "яких",
    "котрий",
    "котра",
    "через",
    "після",
    "перед",
    "поки",
    "майже",
    "просто",
    "саме",
    "тільки",
    "только",
    "уже",
    "ещё",
    "еще",
    "было",
    "была",
    "были",
    "есть",
    "этот",
    "эта",
    "эти",
    "как",
    "что",
    "это",
    "для",
    "при",
    "над",
    "под",
    "без",
    "от",
    "до",
    "со",
    "из",
    "на",
    "у",
    "в",
    "с",
    "по",
    "за",
    "же",
    "ли",
    "бы",
    "то",
    "но",
    "да",
    "нет",
    "мне",
    "меня",
    "нас",
    "вас",
    "их",
    "его",
    "её",
    "ее",
    "моя",
    "мой",
    "наш",
    "наша",
}

# Stems that signal complaint / risk language (EN + UK + RU)
NEGATIVE_SIGNAL_STEMS = {
    "scam",
    "fraud",
    "refund",
    "charge",
    "charged",
    "billing",
    "cancel",
    "cancelled",
    "subscription",
    "expensive",
    "overpriced",
    "waste",
    "broken",
    "crash",
    "bug",
    "awful",
    "terrible",
    "hate",
    "worst",
    "disappoint",
    "ignore",
    "ignored",
    "spam",
    "steal",
    "stolen",
    "ripoff",
    "money",
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
    "позов",
    "скарг",
    "жалоб",
    "зарплат",
    "оклад",
    "овертайм",
    "навантаж",
    "бюрократ",
    "затрим",
    "задерж",
    "відмов",
    "отказ",
    "брехн",
    "лжив",
    "кошмар",
    "гірше",
    "хуже",
    "не рекоменду",
}

BRAND_NOISE = {
    "nebula",
    "obrio",
    "genesis",
    "socialtech",
    "headway",
    "solid",
    "boosters",
    "betterme",
    "lift",
    "stories",
    "editor",
    "asknebula",
    "dou",
    "facebook",
    "instagram",
}

POSITIVE_SIGNAL_STEMS = {
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
    "happy",
    "чудов",
    "прекрас",
    "прекраст",  # common typo
    "круто",
    "дякую",
    "вдячн",
    "позитив",
    "супер",
    "класно",
    "рекоменд",
    "подоба",
    "люблю",
}


def tokenize(text: str) -> list[str]:
    # Normalize typographic apostrophes so "пов'язані" stays one token.
    normalized = (text or "").replace("’", "'").replace("ʼ", "'").replace("`", "'")
    tokens = [t.lower().strip("'") for t in TOKEN_RE.findall(normalized)]
    stop = STOPWORDS | UK_STOPWORDS
    return [t for t in tokens if len(t) > 2 and t not in stop]


def ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 1:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _has_stem(term: str, stems: set[str]) -> bool:
    lowered = term.lower()
    parts = lowered.split()
    for stem in stems:
        stem = stem.strip()
        if not stem:
            continue
        if " " in stem:
            if stem in lowered:
                return True
            continue
        for part in parts:
            if part == stem or (len(stem) >= 4 and part.startswith(stem)):
                return True
    return False


def _is_noise_term(term: str) -> bool:
    if _has_stem(term, POSITIVE_SIGNAL_STEMS):
        return True
    parts = term.split()
    stop = STOPWORDS | UK_STOPWORDS | BRAND_NOISE
    if any(p in BRAND_NOISE for p in parts):
        return True
    if all(p in stop for p in parts):
        return True
    return False


def extract_keywords_and_phrases(
    texts: list[str],
    *,
    top_n: int = 15,
    include_bigrams: bool = True,
    include_trigrams: bool = True,
) -> list[dict]:
    """Return top keywords/phrases with counts from a corpus of texts."""
    unigram_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()

    for text in texts:
        tokens = tokenize(text)
        unigram_counts.update(tokens)
        if include_bigrams:
            phrase_counts.update(ngrams(tokens, 2))
        if include_trigrams:
            phrase_counts.update(ngrams(tokens, 3))

    phrases = [
        {"term": term, "count": count, "type": "phrase"}
        for term, count in phrase_counts.most_common()
        if count >= 2 and not _is_noise_term(term)
    ][: max(1, top_n // 2)]

    used_words = set()
    for item in phrases:
        used_words.update(item["term"].split())

    keywords = []
    for term, count in unigram_counts.most_common():
        if _is_noise_term(term):
            continue
        if term in used_words and count < 3:
            continue
        keywords.append({"term": term, "count": count, "type": "keyword"})
        if len(keywords) + len(phrases) >= top_n:
            break

    combined = phrases + keywords
    combined.sort(key=lambda item: (-item["count"], item["term"]))
    return combined[:top_n]


def extract_negative_keywords(
    negative_texts: list[str],
    positive_texts: list[str] | None = None,
    *,
    top_n: int = 15,
) -> list[dict]:
    """Keywords characteristic of negative reviews (not generic corpus noise).

    Prefers polarity/complaint stems, then contrastive terms that appear more
    in negative than positive reviews.
    """
    if not negative_texts:
        return []

    neg_uni: Counter[str] = Counter()
    neg_phr: Counter[str] = Counter()
    for text in negative_texts:
        tokens = tokenize(text)
        neg_uni.update(tokens)
        neg_phr.update(ngrams(tokens, 2))
        neg_phr.update(ngrams(tokens, 3))

    pos_uni: Counter[str] = Counter()
    pos_phr: Counter[str] = Counter()
    for text in positive_texts or []:
        tokens = tokenize(text)
        pos_uni.update(tokens)
        pos_phr.update(ngrams(tokens, 2))
        pos_phr.update(ngrams(tokens, 3))

    n_neg = max(len(negative_texts), 1)
    n_pos = max(len(positive_texts or []), 1)

    def score_term(term: str, neg_count: int, pos_count: int, *, is_phrase: bool) -> float:
        if _is_noise_term(term):
            return -1e9
        if is_phrase and neg_count < 2 and not _has_stem(term, NEGATIVE_SIGNAL_STEMS):
            return -1e9
        if not is_phrase and neg_count < 1:
            return -1e9

        neg_rate = neg_count / n_neg
        pos_rate = pos_count / n_pos
        lift = neg_rate - pos_rate
        signal_bonus = 3.0 if _has_stem(term, NEGATIVE_SIGNAL_STEMS) else 0.0
        # Require either a complaint signal or clear over-representation in negatives.
        if signal_bonus == 0.0 and lift <= 0.02 and neg_count < 3:
            return -1e9
        return signal_bonus + (2.0 * lift) + (0.15 * neg_count)

    scored: list[tuple[float, dict]] = []
    for term, count in neg_phr.items():
        s = score_term(term, count, pos_phr.get(term, 0), is_phrase=True)
        if s > -1e8:
            scored.append((s, {"term": term, "count": count, "type": "phrase"}))
    for term, count in neg_uni.items():
        s = score_term(term, count, pos_uni.get(term, 0), is_phrase=False)
        if s > -1e8:
            scored.append((s, {"term": term, "count": count, "type": "keyword"}))

    scored.sort(key=lambda item: (-item[0], -item[1]["count"], item[1]["term"]))

    signal_rows = [row for _, row in scored if _has_stem(row["term"], NEGATIVE_SIGNAL_STEMS)]
    other_rows = [row for _, row in scored if not _has_stem(row["term"], NEGATIVE_SIGNAL_STEMS)]
    signal_rows.sort(key=lambda row: (row["type"] != "keyword", -row["count"], row["term"]))

    selected: list[dict] = []
    seen_terms: set[str] = set()
    covered_tokens: set[str] = set()

    def overlaps_selected(term: str) -> bool:
        parts = set(term.split())
        return bool(parts & covered_tokens)

    def take(rows: list[dict]) -> None:
        for row in rows:
            if len(selected) >= top_n:
                return
            term = row["term"]
            if term in seen_terms or _is_noise_term(term):
                continue
            if overlaps_selected(term):
                continue
            seen_terms.add(term)
            covered_tokens.update(term.split())
            selected.append(row)

    take(signal_rows)
    # Only add contrastive non-signal terms when the negative set is large
    # enough; otherwise generic nouns drown out real complaints.
    if n_neg >= 8:
        take(other_rows)

    return selected
