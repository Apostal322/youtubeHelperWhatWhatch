"""Lightweight topic keywords (Stage A) — frequency of content terms in title/tags/description."""

from __future__ import annotations

import re
from collections import Counter

from avd.evaluation.evaluators.heuristic import STOPWORDS

_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
_EXTRA_STOP = {
    "video",
    "videos",
    "channel",
    "subscribe",
    "today",
    "episode",
    "part",
    "watch",
    "this",
    "that",
    "with",
}


def extract_topics(title: str, description: str, tags: list[str], n: int = 8) -> list[str]:
    stop = set().union(*STOPWORDS.values()) | _EXTRA_STOP
    counts: Counter[str] = Counter()
    for text, w in ((title, 3), (" ".join(tags), 2), (description[:1500], 1)):
        for tok in _WORD.findall(text.lower()):
            if tok not in stop:
                counts[tok] += w
    return [t for t, _ in counts.most_common(n)]


def transcript_keywords(text: str, n: int = 30) -> list[str]:
    """Most frequent content words of a transcript (stopwords removed, >= 4 letters)."""
    stop = set().union(*STOPWORDS.values()) | _EXTRA_STOP
    counts: Counter[str] = Counter(t for t in _WORD.findall(text.lower()) if t not in stop)
    return [t for t, _ in counts.most_common(n)]
