"""Remove popularity signals from text the intrinsic-quality evaluator sees.

Titles and descriptions are needed (title-content consistency, scope), but descriptions often carry
popularity claims ("Thanks for 2 million subscribers!", "our most viewed video ever") that could
anchor a model judge. This module removes:

* from descriptions and tags: whole lines/sentences containing popularity statements;
* from titles: only explicit popularity *counts* and bracketed popularity claims, e.g. "(10M views)",
  because words like "viral" can be the legitimate subject of a video.

The sanitised text is what evaluators receive and what the input fingerprint hashes, so adding or
changing popularity statements cannot change the evaluation (metamorphic tests in the test suite).
"""

from __future__ import annotations

import re

_UNITS = (
    r"views?|subscribers?|subs|likes?|followers?|downloads?|plays|streams|watchers"
    r"|visualizaciones|vistas|reproducciones|suscriptores|seguidores"
    r"|aufrufe|abonnenten|abonnent\*innen|follower"
)
_COUNT = re.compile(
    rf"\b\d[\d.,]*\s*(k|m|b|mil|million(s|en)?|millones|thousand|tausend|mio\.?)?\s*\+?\s*({_UNITS})\b",
    re.I,
)
_CLAIM = re.compile(
    r"\b(most[- ](viewed|watched|popular|liked)|went viral|goes viral|viral (video|hit)|trending|#\s?1\b|"
    r"number one (video|channel)|millions? of (views|people watched)|record[- ]breaking views)\b",
    re.I,
)
_MILESTONE = re.compile(
    rf"\b(thank(s| you)|celebrat\w+|we (hit|reached|passed)|road to|milestone)\b[^.!?\n]{{0,60}}\b({_UNITS})\b",
    re.I,
)
_BRACKETED = re.compile(rf"[\[(][^\])]*\b({_UNITS}|viral|trending)\b[^\])]*[\])]", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _is_popularity(sentence: str) -> bool:
    return bool(_COUNT.search(sentence) or _CLAIM.search(sentence) or _MILESTONE.search(sentence))


def sanitize_description(text: str) -> tuple[str, int]:
    """Drop sentences/lines with popularity statements. Returns (clean_text, n_removed)."""
    if not text:
        return "", 0
    kept, removed = [], 0
    for part in _SENTENCE_SPLIT.split(text):
        if not part.strip():
            continue
        if _is_popularity(part):
            removed += 1
        else:
            kept.append(part.strip())
    return " ".join(kept), removed


def sanitize_title(title: str) -> tuple[str, int]:
    removed = 0
    out, n = _BRACKETED.subn("", title)
    removed += n
    out, n = _COUNT.subn("", out)
    removed += n
    return " ".join(out.split()).strip(" -|:"), removed


def sanitize_tags(tags: tuple[str, ...] | list[str]) -> tuple[tuple[str, ...], int]:
    kept = tuple(t for t in tags if not _is_popularity(t))
    return kept, len(tags) - len(kept)
