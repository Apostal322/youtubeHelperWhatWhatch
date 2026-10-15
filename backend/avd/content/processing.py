"""Transcript processing: language identification, normalisation, validation and chunking.

* ``normalize_transcript`` runs at **ingestion** (it may merge/remove segments), so stored segment
  indices — which evidence references — never change afterwards.
* ``validate_transcript`` runs at **evaluation** time and only reports anomalies.
* ``chunk_transcript`` is deterministic: chunks are contiguous ranges of whole segments (so every
  timestamp comes from the source), sized by an estimated token budget, cut preferentially at
  sentence ends and at low lexical-cohesion points (topic shifts, TextTiling-style), with one
  context-only overlap segment that chunk evaluators must not cite.

Token estimates use ``ceil(chars / 4)``; this is a planning approximation, not a tokenizer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

from avd.content.models import Transcript, TranscriptSegment

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Closed-class words are reliable language markers even in short texts.
STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "and",
            "of",
            "to",
            "a",
            "in",
            "is",
            "that",
            "it",
            "you",
            "for",
            "was",
            "on",
            "are",
            "with",
            "as",
            "this",
            "be",
            "at",
            "have",
            "not",
            "but",
        ]
    ),
    "es": frozenset(
        [
            "de",
            "la",
            "que",
            "el",
            "en",
            "y",
            "los",
            "se",
            "del",
            "las",
            "un",
            "por",
            "con",
            "no",
            "una",
            "su",
            "para",
            "es",
            "al",
            "lo",
            "como",
            "más",
            "pero",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "und",
            "in",
            "den",
            "von",
            "zu",
            "das",
            "mit",
            "sich",
            "des",
            "auf",
            "für",
            "ist",
            "im",
            "dem",
            "nicht",
            "ein",
            "eine",
            "als",
            "auch",
            "es",
        ]
    ),
    "fr": frozenset(
        [
            "de",
            "la",
            "le",
            "et",
            "les",
            "des",
            "en",
            "un",
            "une",
            "du",
            "est",
            "que",
            "pour",
            "qui",
            "dans",
            "pas",
            "au",
            "sur",
            "ce",
            "il",
            "avec",
        ]
    ),
    "pt": frozenset(
        [
            "de",
            "a",
            "o",
            "que",
            "e",
            "do",
            "da",
            "em",
            "um",
            "para",
            "é",
            "com",
            "não",
            "uma",
            "os",
            "no",
            "se",
            "na",
            "por",
            "mais",
            "as",
            "dos",
        ]
    ),
    "it": frozenset(
        [
            "di",
            "e",
            "il",
            "la",
            "che",
            "in",
            "un",
            "è",
            "per",
            "non",
            "una",
            "del",
            "con",
            "le",
            "si",
            "da",
            "sono",
            "gli",
            "al",
            "della",
        ]
    ),
    "nl": frozenset(
        [
            "de",
            "en",
            "het",
            "van",
            "een",
            "in",
            "is",
            "dat",
            "op",
            "te",
            "zijn",
            "met",
            "voor",
            "niet",
            "aan",
            "ook",
            "als",
            "er",
            "maar",
        ]
    ),
}


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class LanguageGuess:
    language: str  # ISO 639-1 code or "unknown"
    confidence: float
    scores: dict[str, float]


def identify_language(text: str, min_tokens: int = 20) -> LanguageGuess:
    """Stopword-profile language identification for the languages above; anything else (or text
    too short / too ambiguous) is reported as ``unknown`` rather than guessed."""
    tokens = [t.lower() for t in _WORD.findall(text[:20000])]
    if len(tokens) < min_tokens:
        return LanguageGuess("unknown", 0.0, {})
    scores = {lang: sum(1 for t in tokens if t in sw) / len(tokens) for lang, sw in STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 < 0.08:
        return LanguageGuess("unknown", 0.0, scores)
    confidence = round(min(1.0, (s1 - s2) / s1 * 1.5), 3)
    if confidence < 0.3:
        return LanguageGuess("unknown", confidence, scores)
    return LanguageGuess(best, confidence, {k: round(v, 4) for k, v in scores.items()})


@dataclass
class NormalizationReport:
    removed_empty: int = 0
    merged_duplicates: int = 0
    fixed_overlaps: int = 0
    reordered: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.removed_empty or self.merged_duplicates or self.fixed_overlaps or self.reordered)


def normalize_transcript(t: Transcript) -> tuple[Transcript, NormalizationReport]:
    """Ingestion-time normalisation: whitespace, empty segments, rolling-caption duplicates,
    out-of-order cues and overlaps (start clipped to the previous end)."""
    rep = NormalizationReport()
    segs = list(t.segments)
    if any(b.start < a.start for a, b in pairwise(segs)):
        segs.sort(key=lambda s: (s.start, s.end))
        rep.reordered = True
    out: list[TranscriptSegment] = []
    for s in segs:
        text = " ".join(s.text.split())
        if not text:
            rep.removed_empty += 1
            continue
        if out and text == out[-1].text:
            rep.merged_duplicates += 1
            prev = out[-1]
            out[-1] = TranscriptSegment(start=prev.start, end=max(prev.end, s.end), text=prev.text)
            continue
        start = s.start
        if out and start < out[-1].end:
            start = out[-1].end
            rep.fixed_overlaps += 1
        out.append(TranscriptSegment(start=start, end=max(start, s.end), text=text))
    if not out:
        raise ValueError("transcript has no non-empty segments")
    return Transcript(language=t.language, segments=out), rep


@dataclass
class TranscriptReport:
    segments: int
    words: int
    duration_seconds: float
    estimated_tokens: int
    language: LanguageGuess
    declared_language: str | None
    language_mismatch: bool
    gaps_over_30s: int
    overlapping: int
    non_monotonic: int
    empty_segments: int
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "segments": self.segments,
            "words": self.words,
            "duration_seconds": round(self.duration_seconds, 1),
            "estimated_tokens": self.estimated_tokens,
            "language": self.language.language,
            "language_confidence": self.language.confidence,
            "declared_language": self.declared_language,
            "language_mismatch": self.language_mismatch,
            "gaps_over_30s": self.gaps_over_30s,
            "overlapping": self.overlapping,
            "non_monotonic": self.non_monotonic,
            "empty_segments": self.empty_segments,
            "issues": self.issues,
        }


def validate_transcript(t: Transcript) -> TranscriptReport:
    segs = t.segments
    pairs = list(pairwise(segs))
    lang = identify_language(t.text)
    declared = (t.language or "").split("-")[0].lower() or None
    mismatch = bool(declared and lang.language != "unknown" and lang.language != declared)
    rep = TranscriptReport(
        segments=len(segs),
        words=t.word_count,
        duration_seconds=t.duration,
        estimated_tokens=estimate_tokens(t.text),
        language=lang,
        declared_language=declared,
        language_mismatch=mismatch,
        gaps_over_30s=sum(1 for a, b in pairs if b.start - a.end > 30),
        overlapping=sum(1 for a, b in pairs if b.start < a.end - 0.01),
        non_monotonic=sum(1 for a, b in pairs if b.start < a.start),
        empty_segments=sum(1 for s in segs if not s.text.strip()),
    )
    if rep.words < 50:
        rep.issues.append("very_short_transcript")
    if lang.language == "unknown":
        rep.issues.append("language_unknown")
    if mismatch:
        rep.issues.append("declared_language_mismatch")
    if rep.non_monotonic:
        rep.issues.append("non_monotonic_timestamps")
    return rep


@dataclass(frozen=True)
class Chunk:
    index: int
    first_segment: int  # first segment owned by this chunk (citable)
    last_segment: int  # inclusive
    context_from: int  # first segment shown, including context-only overlap
    start_seconds: float
    end_seconds: float
    estimated_tokens: int
    boundary: str  # why the chunk ended: "topic_shift" | "sentence_end" | "budget" | "end"


def _bag(segs: list[TranscriptSegment]) -> Counter[str]:
    return Counter(w.lower() for s in segs for w in _WORD.findall(s.text) if len(w) > 3)


def _cohesion(a: Counter[str], b: Counter[str]) -> float:
    num = sum(a[k] * b[k] for k in a.keys() & b.keys())
    den = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values())) or 1.0
    return num / den


def chunk_transcript(
    t: Transcript, target_tokens: int = 6000, overlap_segments: int = 1, window: int = 6
) -> list[Chunk]:
    segs = t.segments
    toks = [estimate_tokens(s.text) + 4 for s in segs]  # +4 for the "[i] (t)" prefix
    lo, hi = int(target_tokens * 0.7), int(target_tokens * 1.3)
    chunks: list[Chunk] = []
    start = 0
    while start < len(segs):
        acc, end = 0, start
        candidates: list[tuple[float, int, str]] = []
        while end < len(segs):
            acc += toks[end]
            if acc >= lo:
                left = _bag(segs[max(start, end - window + 1) : end + 1])
                right = _bag(segs[end + 1 : end + 1 + window])
                cohesion = _cohesion(left, right) if right else 0.0
                sentence_end = segs[end].text.rstrip().endswith((".", "?", "!", "…", '"'))
                # Lower is a better boundary: low cohesion (topic shift), at a sentence end.
                candidates.append(
                    (
                        cohesion - (0.15 if sentence_end else 0.0),
                        end,
                        "topic_shift" if cohesion < 0.1 else ("sentence_end" if sentence_end else "budget"),
                    )
                )
            if acc >= hi or end == len(segs) - 1:
                break
            end += 1
        if end == len(segs) - 1 and acc < hi:
            cut, reason = end, "end"
        elif candidates:
            _, cut, reason = min(candidates, key=lambda c: (c[0], -c[1]))
        else:
            cut, reason = end, "budget"
        context_from = max(0, start - overlap_segments) if chunks else start
        chunks.append(
            Chunk(
                index=len(chunks),
                first_segment=start,
                last_segment=cut,
                context_from=context_from,
                start_seconds=segs[start].start,
                end_seconds=segs[cut].end,
                estimated_tokens=sum(toks[context_from : cut + 1]),
                boundary=reason,
            )
        )
        start = cut + 1
    return chunks
