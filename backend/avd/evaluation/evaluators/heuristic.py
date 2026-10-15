"""Deterministic heuristic baseline ("synthetic demo" evaluator).

This is NOT an AI assessment. It maps transparent text statistics (filler ratio, explanatory
connectives, structure markers, lexical diversity, speaking rate, title/transcript overlap) to the
rubric scale. It exists so that:

1. the full system is demonstrable without paid credentials, deterministically;
2. the benchmark harness has a cheap, reproducible baseline to compare LLM evaluators against.

Dimensions that text statistics cannot meaningfully judge (originality, factual accuracy) are
reported as ``insufficient_evidence`` rather than guessed. Lexicons cover English, Spanish and
German; other languages get lower confidence and more abstentions.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from avd.content.models import Transcript
from avd.evaluation.evaluators.base import EvaluatorOutput
from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.profiles import GenreProfile
from avd.evaluation.schema import (
    Dimension,
    DimensionScore,
    DimensionStatus,
    EvidenceRef,
    Modality,
    ReasonCode,
)

METHOD = "heuristic_baseline_v1"
MODEL = "heuristic-baseline"
PREFIX = "[Synthetic heuristic baseline — not an AI assessment] "

_WORD = re.compile(r"[\w'’]+", re.UNICODE)

LEX: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "filler": (
            "um",
            "uh",
            "like",
            "you know",
            "basically",
            "literally",
            "honestly",
            "kind of",
            "sort of",
            "i mean",
            "anyway",
            "stuff",
            "whatever",
        ),
        "hype": (
            "insane",
            "crazy",
            "unbelievable",
            "mind blowing",
            "mind-blowing",
            "epic",
            "secret",
            "shocking",
            "you won't believe",
            "game changer",
            "guys",
            "smash",
            "hype",
            "amazing",
        ),
        "promo": (
            "subscribe",
            "like button",
            "bell",
            "sponsor",
            "sponsored",
            "link in the description",
            "merch",
            "patreon",
            "discount code",
            "comment below",
            "giveaway",
        ),
        "causal": (
            "because",
            "therefore",
            "which means",
            "this means",
            "so that",
            "this is why",
            "the reason",
            "as a result",
            "due to",
            "consequently",
            "that's why",
            "in other words",
            "however",
            "although",
            "whereas",
            "unless",
            "if",
            "then",
        ),
        "structure": (
            "first",
            "firstly",
            "second",
            "secondly",
            "third",
            "next",
            "then",
            "finally",
            "in summary",
            "to summarize",
            "to recap",
            "let's start",
            "in conclusion",
            "part one",
            "step",
            "overview",
            "we'll cover",
            "let's move on",
            "to wrap up",
            "now let's",
        ),
        "example": (
            "for example",
            "for instance",
            "imagine",
            "consider",
            "such as",
            "e.g",
            "let's say",
            "picture",
            "analogy",
            "think of",
        ),
        "definition": (
            "is called",
            "means",
            "is defined as",
            "refers to",
            "we call",
            "known as",
            "in other words",
            "that is",
        ),
        "narrative": (
            "story",
            "imagine",
            "suddenly",
            "it turns out",
            "but then",
            "the problem was",
            "years ago",
            "when i was",
            "what happened",
            "mystery",
            "surprise",
            "twist",
        ),
        "caveat": (
            "however",
            "but",
            "caveat",
            "limitation",
            "not always",
            "depends",
            "uncertain",
            "we don't know",
            "evidence suggests",
            "approximately",
            "roughly",
            "exception",
        ),
    },
    "es": {
        "filler": ("eh", "este", "o sea", "bueno", "pues", "tipo", "como que"),
        "hype": ("increíble", "locura", "secreto", "brutal", "impactante", "chicos"),
        "promo": ("suscríbete", "suscribete", "campanita", "patrocinador", "link en la descripción", "like"),
        "causal": (
            "porque",
            "por lo tanto",
            "esto significa",
            "por eso",
            "debido a",
            "sin embargo",
            "aunque",
            "si",
            "entonces",
        ),
        "structure": (
            "primero",
            "segundo",
            "tercero",
            "luego",
            "después",
            "finalmente",
            "en resumen",
            "para empezar",
            "paso",
            "en conclusión",
        ),
        "example": ("por ejemplo", "imagina", "imaginemos", "como", "pensemos"),
        "definition": ("se llama", "significa", "se define como", "se refiere a", "llamamos"),
        "narrative": ("historia", "de repente", "resulta que", "hace años", "sorpresa"),
        "caveat": ("sin embargo", "pero", "depende", "aproximadamente", "excepción", "no siempre"),
    },
    "de": {
        "filler": ("äh", "ähm", "also", "halt", "irgendwie", "quasi", "sozusagen", "eigentlich"),
        "hype": ("krass", "unglaublich", "wahnsinn", "geheimnis", "mega", "leute"),
        "promo": ("abonniert", "abonnieren", "glocke", "sponsor", "link in der beschreibung", "daumen hoch"),
        "causal": (
            "weil",
            "deshalb",
            "daher",
            "das bedeutet",
            "folglich",
            "jedoch",
            "obwohl",
            "wenn",
            "dann",
        ),
        "structure": (
            "erstens",
            "zweitens",
            "drittens",
            "zuerst",
            "danach",
            "schließlich",
            "zusammenfassend",
            "schritt",
            "zum schluss",
            "fangen wir",
        ),
        "example": ("zum beispiel", "stellt euch vor", "beispielsweise", "nehmen wir"),
        "definition": ("nennt man", "bedeutet", "wird definiert als", "bezeichnet", "heißt"),
        "narrative": ("geschichte", "plötzlich", "es stellte sich heraus", "vor jahren", "überraschung"),
        "caveat": ("jedoch", "aber", "kommt darauf an", "ungefähr", "ausnahme", "nicht immer"),
    },
}

STOPWORDS = {
    "en": set(
        [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "of",
            "to",
            "in",
            "on",
            "at",
            "for",
            "with",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "it",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "my",
            "your",
            "our",
            "their",
            "me",
            "him",
            "her",
            "us",
            "them",
            "so",
            "as",
            "by",
            "from",
            "not",
            "no",
            "do",
            "does",
            "did",
            "have",
            "has",
            "had",
            "can",
            "will",
            "would",
            "could",
            "should",
            "just",
            "about",
            "into",
            "over",
            "than",
            "then",
            "there",
            "here",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "some",
            "more",
            "most",
            "very",
            "really",
            "also",
            "its",
            "it's",
            "i'm",
            "you're",
            "we're",
            "they're",
            "don't",
        ]
    ),
    "es": set(
        [
            "el",
            "la",
            "los",
            "las",
            "un",
            "una",
            "y",
            "o",
            "pero",
            "de",
            "del",
            "a",
            "en",
            "con",
            "por",
            "para",
            "es",
            "son",
            "fue",
            "que",
            "se",
            "lo",
            "como",
            "más",
            "muy",
            "no",
            "sí",
            "este",
            "esta",
            "esto",
        ]
    ),
    "de": set(
        [
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "und",
            "oder",
            "aber",
            "von",
            "zu",
            "in",
            "an",
            "auf",
            "für",
            "mit",
            "ist",
            "sind",
            "war",
            "es",
            "dass",
            "sich",
            "nicht",
            "ja",
            "wie",
            "so",
            "auch",
            "sehr",
        ]
    ),
}


def _clamp(x: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, x))


def _lin(x: float, x0: float, x1: float, y0: float = 1.0, y1: float = 5.0) -> float:
    """Piecewise-linear map of x in [x0, x1] onto [y0, y1], clamped."""
    if x1 == x0:
        return y1
    t = (x - x0) / (x1 - x0)
    return _clamp(y0 + t * (y1 - y0), min(y0, y1), max(y0, y1))


def _count(text: str, phrases: Iterable[str]) -> int:
    padded = f" {text} "
    return sum(padded.count(f" {p} ") for p in phrases)


def _mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-average type-token ratio (length-robust lexical diversity)."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [len(set(tokens[i : i + window])) / window for i in range(0, len(tokens) - window + 1, 10)]
    return sum(ratios) / len(ratios)


def _stem(w: str) -> str:
    return w[:5]


class _Features:
    def __init__(self, inp: EvaluationInput, transcript: Transcript) -> None:
        self.lang = (inp.language or transcript.language or "en").split("-")[0].lower()
        self.lex = LEX.get(self.lang)
        lex = self.lex or LEX["en"]
        self.supported_language = self.lex is not None
        stop = STOPWORDS.get(self.lang, STOPWORDS["en"])

        self.seg_norm = [" ".join(_WORD.findall(s.text.lower())) for s in transcript.segments]
        norm = " ".join(self.seg_norm)
        tokens = norm.split()
        self.words = max(1, len(tokens))
        per100 = 100.0 / self.words

        self.filler = _count(norm, lex["filler"]) * per100
        self.hype = _count(norm, lex["hype"]) * per100
        self.promo = _count(norm, lex["promo"]) * per100
        self.causal = _count(norm, lex["causal"]) * per100
        self.structure = _count(norm, lex["structure"]) * per100
        self.example = _count(norm, lex["example"]) * per100
        self.definition = _count(norm, lex["definition"]) * per100
        self.narrative = _count(norm, lex["narrative"]) * per100
        self.caveat = _count(norm, lex["caveat"]) * per100
        self.questions = transcript.text.count("?") * per100

        content = [t for t in tokens if t not in stop and len(t) > 2]
        self.content_ratio = len(content) / self.words
        self.long_word_ratio = sum(1 for t in content if len(t) >= 9) / self.words
        self.numeric_ratio = sum(1 for t in tokens if any(c.isdigit() for c in t)) * per100
        self.mattr = _mattr(tokens)
        trigrams = list(zip(tokens, tokens[1:], tokens[2:], strict=False))
        tri_counts = Counter(trigrams)
        self.repetition = sum(c - 1 for c in tri_counts.values() if c > 1) / max(1, len(trigrams))

        duration = max(1.0, transcript.duration)
        self.wpm = self.words / (duration / 60.0)
        seg_wpm = [len(s.text.split()) / max(0.5, (s.end - s.start) / 60.0) for s in transcript.segments]
        mean = sum(seg_wpm) / len(seg_wpm)
        self.wpm_cv = (sum((x - mean) ** 2 for x in seg_wpm) / len(seg_wpm)) ** 0.5 / max(1.0, mean)

        title_words = [_stem(w) for w in _WORD.findall(inp.title.lower()) if w not in stop and len(w) >= 4]
        body_stems = {_stem(t) for t in content}
        self.title_terms = len(title_words)
        self.title_overlap = (
            sum(1 for w in title_words if w in body_stems) / len(title_words) if title_words else 0.0
        )
        title_norm = " ".join(_WORD.findall(inp.title.lower()))
        self.title_hype = _count(title_norm, lex["hype"]) + inp.title.count("!") * 0.5

    def segments_with(self, phrases: Iterable[str], limit: int = 2) -> list[int]:
        phrases = tuple(phrases)
        scored = [(sum(f" {s} ".count(f" {p} ") for p in phrases), i) for i, s in enumerate(self.seg_norm)]
        hits = sorted([x for x in scored if x[0] > 0], key=lambda x: (-x[0], x[1]))
        return [i for _, i in hits[:limit]]


class HeuristicTextEvaluator:
    name = "heuristic_text"
    version = METHOD
    synthetic = True
    requires = frozenset({Modality.TRANSCRIPT})
    dimensions: tuple[Dimension, ...] = (
        Dimension.INFORMATION_DENSITY,
        Dimension.DEPTH,
        Dimension.FACTUAL_ACCURACY,
        Dimension.ORIGINALITY,
        Dimension.INFORMATIONAL_VALUE,
        Dimension.CLARITY,
        Dimension.STRUCTURE,
        Dimension.PACING,
        Dimension.STORYTELLING,
        Dimension.TITLE_CONSISTENCY,
    )

    def evaluate(self, inp: EvaluationInput, profile: GenreProfile) -> EvaluatorOutput:
        t = inp.transcript
        if t is None:
            return EvaluatorOutput(
                dimensions=[
                    DimensionScore.abstain(
                        d,
                        DimensionStatus.REQUIRES_MEDIA,
                        "No authorized transcript.",
                        METHOD,
                        missing_inputs=["transcript"],
                    )
                    for d in self.dimensions
                ]
            )
        f = _Features(inp, t)
        lex = f.lex or LEX["en"]
        base_conf = 0.45 if f.supported_language else 0.25
        if f.words < 150:
            base_conf *= 0.5

        def ev(idxs: list[int], note: str | None = None) -> list[EvidenceRef]:
            return [
                EvidenceRef(
                    content_source_id=inp.transcript_source_id,
                    segment_index=i,
                    start_seconds=t.segments[i].start,
                    end_seconds=t.segments[i].end,
                    quote=t.segments[i].text.strip()[:300],
                    note=note,
                    verified=True,
                )
                for i in idxs
            ]

        def scored(
            d: Dimension, score: float, why: str, evidence: list[EvidenceRef], conf: float
        ) -> DimensionScore:
            return DimensionScore(
                dimension=d,
                status=DimensionStatus.SCORED,
                score=round(_clamp(score), 2),
                confidence=round(conf, 3),
                justification=PREFIX + why,
                evidence=evidence,
                method=METHOD,
                model=MODEL,
            )

        noise = f.filler + 1.5 * f.hype + 2.0 * f.promo
        density = (
            _lin(f.content_ratio, 0.30, 0.55) * 0.35
            + _lin(f.long_word_ratio + f.numeric_ratio / 100, 0.02, 0.12) * 0.25
            + _lin(f.mattr, 0.45, 0.75) * 0.2
            + _lin(noise, 6.0, 0.5) * 0.2
        ) - _lin(f.repetition, 0.02, 0.15, 0.0, 1.5)
        depth = (
            _lin(f.causal, 0.5, 3.5) * 0.45
            + _lin(f.caveat, 0.2, 1.5) * 0.2
            + _lin(f.example + f.definition, 0.1, 1.2) * 0.2
            + _lin(f.words, 200, 1500) * 0.15
        ) - _lin(noise, 4.0, 10.0, 0.0, 1.0)
        clarity = (
            _lin(f.example + f.definition, 0.1, 1.2) * 0.4
            + _lin(f.filler, 5.0, 0.5) * 0.3
            + _lin(f.causal, 0.5, 3.0) * 0.3
        )
        structure = _lin(f.structure, 0.2, 2.0) * 0.8 + _lin(f.repetition, 0.15, 0.02) * 0.2
        pacing_dist = abs(f.wpm - 150.0)
        pacing = (
            _lin(pacing_dist, 70.0, 15.0) * 0.6
            + _lin(f.wpm_cv, 0.8, 0.2) * 0.2
            + _lin(f.repetition, 0.15, 0.02) * 0.2
        )
        story = _lin(f.narrative + 0.5 * f.questions, 0.1, 1.5) * 0.7 + _lin(f.structure, 0.2, 1.5) * 0.3
        value = 0.5 * density + 0.5 * depth

        out: list[DimensionScore] = [
            scored(
                Dimension.INFORMATION_DENSITY,
                density,
                f"content-word ratio {f.content_ratio:.2f}, lexical diversity (MATTR) {f.mattr:.2f}, "
                f"filler/hype/promo {noise:.1f} per 100 words, trigram repetition {f.repetition:.3f}.",
                ev(f.segments_with(lex["filler"] + lex["hype"] + lex["promo"], 1), "most filler/promo")
                or ev([0]),
                base_conf,
            ),
            scored(
                Dimension.DEPTH,
                depth,
                f"explanatory connectives {f.causal:.2f}/100 words, caveats {f.caveat:.2f}/100, "
                f"examples+definitions {f.example + f.definition:.2f}/100, {f.words} words.",
                ev(f.segments_with(lex["causal"] + lex["caveat"])),
                base_conf * 0.9,
            ),
            DimensionScore.abstain(
                Dimension.FACTUAL_ACCURACY,
                DimensionStatus.INSUFFICIENT_EVIDENCE,
                PREFIX + "Text statistics cannot verify factual claims.",
                METHOD,
                model=MODEL,
                reason_code=ReasonCode.METHOD_CANNOT_JUDGE,
            ),
            DimensionScore.abstain(
                Dimension.ORIGINALITY,
                DimensionStatus.INSUFFICIENT_EVIDENCE,
                PREFIX + "Text statistics cannot judge originality.",
                METHOD,
                model=MODEL,
                reason_code=ReasonCode.METHOD_CANNOT_JUDGE,
            ),
            scored(
                Dimension.INFORMATIONAL_VALUE,
                value,
                "mean of the information-density and depth heuristics.",
                [],
                base_conf * 0.8,
            ),
            scored(
                Dimension.CLARITY,
                clarity,
                f"examples+definitions {f.example + f.definition:.2f}/100 words, filler "
                f"{f.filler:.2f}/100, explanatory connectives {f.causal:.2f}/100.",
                ev(f.segments_with(lex["example"] + lex["definition"])),
                base_conf * 0.9,
            ),
            scored(
                Dimension.STRUCTURE,
                structure,
                f"structure/signposting markers {f.structure:.2f}/100 words.",
                ev(f.segments_with(lex["structure"])),
                base_conf,
            ),
            scored(
                Dimension.PACING,
                pacing,
                f"speaking rate {f.wpm:.0f} wpm (reference ~150), per-segment rate variation {f.wpm_cv:.2f}.",
                [],
                base_conf * 0.8,
            ),
            scored(
                Dimension.STORYTELLING,
                story,
                f"narrative markers {f.narrative:.2f}/100 words, questions {f.questions:.2f}/100.",
                ev(f.segments_with(lex["narrative"])),
                base_conf * 0.7,
            ),
        ]
        if f.title_terms == 0:
            out.append(
                DimensionScore.abstain(
                    Dimension.TITLE_CONSISTENCY,
                    DimensionStatus.INSUFFICIENT_EVIDENCE,
                    PREFIX + "Title has no content terms to compare.",
                    METHOD,
                    model=MODEL,
                )
            )
        else:
            title = _lin(f.title_overlap, 0.2, 0.8) - min(1.5, 0.75 * f.title_hype) * (1.0 - f.title_overlap)
            out.append(
                scored(
                    Dimension.TITLE_CONSISTENCY,
                    title,
                    f"{f.title_overlap:.0%} of title content terms appear in the transcript; "
                    f"title hype markers {f.title_hype:.1f}.",
                    [],
                    base_conf,
                )
            )
        return EvaluatorOutput(dimensions=out, models=[MODEL])
