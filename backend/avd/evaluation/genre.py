"""Genre classification.

``KeywordGenreClassifier`` combines the YouTube category id (a weak prior set by the uploader) with
a multilingual keyword lexicon over title, description, tags and (in Stage B) the transcript. It
is transparent and cheap; the ``GenreClassifier`` protocol allows an embedding- or LLM-based
classifier to replace it. Confidence is the softmax-like share of the winning genre and is
reported with every evaluation, because the genre selects the aggregation profile.
"""

from __future__ import annotations

import re
from typing import Protocol

from avd.evaluation.schema import GENRE_PARENT, Genre, GenreAssessment, GenreStatus

# YouTube category ids -> genre prior.
CATEGORY_PRIOR: dict[str, Genre] = {
    "1": Genre.ENTERTAINMENT,  # Film & Animation
    "2": Genre.OTHER,  # Autos
    "10": Genre.MUSIC,
    "15": Genre.OTHER,  # Pets & Animals
    "17": Genre.OTHER,  # Sports
    "19": Genre.OTHER,  # Travel & Events
    "20": Genre.GAMING,
    "22": Genre.OTHER,  # People & Blogs
    "23": Genre.ENTERTAINMENT,  # Comedy
    "24": Genre.ENTERTAINMENT,
    "25": Genre.NEWS_COMMENTARY,
    "26": Genre.HOWTO,
    "27": Genre.EDUCATION,
    "28": Genre.TECHNOLOGY,  # Science & Technology (disambiguated by keywords)
    "29": Genre.DOCUMENTARY,  # Nonprofits & Activism
}

LEXICON: dict[Genre, tuple[str, ...]] = {
    Genre.EDUCATION: (
        "lesson",
        "course",
        "explained",
        "explainer",
        "learn",
        "tutorial",
        "introduction",
        "students",
        "exam",
        "history",
        "theorem",
        "proof",
        "grammar",
        "clase",
        "lección",
        "aprender",
        "vorlesung",
        "lernen",
        "erklärt",
    ),
    Genre.SCIENCE: (
        "physics",
        "chemistry",
        "biology",
        "experiment",
        "scientist",
        "research",
        "quantum",
        "evolution",
        "molecule",
        "cell",
        "astronomy",
        "planet",
        "energy",
        "hypothesis",
        "photosynthesis",
        "neuron",
        "climate",
        "genome",
        "ciencia",
        "física",
        "biología",
        "wissenschaft",
        "physik",
        "experiment",
    ),
    Genre.TECHNOLOGY: (
        "software",
        "computer",
        "algorithm",
        "database",
        "app",
        "hardware",
        "linux",
        "ai",
        "machine learning",
        "network",
        "smartphone",
        "gpu",
        "cpu",
        "tecnología",
        "technik",
        "software",
    ),
    Genre.GAMING: (
        "game",
        "gameplay",
        "let's play",
        "playthrough",
        "speedrun",
        "boss",
        "level",
        "quest",
        "player",
        "gaming",
        "rpg",
        "strategy game",
        "juego",
        "spiel",
    ),
    Genre.ENTERTAINMENT: (
        "prank",
        "challenge",
        "vlog",
        "reaction",
        "laugh",
        "entertainment",
        "show",
    ),
    Genre.DOCUMENTARY: (
        "documentary",
        "investigation",
        "untold",
        "the story of",
        "history of",
        "archive",
        "documental",
        "dokumentation",
    ),
    Genre.ART: (
        "painting",
        "drawing",
        "sculpture",
        "art",
        "artist",
        "illustration",
        "watercolor",
        "canvas",
        "pottery",
        "animation",
        "sketchbook",
        "craft",
        "arte",
        "kunst",
    ),
    Genre.MUSIC: (
        "song",
        "music",
        "guitar",
        "piano",
        "composition",
        "melody",
        "chord",
        "album",
        "lyrics",
        "cover",
        "beat",
        "música",
        "musik",
    ),
    Genre.HOWTO: (
        "how to",
        "step by step",
        "diy",
        "fix",
        "repair",
        "build",
        "install",
        "guide",
        "recipe",
        "cómo",
        "anleitung",
        "tips",
    ),
    Genre.NEWS_COMMENTARY: (
        "news",
        "breaking",
        "policy",
        "election",
        "government",
        "analysis",
        "commentary",
        "report",
        "economy",
        "noticias",
        "nachrichten",
    ),
    Genre.LECTURE: (
        "lecture",
        "lectures",
        "professor",
        "seminar",
        "syllabus",
        "homework",
        "in this lecture",
        "today we will",
        "vorlesung",
        "clase magistral",
    ),
    Genre.PROGRAMMING: (
        "programming",
        "code",
        "coding",
        "compiler",
        "function",
        "python",
        "javascript",
        "typescript",
        "rust",
        "golang",
        "api",
        "framework",
        "library",
        "refactoring",
        "debugging",
        "open source",
        "repository",
        "kubernetes",
        "docker",
        "sql",
        "programación",
    ),
    Genre.PRODUCT_REVIEW: (
        "review",
        "unboxing",
        "hands-on",
        "worth it",
        "pros and cons",
        "should you buy",
        "first impressions",
        "battery life",
        "specs",
        "reseña",
        "testbericht",
    ),
    Genre.INTERVIEW_PODCAST: (
        "podcast",
        "episode",
        "interview",
        "my guest",
        "our guest",
        "welcome to the show",
        "joined by",
        "conversation with",
        "panel",
        "entrevista",
    ),
    Genre.COMEDY: (
        "comedy",
        "stand-up",
        "standup",
        "sketch",
        "funny",
        "jokes",
        "parody",
        "satire",
        "comedia",
    ),
    Genre.PERSONAL_STORYTELLING: (
        "my story",
        "story time",
        "storytime",
        "my journey",
        "my experience",
        "when i was",
        "what happened to me",
        "life update",
    ),
}

_WORD = re.compile(r"[\w'’]+", re.UNICODE)


class GenreClassifier(Protocol):
    def classify(
        self,
        title: str,
        description: str,
        tags: list[str],
        category_id: str | None,
        transcript_text: str | None = None,
    ) -> GenreAssessment: ...


# "Name Surname: text" at the start of a transcript segment/paragraph.
_SPEAKER = re.compile(r"(?m)^\s*([A-Z][\w.'’-]+(?: [A-Z][\w.'’-]+){0,2}):\s")

MIN_EVIDENCE = 4.0  # below this total score (e.g. a single title keyword) the assessment is "ambiguous"
AMBIGUOUS_TOP_SHARE = 0.35
MIXED_SECOND_SHARE = 0.25
MIXED_RATIO = 0.6


def speaker_turn_cue(transcript_text: str | None) -> tuple[int, int]:
    """(distinct labelled speakers, labelled turns) in the first 50k characters."""
    if not transcript_text:
        return 0, 0
    labels = _SPEAKER.findall(transcript_text[:50000].replace(". ", ".\n"))
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    speakers = [k for k, v in counts.items() if v >= 3]
    return len(speakers), sum(counts[k] for k in speakers)


class KeywordGenreClassifier:
    """Genre v2: keyword lexicon over metadata (and transcript in Stage B), a YouTube category
    prior, and a transcript *format* cue (several labelled speakers taking turns indicates an
    interview or discussion). Returns the full distribution and a status:
    ``confident`` / ``mixed`` (two genres carry weight -> blended profile) / ``ambiguous``
    (weak or spread-out evidence -> neutral profile) / ``unknown`` (no evidence)."""

    method = "keyword_lexicon_v2"

    def classify(
        self,
        title: str,
        description: str,
        tags: list[str],
        category_id: str | None,
        transcript_text: str | None = None,
    ) -> GenreAssessment:
        scores: dict[Genre, float] = {}
        prior = CATEGORY_PRIOR.get(category_id or "")
        if prior is not None and prior != Genre.OTHER:
            scores[prior] = scores.get(prior, 0.0) + 2.0
        if category_id == "28":
            scores[Genre.SCIENCE] = scores.get(Genre.SCIENCE, 0.0) + 1.5  # Science & Technology is ambiguous

        fields = [(title, 3.0), (description, 1.0), (" ".join(tags), 1.5)]
        if transcript_text:
            fields.append((transcript_text[:20000], 0.15))
        for text, weight in fields:
            low = " " + " ".join(_WORD.findall(text.lower())) + " "
            for genre, words in LEXICON.items():
                hits = sum(low.count(f" {w} ") for w in words)
                if hits:
                    scores[genre] = scores.get(genre, 0.0) + weight * min(hits, 8)
        speakers, turns = speaker_turn_cue(transcript_text)
        if speakers >= 2 and turns >= 10:
            scores[Genre.INTERVIEW_PODCAST] = scores.get(Genre.INTERVIEW_PODCAST, 0.0) + 3.0

        if not scores:
            return GenreAssessment(
                genre=Genre.OTHER, confidence=0.2, method=self.method, status=GenreStatus.UNKNOWN
            )
        total = sum(scores.values())
        # Ties prefer the more specific v2 genre over its v1 parent (e.g. lecture over education).
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0] not in GENRE_PARENT, kv[0].value))
        dist = {g.value: round(v / total, 3) for g, v in ranked}
        (genre, top), second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        p1, p2 = top / total, (second[1] / total if second else 0.0)
        secondary = None
        if second is not None and p2 >= MIXED_SECOND_SHARE and p2 / p1 >= MIXED_RATIO:
            status, secondary = GenreStatus.MIXED, second[0]
        elif total < MIN_EVIDENCE or p1 < AMBIGUOUS_TOP_SHARE:
            status = GenreStatus.AMBIGUOUS
        else:
            status = GenreStatus.CONFIDENT
        return GenreAssessment(
            genre=genre,
            confidence=round(min(0.95, p1), 3),
            method=self.method,
            scores=dict(list(dist.items())[:4]),
            distribution=dist,
            status=status,
            secondary=secondary,
        )


def apply_override(assessment: GenreAssessment, override: str | None, by: str | None) -> GenreAssessment:
    """A person's genre correction replaces the classifier's choice (the classifier's distribution
    is kept for audit)."""
    if not override:
        return assessment
    return assessment.model_copy(
        update={
            "genre": Genre(override),
            "confidence": 1.0,
            "method": f"manual_override:{by or 'unknown'}",
            "status": GenreStatus.OVERRIDDEN,
            "secondary": None,
        }
    )
