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

from avd.evaluation.schema import Genre, GenreAssessment

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
        "lecture",
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
        "programming",
        "code",
        "computer",
        "algorithm",
        "database",
        "app",
        "hardware",
        "linux",
        "python",
        "javascript",
        "ai",
        "machine learning",
        "network",
        "smartphone",
        "review",
        "gpu",
        "cpu",
        "tecnología",
        "programación",
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
        "funny",
        "comedy",
        "sketch",
        "prank",
        "challenge",
        "vlog",
        "reaction",
        "story time",
        "storytime",
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
        "interview",
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


class KeywordGenreClassifier:
    method = "keyword_lexicon_v1"

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

        if not scores:
            return GenreAssessment(genre=Genre.OTHER, confidence=0.2, method=self.method, scores={})
        total = sum(scores.values())
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0].value))
        genre, top = ranked[0]
        confidence = round(min(0.95, top / total), 3)
        return GenreAssessment(
            genre=genre,
            confidence=confidence,
            method=self.method,
            scores={g.value: round(s / total, 3) for g, s in ranked[:4]},
        )
