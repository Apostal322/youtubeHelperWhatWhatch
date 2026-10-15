"""Stratified, popularity-blind discovery planning.

Seed queries are spread across genres and languages. Each query is walked *by upload date*
(``order=date``) backwards in time using ``publishedBefore`` cursors, which samples uploads
regardless of their popularity — unlike ``order=viewCount`` or default relevance, which would
over-represent already-popular channels. Queries are scheduled round-robin (least recently run
first) so no stratum starves.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.db.models import DiscoveryQuery

DEFAULT_SEEDS: dict[str, dict[str, list[str]]] = {
    "science": {
        "en": ["physics explained lecture", "biology explainer", "chemistry demonstration explained"],
        "es": ["física explicada", "biología explicación"],
        "de": ["physik erklärt", "biologie erklärt"],
    },
    "education": {
        "en": ["history lecture", "mathematics proof explained", "linguistics lesson"],
        "es": ["clase de historia", "matemáticas explicación"],
        "de": ["geschichte vorlesung", "mathe erklärt"],
    },
    "technology": {
        "en": [
            "programming tutorial data structures",
            "computer architecture explained",
            "open source software review",
        ],
        "es": ["tutorial programación"],
        "de": ["programmieren tutorial"],
    },
    "howto": {
        "en": ["woodworking how to", "home repair step by step", "bread baking tutorial"],
        "de": ["sauerteig anleitung"],
    },
    "documentary": {
        "en": ["independent documentary", "local history documentary"],
        "es": ["documental independiente"],
    },
    "gaming": {"en": ["game design analysis", "indie game let's play commentary"]},
    "entertainment": {"en": ["short comedy sketch", "animated short film"]},
    "art": {"en": ["painting process", "pottery wheel throwing", "illustration process"]},
    "music": {"en": ["original song acoustic", "music theory analysis"]},
    "news_commentary": {"en": ["policy analysis explained", "local news commentary"]},
}


def ensure_seed_queries(session: Session, seeds: dict[str, dict[str, list[str]]] | None = None) -> int:
    created = 0
    for genre, by_lang in (seeds or DEFAULT_SEEDS).items():
        for lang, queries in by_lang.items():
            for q in queries:
                exists = session.scalar(
                    select(DiscoveryQuery).where(
                        DiscoveryQuery.query == q,
                        DiscoveryQuery.language == lang,
                        DiscoveryQuery.order == "date",
                    )
                )
                if exists is None:
                    session.add(DiscoveryQuery(query=q, genre=genre, language=lang, order="date"))
                    created += 1
    session.flush()
    return created


def next_queries(session: Session, n: int) -> list[DiscoveryQuery]:
    """Least-recently-run enabled queries first (never-run queries before all others)."""
    rows = session.scalars(select(DiscoveryQuery).where(DiscoveryQuery.enabled.is_(True))).all()
    epoch = datetime(1970, 1, 1, tzinfo=UTC)

    def key(q: DiscoveryQuery) -> datetime:
        ts = q.last_run_at
        if ts is None:
            return epoch
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)

    return sorted(rows, key=key)[:n]
