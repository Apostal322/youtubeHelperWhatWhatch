"""Topic search over analyzed videos (semantic vector search + lexical title boost)."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.datasets.blinding import blinded_video_ids
from avd.db.models import Evaluation, Video
from avd.embeddings.base import Embedder
from avd.evaluation.service import summary
from avd.recsys.index import SearchFilters, vector_search
from avd.recsys.recommender import intrinsic_value, video_card

_TOK = re.compile(r"\w+", re.UNICODE)
QUALITY_TIEBREAK = 0.15
TITLE_BOOST = 0.2
MAX_WINDOW = 5000
MAX_BONUS = TITLE_BOOST + QUALITY_TIEBREAK  # title_match and quality_norm are at most 1
# Function words that matched almost every title ("what", "and", "the") and boosted unrelated ones.
_STOPWORDS = frozenset(
    "the and for with what how why who are was were can cannot not you your this that from into about "  # noqa: SIM905
    "have has had will would should could does did its our their there here than then them they".split()
)


def search(
    session: Session,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 20,
    genres: list[str] | None = None,
    languages: list[str] | None = None,
    min_quality: float | None = None,
    include_synthetic: bool = True,
) -> list[dict[str, Any]]:
    qvec = embedder.embed([query])[0]
    filters = SearchFilters(genres=genres, languages=languages, include_synthetic=include_synthetic)
    q_tokens = {t for t in _TOK.findall(query.lower()) if len(t) > 2 and t not in _STOPWORDS}
    # Below the embedder's calibrated floor a similarity means "unrelated" (the recommender uses
    # the same floor). Without it every query, even nonsense, returned the whole catalog.
    floor = float(getattr(embedder, "relevance_floor", 0.0))
    blinded = blinded_video_ids(session)
    # Results are re-ranked after the vector search (filters, title and quality bonuses), so a
    # fixed window could miss videos that belong on the page: filtered-out videos used it up, or
    # a video just outside it had a bonus that lifted it above the last one shown. Widen the window
    # until the page is full and nothing outside the window can outscore its last result (hits come
    # best first, and no bonus exceeds MAX_BONUS), or until the window covers the whole index.
    window = limit * 4
    while True:
        hits = vector_search(session, qvec, window, embedder.model_id, filters)
        results = sorted(
            _score(session, hits, q_tokens, floor, blinded, min_quality), key=lambda r: -r["match"]["score"]
        )
        if len(hits) < window or window >= MAX_WINDOW:
            break
        if len(results) >= limit and hits[-1][1] + MAX_BONUS <= results[limit - 1]["match"]["score"]:
            break
        window *= 4
    return results[:limit]


def _score(
    session: Session,
    hits: list[tuple[str, float]],
    q_tokens: set[str],
    floor: float,
    blinded: set[str] | frozenset[str],
    min_quality: float | None,
) -> list[dict[str, Any]]:
    if not hits:
        return []
    ids = [h[0] for h in hits]
    videos = {v.id: v for v in session.scalars(select(Video).where(Video.id.in_(ids))).all()}
    evals = {
        e.video_id: e
        for e in session.scalars(
            select(Evaluation).where(Evaluation.video_id.in_(ids), Evaluation.is_current.is_(True))
        ).all()
    }
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for vid, sim in hits:
        v, ev = videos.get(vid), evals.get(vid)
        if v is None or ev is None or vid in blinded or vid in seen:
            continue
        seen.add(vid)
        q = intrinsic_value(ev)
        if min_quality is not None and (q is None or q < min_quality):
            continue
        title_tokens = set(_TOK.findall(v.title.lower()))
        lexical = len(q_tokens & title_tokens) / len(q_tokens) if q_tokens else 0.0
        if sim < floor and lexical == 0.0:
            continue
        quality_norm = 0.0 if q is None else (q - 1) / 4
        score = sim + TITLE_BOOST * lexical + QUALITY_TIEBREAK * quality_norm
        results.append(
            {
                "video": video_card(session, v),
                "intrinsic_quality": summary(ev),
                "match": {
                    "semantic_similarity": round(sim, 4),
                    "title_match": round(lexical, 3),
                    "score": round(score, 4),
                },
            }
        )
    return results
