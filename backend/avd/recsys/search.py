"""Topic search over analyzed videos (semantic vector search + lexical title boost)."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.db.models import Evaluation, Video
from avd.embeddings.base import Embedder
from avd.evaluation.service import summary
from avd.recsys.index import SearchFilters, vector_search
from avd.recsys.recommender import intrinsic_value, video_card

_TOK = re.compile(r"\w+", re.UNICODE)
QUALITY_TIEBREAK = 0.15
TITLE_BOOST = 0.2


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
    hits = vector_search(
        session,
        qvec,
        limit * 4,
        embedder.model_id,
        SearchFilters(genres=genres, languages=languages, include_synthetic=include_synthetic),
    )
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
    q_tokens = {t for t in _TOK.findall(query.lower()) if len(t) > 2}
    results: list[dict[str, Any]] = []
    for vid, sim in hits:
        v, ev = videos.get(vid), evals.get(vid)
        if v is None or ev is None:
            continue
        q = intrinsic_value(ev)
        if min_quality is not None and (q is None or q < min_quality):
            continue
        title_tokens = set(_TOK.findall(v.title.lower()))
        lexical = len(q_tokens & title_tokens) / len(q_tokens) if q_tokens else 0.0
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
    results.sort(key=lambda r: -r["match"]["score"])
    return results[:limit]
