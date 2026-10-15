"""Vector search over video embeddings: pgvector (HNSW, cosine) on PostgreSQL, exact numpy scan
otherwise. Only vectors produced by the active embedding model are compared."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import Float, Select, bindparam, select
from sqlalchemy.orm import Session

from avd.db.models import Channel, Evaluation, Video
from avd.db.types import EmbeddingVector
from avd.evaluation.schema import expand_genres


@dataclass
class SearchFilters:
    genres: list[str] | None = None
    languages: list[str] | None = None
    evaluated_only: bool = True
    include_synthetic: bool = True
    min_duration_seconds: int | None = None
    max_duration_seconds: int | None = None
    exclude_ids: set[str] | None = None


def _apply_filters(q: Select, f: SearchFilters, embedding_model: str) -> Select:
    q = q.where(Video.embedding.is_not(None), Video.embedding_model == embedding_model)
    if f.evaluated_only:
        q = q.join(Evaluation, (Evaluation.video_id == Video.id) & Evaluation.is_current.is_(True)).where(
            Evaluation.status.in_(("complete", "partial"))
        )
    if f.genres:
        q = q.where(Video.genre.in_(expand_genres(f.genres)))
    if f.languages:
        q = q.where(Video.language.in_(f.languages))
    if not f.include_synthetic:
        q = q.where(Video.is_synthetic.is_(False))
    if f.max_duration_seconds:
        q = q.where((Video.duration_seconds.is_(None)) | (Video.duration_seconds <= f.max_duration_seconds))
    if f.min_duration_seconds:
        q = q.where((Video.duration_seconds.is_(None)) | (Video.duration_seconds >= f.min_duration_seconds))
    if f.exclude_ids:
        q = q.where(Video.id.not_in(f.exclude_ids))
    return q


def vector_search(
    session: Session,
    query: list[float],
    k: int,
    embedding_model: str,
    filters: SearchFilters | None = None,
) -> list[tuple[str, float]]:
    """Return [(video_id, cosine_similarity)] best first."""
    f = filters or SearchFilters()
    dialect = session.bind.dialect.name if session.bind is not None else "sqlite"
    if dialect == "postgresql":
        qv = bindparam("qv", query, type_=EmbeddingVector())
        dist = Video.embedding.op("<=>", return_type=Float)(qv)
        q = _apply_filters(select(Video.id, dist.label("d")), f, embedding_model).order_by(dist).limit(k)
        return [(vid, 1.0 - float(d)) for vid, d in session.execute(q).all()]
    rows = session.execute(_apply_filters(select(Video.id, Video.embedding), f, embedding_model)).all()
    if not rows:
        return []
    ids = [r[0] for r in rows]
    mat = np.array([r[1] for r in rows], dtype=np.float64)
    qa = np.array(query, dtype=np.float64)
    sims = mat @ qa / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(qa) or 1.0) + 1e-12)
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def channel_size(session: Session, video: Video) -> str:
    ch = session.get(Channel, video.channel_id)
    return ch.size_bucket if ch else "unknown"
