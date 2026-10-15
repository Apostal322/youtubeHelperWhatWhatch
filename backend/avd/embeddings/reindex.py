"""Safe re-indexing after an embedding model or recipe change.

Rows are re-embedded in batches, and each batch is committed, so an interrupted run resumes where
it stopped. Queries only compare vectors from the active model (``Video.embedding_model``). During
a migration, rows that still hold old-model vectors are therefore *excluded* from search; they are
never compared across vector spaces. The report shows how many rows are still pending.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.content.store import load_bundle
from avd.db.models import Video
from avd.embeddings import documents
from avd.embeddings.base import Embedder


def reindex(
    session_factory: Callable[[], Session], embedder: Embedder, batch_size: int = 64, dry_run: bool = False
) -> dict[str, Any]:
    with session_factory() as s:
        ids = list(s.scalars(select(Video.id).order_by(Video.id)).all())
    checked = reembedded = 0
    for i in range(0, len(ids), batch_size):
        with session_factory() as s:
            for vid in ids[i : i + batch_size]:
                video = s.get(Video, vid)
                if video is None:
                    continue
                checked += 1
                transcript = load_bundle(s, vid).transcript
                if documents.is_stale(embedder, video, transcript):
                    reembedded += 1
                    if not dry_run:
                        documents.apply(video, documents.embed_video(embedder, video, transcript))
            if not dry_run:
                s.commit()
    with session_factory() as s:
        other = s.scalar(
            select(func.count())
            .select_from(Video)
            .where(Video.embedding.is_not(None), Video.embedding_model != embedder.model_id)
        )
    return {
        "model_id": embedder.model_id,
        "preprocessing": embedder.preprocessing,
        "checked": checked,
        "reembedded" if not dry_run else "would_reembed": reembedded,
        "rows_with_other_model_vectors": int(other or 0),
        "dry_run": dry_run,
    }
