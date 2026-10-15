"""Persisting and loading authorized content."""

from __future__ import annotations

import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.content.models import (
    ContentBundle,
    SourceRef,
    SourceType,
    Transcript,
    sha256_file,
)
from avd.db.models import ContentSource
from avd.evaluation.schema import Modality

# When several transcripts exist, prefer the most authoritative one.
SOURCE_PRIORITY = {
    SourceType.CREATOR_AUTHORIZED_CAPTIONS: 0,
    SourceType.LICENSED_DATASET: 1,
    SourceType.USER_PROVIDED_TRANSCRIPT: 2,
    SourceType.SYNTHETIC_FIXTURE: 3,
}


def add_transcript_source(
    session: Session,
    *,
    video_id: str,
    transcript: Transcript,
    source_type: SourceType,
    authorization_basis: str,
    license: str | None = None,
    provided_by_user_id: str | None = None,
    media_uri: str | None = None,
    extra_modalities: list[Modality] | None = None,
) -> tuple[ContentSource, bool]:
    """Store a transcript source; returns (row, created). Identical content is deduplicated."""
    if not authorization_basis.strip():
        raise ValueError("an authorization basis is required for every content source")
    content_hash = transcript.content_hash()
    if media_uri:
        content_hash = sha256_file(media_uri)[:32] + content_hash[:32]
    existing = session.scalar(
        select(ContentSource).where(
            ContentSource.video_id == video_id,
            ContentSource.source_type == source_type.value,
            ContentSource.content_hash == content_hash,
        )
    )
    if existing is not None:
        return existing, False
    modalities = [Modality.TRANSCRIPT.value] + [m.value for m in (extra_modalities or [])]
    row = ContentSource(
        video_id=video_id,
        source_type=source_type.value,
        authorization_basis=authorization_basis,
        license=license,
        provided_by_user_id=provided_by_user_id,
        modalities=modalities,
        language=transcript.language,
        content_hash=content_hash,
        transcript=transcript.to_json(),
        media_uri=media_uri,
    )
    session.add(row)
    session.flush()
    return row, True


def active_sources(session: Session, video_id: str) -> list[ContentSource]:
    rows = session.scalars(
        select(ContentSource).where(ContentSource.video_id == video_id, ContentSource.status == "active")
    ).all()
    return sorted(rows, key=lambda r: (SOURCE_PRIORITY.get(SourceType(r.source_type), 9), r.created_at))


def available_modalities(session: Session, video_id: str) -> set[Modality]:
    mods: set[Modality] = set()
    for row in active_sources(session, video_id):
        for m in row.modalities:
            if m == Modality.AUDIO.value and not (row.media_uri and os.path.exists(row.media_uri)):
                continue
            mods.add(Modality(m))
    return mods


def load_bundle(session: Session, video_id: str) -> ContentBundle:
    rows = active_sources(session, video_id)
    bundle = ContentBundle(video_id=video_id, sources=[])
    used: list[ContentSource] = []
    for row in rows:
        if bundle.transcript is None and row.transcript:
            bundle.transcript = Transcript.model_validate(row.transcript)
            bundle.transcript_source_id = row.id
            used.append(row)
        if (
            bundle.audio_path is None
            and Modality.AUDIO.value in row.modalities
            and row.media_uri
            and os.path.exists(row.media_uri)
        ):
            bundle.audio_path = row.media_uri
            bundle.audio_sha256 = sha256_file(row.media_uri)
            bundle.audio_source_id = row.id
            if row not in used:
                used.append(row)
    bundle.sources = [
        SourceRef(
            id=r.id,
            source_type=SourceType(r.source_type),
            authorization_basis=r.authorization_basis,
            license=r.license,
        )
        for r in used
    ]
    return bundle
