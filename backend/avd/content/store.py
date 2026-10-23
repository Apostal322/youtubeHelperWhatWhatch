"""Persisting and loading authorized content."""

from __future__ import annotations

import os
from pathlib import Path

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
    SourceType.USER_PROVIDED_MEDIA: 2,
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
        reason = (existing.preprocessing or {}).get("revocation_reason")
        if existing.status == "revoked" and reason == "user account deleted" and not media_uri:
            # The account that first submitted this text was deleted. Someone else submitting it
            # now is a new, attested source (the unique key allows one row per text and video).
            existing.status = "active"
            existing.revoked_at = None
            existing.transcript = transcript.to_json()
            existing.authorization_basis = authorization_basis
            existing.license = license
            existing.provided_by_user_id = provided_by_user_id
            existing.preprocessing = {
                k: v for k, v in (existing.preprocessing or {}).items() if k != "revocation_reason"
            }
            session.flush()
            return existing, True
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


def _audio_file(row: ContentSource) -> str | None:
    """A decodable WAV for the audio evaluator: the extracted track of a media upload, or a WAV
    source itself."""
    derived = (row.preprocessing or {}).get("audio_wav")
    for cand in (derived, row.media_uri):
        if cand and str(cand).lower().endswith(".wav") and os.path.exists(cand):
            return str(cand)
    return None


def _video_file(row: ContentSource) -> str | None:
    if Modality.VIDEO.value in row.modalities and row.media_uri and os.path.exists(row.media_uri):
        return row.media_uri
    return None


def available_modalities(session: Session, video_id: str) -> set[Modality]:
    mods: set[Modality] = set()
    for row in active_sources(session, video_id):
        for m in row.modalities:
            if m == Modality.AUDIO.value and _audio_file(row) is None:
                continue
            if m == Modality.VIDEO.value and _video_file(row) is None:
                continue
            mods.add(Modality(m))
    return mods


def add_media_source(
    session: Session,
    *,
    video_id: str,
    path: Path,
    media_dir: Path,
    source_type: SourceType,
    authorization_basis: str,
    provided_by_user_id: str | None = None,
    license: str | None = None,
) -> tuple[ContentSource, bool]:
    """Register an authorized audio/video file. The file is moved to a content-addressed name in
    ``media_dir``; its audio track is decoded to WAV for the audio evaluator. Raises
    ``media.MediaError`` for files that are not audio/video."""
    from avd.content import media

    if not authorization_basis.strip():
        raise ValueError("an authorization basis is required for every content source")
    digest = sha256_file(str(path))
    existing = session.scalar(
        select(ContentSource).where(ContentSource.video_id == video_id, ContentSource.content_hash == digest)
    )
    if existing is not None:
        path.unlink(missing_ok=True)
        return existing, False
    media_dir.mkdir(parents=True, exist_ok=True)
    final = media_dir / f"{digest}{path.suffix.lower()[:8] or '.bin'}"
    path.replace(final)
    try:
        info = media.probe(final, root=media_dir)
        wav = (
            media.extract_audio_wav(final, media_dir / f"{digest}.audio.wav", root=media_dir)
            if info.has_audio
            else None
        )
    except media.MediaError:
        final.unlink(missing_ok=True)
        raise
    modalities = ([Modality.AUDIO.value] if wav else []) + ([Modality.VIDEO.value] if info.has_video else [])
    row = ContentSource(
        video_id=video_id,
        source_type=source_type.value,
        authorization_basis=authorization_basis,
        license=license,
        provided_by_user_id=provided_by_user_id,
        modalities=modalities,
        content_hash=digest,
        media_uri=str(final),
        raw_sha256=digest,
        preprocessing={
            "audio_wav": str(wav) if wav else None,
            "probe": {
                "duration_seconds": info.duration_seconds,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "format": info.format_name,
            },
        },
    )
    session.add(row)
    session.flush()
    return row, True


def load_bundle(session: Session, video_id: str) -> ContentBundle:
    rows = active_sources(session, video_id)
    bundle = ContentBundle(video_id=video_id, sources=[])
    used: list[ContentSource] = []
    for row in rows:
        if bundle.transcript is None and row.transcript:
            bundle.transcript = Transcript.model_validate(row.transcript)
            bundle.transcript_source_id = row.id
            used.append(row)
        audio = _audio_file(row) if Modality.AUDIO.value in row.modalities else None
        if bundle.audio_path is None and audio is not None:
            bundle.audio_path = audio
            bundle.audio_sha256 = sha256_file(audio)
            bundle.audio_source_id = row.id
            if row not in used:
                used.append(row)
        video = _video_file(row)
        if bundle.video_path is None and video is not None:
            bundle.video_path = video
            bundle.video_sha256 = row.raw_sha256 or sha256_file(video)
            bundle.video_source_id = row.id
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
