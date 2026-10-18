"""Application services shared by the API, CLI and Celery tasks."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from avd.config import Settings
from avd.content.models import SourceType, Transcript
from avd.content.parsing import parse_timed_text, transcript_from_plain_text
from avd.content.store import add_transcript_source
from avd.db.models import ContentSource, Video
from avd.evaluation.schema import FRAMEWORK_VERSION
from avd.ingest import load_fixture_files, register_discovered, seed_fixture_video
from avd.workers.queue import enqueue
from avd.youtube.discovery import ensure_seed_queries, next_queries


def plan_discovery(session: Session, n_queries: int = 4) -> int:
    ensure_seed_queries(session)
    bucket = datetime.now(UTC).strftime("%Y%m%d%H")
    created = 0
    for q in next_queries(session, n_queries):
        _, new = enqueue(
            session, "discover", f"discover:{q.id}:{bucket}", payload={"query_id": q.id}, priority=6.0
        )
        created += int(new)
    return created


def enqueue_stage_a(
    session: Session,
    video_id: str,
    priority: float = 10.0,
    force: bool = False,
    payload: dict[str, Any] | None = None,
) -> bool:
    _, created = enqueue(
        session,
        "stage_a",
        f"stage_a:{video_id}:{FRAMEWORK_VERSION}",
        video_id=video_id,
        priority=priority,
        payload=payload,
        requeue_terminal=force,
    )
    return created


def seed_demo(session: Session, settings: Settings) -> int:
    """Load the synthetic fixture catalog and queue it for processing."""
    fixtures = load_fixture_files(settings.fixtures_dir)
    for fx in fixtures:
        video = seed_fixture_video(session, fx, settings.media_cache_dir)
        enqueue_stage_a(session, video.id)
    session.flush()
    return len(fixtures)


class TranscriptSubmissionError(ValueError):
    pass


def submit_user_transcript(
    session: Session,
    *,
    user_id: str,
    video_id: str,
    content: str,
    fmt: str,
    language: str | None,
    attestation: str,
    youtube_enabled: bool,
) -> tuple[ContentSource, bool]:
    """Store a user-provided transcript after an explicit rights attestation, then (re)queue the
    video. Plain-text transcripts get *estimated* timings, which is recorded in the source's
    authorization metadata."""
    if len(attestation.strip()) < 10:
        raise TranscriptSubmissionError(
            "an attestation describing your right to submit this transcript is required"
        )
    if fmt in ("vtt", "srt"):
        transcript = parse_timed_text(content, language)
        timing = "timed"
    elif fmt == "text":
        transcript = transcript_from_plain_text(content, language)
        timing = "timing_estimated"
    elif fmt == "json":
        transcript = Transcript.model_validate_json(content)
        timing = "timed"
    else:
        raise TranscriptSubmissionError(f"unsupported format {fmt!r}")
    if transcript.word_count < 50:
        raise TranscriptSubmissionError("transcript is too short to evaluate (< 50 words)")
    video = session.get(Video, video_id)
    if video is None:
        register_discovered(session, video_id, None, "user_submission")
        if youtube_enabled:
            enqueue(
                session,
                "ingest_metadata",
                f"ingest:user:{video_id}",
                payload={"video_ids": [video_id]},
                priority=9.0,
            )
    row, created = add_transcript_source(
        session,
        video_id=video_id,
        transcript=transcript,
        source_type=SourceType.USER_PROVIDED_TRANSCRIPT,
        authorization_basis=f"User attestation ({timing}): {attestation.strip()[:1000]}",
        license="user-attested",
        provided_by_user_id=user_id,
    )
    if created:
        enqueue_stage_a(session, video_id, priority=9.5, force=True)
    return row, created


MEDIA_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
}


def submit_user_media(
    session: Session,
    *,
    user_id: str,
    video_id: str,
    path: Path,
    attestation: str,
    media_dir: Path,
    youtube_enabled: bool,
) -> tuple[ContentSource, bool]:
    """Store user-submitted audio/video after an explicit rights attestation (e.g. the user is the
    creator), then (re)queue the video. Media is never fetched from YouTube."""
    from avd.content.store import add_media_source

    if len(attestation.strip()) < 10:
        raise TranscriptSubmissionError(
            "an attestation describing your right to submit this media is required"
        )
    if session.get(Video, video_id) is None:
        register_discovered(session, video_id, None, "user_submission")
        if youtube_enabled:
            enqueue(
                session,
                "ingest_metadata",
                f"ingest:user:{video_id}",
                payload={"video_ids": [video_id]},
                priority=9.0,
            )
    row, created = add_media_source(
        session,
        video_id=video_id,
        path=path,
        media_dir=media_dir,
        source_type=SourceType.USER_PROVIDED_MEDIA,
        authorization_basis=f"User attestation (media): {attestation.strip()[:1000]}",
        license="user-attested",
        provided_by_user_id=user_id,
    )
    if created:
        enqueue_stage_a(session, video_id, priority=9.5, force=True)
    return row, created
