"""Metadata ingestion, fixture seeding and licensed-dataset import."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from avd.content.models import SourceType, Transcript
from avd.content.parsing import parse_timed_text
from avd.content.store import add_transcript_source
from avd.content.synthetic_audio import generate, write_wav
from avd.db.models import Channel, Video, VideoStats
from avd.evaluation.schema import Modality
from avd.logging import get_logger
from avd.youtube.client import ChannelMetadata, VideoMetadata

log = get_logger(__name__)

SIZE_BUCKETS = [
    (1_000, "micro"),
    (10_000, "small"),
    (100_000, "medium"),
    (1_000_000, "large"),
]


def size_bucket(subscribers: int | None) -> str:
    """Channel size strata used for discovery coverage monitoring only."""
    if subscribers is None:
        return "unknown"
    for limit, name in SIZE_BUCKETS:
        if subscribers < limit:
            return name
    return "huge"


def upsert_channel(
    session: Session,
    channel_id: str,
    title: str,
    subscriber_count: int | None = None,
    *,
    is_synthetic: bool = False,
    country: str | None = None,
) -> Channel:
    ch = session.get(Channel, channel_id)
    if ch is None:
        ch = Channel(id=channel_id, title=title, is_synthetic=is_synthetic)
        session.add(ch)
    ch.title = title or ch.title
    if subscriber_count is not None:
        ch.subscriber_count = subscriber_count
        ch.size_bucket = size_bucket(subscriber_count)
        ch.stats_fetched_at = datetime.now(UTC)
    if country:
        ch.country = country
    session.flush()
    return ch


def apply_channel_metadata(session: Session, meta: ChannelMetadata) -> Channel:
    return upsert_channel(session, meta.channel_id, meta.title, meta.subscriber_count, country=meta.country)


def apply_video_metadata(session: Session, meta: VideoMetadata, discovery_source: str) -> Video:
    upsert_channel(session, meta.channel_id, meta.channel_title)
    video = session.get(Video, meta.video_id)
    if video is None:
        video = Video(
            id=meta.video_id, channel_id=meta.channel_id, title=meta.title, discovery_source=discovery_source
        )
        session.add(video)
    video.channel_id = meta.channel_id  # replaces the 'unknown' placeholder from search results
    video.title = meta.title
    video.description = meta.description
    video.published_at = meta.published_at
    video.duration_seconds = meta.duration_seconds
    video.language = meta.language
    video.youtube_category_id = meta.category_id
    video.tags = meta.tags
    video.metadata_fetched_at = datetime.now(UTC)
    if video.status == "discovered":
        video.status = "metadata_ready"
    session.add(
        VideoStats(
            video_id=meta.video_id,
            view_count=meta.view_count,
            like_count=meta.like_count,
            comment_count=meta.comment_count,
        )
    )
    session.flush()
    return video


def register_discovered(session: Session, video_id: str, channel_id: str | None, source: str) -> bool:
    """Record a discovered id (deduplicated). Returns True if new."""
    if session.get(Video, video_id) is not None:
        return False
    if channel_id is None:
        channel_id = "unknown"
    upsert_channel(session, channel_id, "")
    session.add(Video(id=video_id, channel_id=channel_id, title="", discovery_source=source))
    session.flush()
    return True


# --------------------------------------------------------------------------- synthetic fixtures


def load_fixture_files(fixtures_dir: Path) -> list[dict[str, Any]]:
    files = sorted((fixtures_dir / "videos").glob("*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def seed_fixture_video(session: Session, fx: dict[str, Any], media_dir: Path) -> Video:
    if not fx.get("synthetic", False):
        raise ValueError("fixture files must be marked synthetic")
    ch = fx["channel"]
    upsert_channel(session, ch["channel_id"], ch["title"], ch.get("subscriber_count"), is_synthetic=True)
    video = session.get(Video, fx["video_id"])
    if video is None:
        video = Video(
            id=fx["video_id"], channel_id=ch["channel_id"], title=fx["title"], discovery_source="fixture"
        )
        session.add(video)
    video.title = fx["title"]
    video.description = fx.get("description", "")
    video.published_at = datetime.fromisoformat(fx["published_at"].replace("Z", "+00:00"))
    video.duration_seconds = fx.get("duration_seconds")
    video.language = fx.get("language")
    video.youtube_category_id = fx.get("youtube_category_id")
    video.tags = fx.get("tags", [])
    video.is_synthetic = True
    video.metadata_fetched_at = datetime.now(UTC)
    if video.status == "discovered":
        video.status = "metadata_ready"
    pop = fx.get("popularity", {})
    session.add(
        VideoStats(
            video_id=video.id,
            view_count=pop.get("view_count"),
            like_count=pop.get("like_count"),
            comment_count=pop.get("comment_count"),
        )
    )
    session.flush()
    content = fx["content"]
    transcript = Transcript.model_validate(content["transcript"])
    media_uri = None
    extra: list[Modality] = []
    audio = content.get("audio_fixture")
    if audio:
        path = media_dir / f"{video.id}.wav"
        if not path.exists():
            write_wav(path, generate(audio["profile"], seed=sum(map(ord, video.id))))
        media_uri = str(path)
        extra = [Modality.AUDIO]
    add_transcript_source(
        session,
        video_id=video.id,
        transcript=transcript,
        source_type=SourceType.SYNTHETIC_FIXTURE,
        authorization_basis=content["authorization_basis"],
        license=content.get("license"),
        media_uri=media_uri,
        extra_modalities=extra,
    )
    return video


# --------------------------------------------------------------------------- licensed datasets


def import_licensed_dataset(session: Session, path: Path) -> tuple[dict[str, int], list[str]]:
    """Import transcripts from a JSONL file whose license permits evaluation.

    Each line: ``{"video_id", "license", "authorization_basis", "language"?, "title"?,
    "channel_id"?, "channel_title"?, and either "segments": [...] or "vtt": "..."}``.
    Lines without a license or authorization basis are rejected, not imported.
    """
    stats = {"imported": 0, "duplicates": 0, "rejected": 0}
    new_ids: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("license") or not row.get("authorization_basis"):
            log.warning("rejected dataset row without license", extra={"line": lineno})
            stats["rejected"] += 1
            continue
        if "segments" in row:
            transcript = Transcript.model_validate(
                {"language": row.get("language"), "segments": row["segments"]}
            )
        elif "vtt" in row:
            transcript = parse_timed_text(row["vtt"], row.get("language"))
        else:
            stats["rejected"] += 1
            continue
        if session.get(Video, row["video_id"]) is None:
            register_discovered(session, row["video_id"], row.get("channel_id"), "licensed_dataset")
            v = session.get(Video, row["video_id"])
            assert v is not None
            v.title = row.get("title", "")
            v.language = row.get("language")
        _, created = add_transcript_source(
            session,
            video_id=row["video_id"],
            transcript=transcript,
            source_type=SourceType.LICENSED_DATASET,
            authorization_basis=row["authorization_basis"],
            license=row["license"],
        )
        stats["imported" if created else "duplicates"] += 1
        if created:
            new_ids.append(row["video_id"])
    return stats, new_ids
