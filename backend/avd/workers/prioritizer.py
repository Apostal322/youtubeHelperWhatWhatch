"""Popularity-blind, coverage-aware prioritisation for deep analysis.

priority = stage base
         + content availability bonus
         + coverage-gap bonuses for the video's genre, channel-size stratum and language

Coverage gaps use ``1 / (1 + n_evaluated_in_stratum)``, so under-represented strata (e.g. a
language or a micro-channel bucket with few evaluations) rise to the top. Channel size is used
*only* as a fairness stratum; views, likes and subscriber counts never increase priority. A video
from a huge channel gets exactly the same treatment as one from a micro channel when both strata
are equally covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.db.models import Channel, Evaluation, Video

STAGE_BASE = {"stage_a": 10.0, "stage_b": 5.0, "stage_c": 1.0}
W_GENRE, W_SIZE, W_LANG = 3.0, 3.0, 2.0


@dataclass
class CoverageCounts:
    by_genre: dict[str, int] = field(default_factory=dict)
    by_size: dict[str, int] = field(default_factory=dict)
    by_language: dict[str, int] = field(default_factory=dict)


def coverage_counts(session: Session) -> CoverageCounts:
    base = (
        select(Video.genre, Channel.size_bucket, Video.language, func.count())
        .join(Evaluation, (Evaluation.video_id == Video.id) & Evaluation.is_current.is_(True))
        .join(Channel, Channel.id == Video.channel_id)
        .group_by(Video.genre, Channel.size_bucket, Video.language)
    )
    c = CoverageCounts()
    for genre, size, lang, n in session.execute(base).all():
        g, s, lg = genre or "other", size or "unknown", (lang or "unknown").split("-")[0]
        c.by_genre[g] = c.by_genre.get(g, 0) + n
        c.by_size[s] = c.by_size.get(s, 0) + n
        c.by_language[lg] = c.by_language.get(lg, 0) + n
    return c


def priority_for(
    job_type: str,
    *,
    genre: str | None,
    size_bucket: str | None,
    language: str | None,
    has_transcript: bool,
    has_audio: bool,
    counts: CoverageCounts,
) -> tuple[float, dict[str, float]]:
    parts = {
        "stage": STAGE_BASE.get(job_type, 0.0),
        "content": (3.0 if has_transcript else 0.0) + (0.5 if has_audio else 0.0),
        "genre_gap": W_GENRE / (1 + counts.by_genre.get(genre or "other", 0)),
        "size_gap": W_SIZE / (1 + counts.by_size.get(size_bucket or "unknown", 0)),
        "language_gap": W_LANG / (1 + counts.by_language.get((language or "unknown").split("-")[0], 0)),
    }
    return round(sum(parts.values()), 4), parts


def priority_for_video(
    session: Session, video: Video, job_type: str, counts: CoverageCounts | None = None
) -> tuple[float, dict[str, float]]:
    counts = counts or coverage_counts(session)
    channel = session.get(Channel, video.channel_id)
    mods = set(video.available_modalities or [])
    return priority_for(
        job_type,
        genre=video.genre,
        size_bucket=channel.size_bucket if channel else None,
        language=video.language,
        has_transcript="transcript" in mods,
        has_audio="audio" in mods,
        counts=counts,
    )
