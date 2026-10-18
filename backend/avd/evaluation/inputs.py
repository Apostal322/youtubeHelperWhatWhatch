"""Popularity-blind evaluator inputs.

``EvaluationInput`` is the *only* object evaluators receive. It is built from a whitelist of
descriptive video fields plus authorized content. It deliberately has no fields for views, likes,
comments, subscriber counts, or channel identity/name ("blind review"), so popularity cannot
influence intrinsic quality scores — not even through the prompt. ``fingerprint`` hashes everything
an evaluator can see; it is used for caching, incremental processing and invariance tests.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict

from avd.content.models import ContentBundle, Transcript
from avd.db.models import Video
from avd.evaluation.sanitize import sanitize_description, sanitize_tags, sanitize_title
from avd.evaluation.schema import FRAMEWORK_VERSION, GenreAssessment, Modality

# Fields of ``Video`` that evaluators may see. Anything else is excluded by construction.
ALLOWED_VIDEO_FIELDS = ("id", "title", "description", "duration_seconds", "language", "tags")

FORBIDDEN_INPUT_TERMS = (
    "view",
    "like",
    "comment_count",
    "subscriber",
    "channel",
    "popularity",
)


class EvaluationInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    title: str
    description: str
    duration_seconds: int | None
    language: str | None
    tags: tuple[str, ...]
    genre: GenreAssessment
    transcript: Transcript | None
    transcript_source_id: str | None
    audio_path: str | None
    audio_sha256: str | None
    audio_source_id: str | None
    video_path: str | None = None
    video_sha256: str | None = None
    video_source_id: str | None = None
    modalities: frozenset[Modality]
    content_source_ids: tuple[str, ...]
    # Number of popularity statements removed from title/description/tags (audit only; excluded
    # from the fingerprint so the evaluation is invariant to them).
    sanitized_spans: int = 0
    # False when the title is a placeholder (e.g. "Show #12" derived from a file path).
    title_is_original: bool = True

    def fingerprint(self) -> str:
        payload = {
            "framework_version": FRAMEWORK_VERSION,
            "video_id": self.video_id,
            "title": self.title,
            "description": self.description,
            "duration_seconds": self.duration_seconds,
            "language": self.language,
            "tags": list(self.tags),
            "genre": self.genre.genre.value,
            "transcript_hash": self.transcript.content_hash() if self.transcript else None,
            "audio_sha256": self.audio_sha256,
            "modalities": sorted(m.value for m in self.modalities),
        }
        if self.video_sha256:  # only when present, so transcript/audio-only fingerprints stay stable
            payload["video_sha256"] = self.video_sha256
        if not self.title_is_original:  # only when set, so existing fingerprints stay stable
            payload["title_is_original"] = False
        canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()


def build_evaluation_input(video: Video, bundle: ContentBundle, genre: GenreAssessment) -> EvaluationInput:
    # Popularity statements in free text are removed *before* evaluators see anything.
    title, n_title = sanitize_title(video.title)
    description, n_desc = sanitize_description(video.description or "")
    tags, n_tags = sanitize_tags(tuple(video.tags or []))
    return EvaluationInput(
        video_id=video.id,
        title=title,
        description=description,
        duration_seconds=video.duration_seconds,
        language=video.language or (bundle.transcript.language if bundle.transcript else None),
        tags=tags,
        sanitized_spans=n_title + n_desc + n_tags,
        genre=genre,
        transcript=bundle.transcript,
        transcript_source_id=bundle.transcript_source_id,
        audio_path=bundle.audio_path,
        audio_sha256=bundle.audio_sha256,
        audio_source_id=bundle.audio_source_id,
        video_path=bundle.video_path,
        video_sha256=bundle.video_sha256,
        video_source_id=bundle.video_source_id,
        modalities=frozenset(bundle.modalities),
        content_source_ids=tuple(s.id for s in bundle.sources),
        title_is_original=video.title_source in (None, "platform", "creator_file_slug"),
    )
