"""In-memory representations of authorized content."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from avd.evaluation.schema import Modality


class SourceType(StrEnum):
    # Captions downloaded via the YouTube Data API with the channel owner's OAuth grant.
    CREATOR_AUTHORIZED_CAPTIONS = "creator_authorized_captions"
    # Transcript uploaded by a user who attests they have the right to submit it.
    USER_PROVIDED_TRANSCRIPT = "user_provided_transcript"
    # Audio/video uploaded by a user who attests they have the right to submit it.
    USER_PROVIDED_MEDIA = "user_provided_media"
    # Imported from a dataset whose license permits this processing.
    LICENSED_DATASET = "licensed_dataset"
    # Original synthetic demo content authored for this project.
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _ordered(self) -> TranscriptSegment:
        if self.end < self.start:
            raise ValueError("segment end precedes start")
        return self


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)

    language: str | None = None
    segments: list[TranscriptSegment]

    @field_validator("segments")
    @classmethod
    def _non_empty(cls, v: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if not v:
            raise ValueError("transcript has no segments")
        return v

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def duration(self) -> float:
        return self.segments[-1].end - self.segments[0].start

    def content_hash(self) -> str:
        canon = json.dumps(
            [[round(s.start, 3), round(s.end, 3), s.text] for s in self.segments],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canon.encode()).hexdigest()

    def numbered(self, max_chars: int | None = None) -> str:
        """Segment-indexed rendering used in prompts so models can cite ``[index]``."""
        lines = [f"[{i}] ({s.start:.0f}s) {s.text.strip()}" for i, s in enumerate(self.segments)]
        out = "\n".join(lines)
        if max_chars is not None and len(out) > max_chars:
            raise ValueError(
                f"transcript is {len(out)} chars, above the {max_chars} limit; chunked evaluation "
                "is not implemented — refusing to silently truncate"
            )
        return out

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SourceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source_type: SourceType
    authorization_basis: str
    license: str | None = None


class ContentBundle(BaseModel):
    """Everything the evaluator may use for one video, with provenance."""

    video_id: str
    sources: list[SourceRef]
    transcript: Transcript | None = None
    transcript_source_id: str | None = None
    audio_path: str | None = None
    audio_sha256: str | None = None
    audio_source_id: str | None = None
    video_path: str | None = None
    video_sha256: str | None = None
    video_source_id: str | None = None

    @property
    def modalities(self) -> set[Modality]:
        m: set[Modality] = set()
        if self.transcript is not None:
            m.add(Modality.TRANSCRIPT)
        if self.audio_path is not None:
            m.add(Modality.AUDIO)
        if self.video_path is not None:
            m.add(Modality.VIDEO)
        return m


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
