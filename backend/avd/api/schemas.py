"""Request bodies (responses are plain dicts documented in docs/api.md)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from avd.evaluation.schema import Dimension, Genre


class PreferencesIn(BaseModel):
    interests: list[str] = Field(default_factory=list, max_length=30)
    genre_weights: dict[str, float] = Field(default_factory=dict)
    dimension_weights: dict[str, float] = Field(default_factory=dict)
    depth_preference: Literal["any", "intro", "intermediate", "advanced"] = "any"
    intent: Literal["any", "learn", "stay_informed", "be_entertained", "deep_dive"] = "any"
    languages: list[str] = Field(default_factory=list, max_length=20)
    min_duration_minutes: int | None = Field(default=None, ge=1, le=600)
    max_duration_minutes: int | None = Field(default=None, ge=1, le=600)
    diversity: float = Field(default=0.3, ge=0.0, le=1.0)
    min_quality: float = Field(default=0.0, ge=0.0, le=5.0)
    small_creator_boost: float = Field(default=0.0, ge=0.0, le=0.3)
    include_synthetic: bool = True

    @field_validator("interests")
    @classmethod
    def _clean_interests(cls, v: list[str]) -> list[str]:
        out = [i.strip()[:200] for i in v if i.strip()]
        return list(dict.fromkeys(out))

    @field_validator("genre_weights")
    @classmethod
    def _genres(cls, v: dict[str, float]) -> dict[str, float]:
        for k, w in v.items():
            Genre(k)
            if not 0.0 <= w <= 1.0:
                raise ValueError("genre weights must be within [0, 1]")
        return v

    @field_validator("dimension_weights")
    @classmethod
    def _dims(cls, v: dict[str, float]) -> dict[str, float]:
        for k, w in v.items():
            Dimension(k)
            if not 0.0 <= w <= 3.0:
                raise ValueError("dimension priorities must be within [0, 3]")
        return v

    @model_validator(mode="after")
    def _duration_range(self) -> PreferencesIn:
        lo, hi = self.min_duration_minutes, self.max_duration_minutes
        if lo and hi and lo > hi:
            raise ValueError("min_duration_minutes must not exceed max_duration_minutes")
        return self


FeedbackKind = Literal[
    "recommendation_useful", "recommendation_not_useful", "not_interested", "evaluation_incorrect"
]


class FeedbackIn(BaseModel):
    video_id: str = Field(max_length=32)
    kind: FeedbackKind
    impression_id: str | None = None
    evaluation_id: str | None = None
    dimension: Dimension | None = None
    suggested_score: float | None = Field(default=None, ge=1.0, le=5.0)
    comment: str | None = Field(default=None, max_length=2000)


class TranscriptIn(BaseModel):
    format: Literal["vtt", "srt", "text", "json"]
    content: str = Field(min_length=1, max_length=2_000_000)
    language: str | None = Field(default=None, max_length=16)
    attestation: str = Field(
        min_length=10,
        max_length=2000,
        description="Statement of your right to submit this transcript (e.g. you are the creator, "
        "or the transcript is published under a license permitting this use).",
    )


class DiscoverIn(BaseModel):
    n_queries: int = Field(default=4, ge=1, le=50)
