"""ORM models.

Design notes
------------
* Popularity statistics (views, likes, subscribers) live in ``video_stats`` / ``channels`` and are
  never read by the evaluation pipeline. Evaluator inputs are built from ``Video`` descriptive
  fields and authorized ``ContentSource`` rows only (see ``avd.evaluation.inputs``).
* Evaluations are immutable and versioned. Re-evaluation creates a new row and flips
  ``is_current``; history is preserved for audits and benchmark comparisons.
* Intrinsic quality (``evaluations``), user-specific relevance and recommendation ranking
  (``recommendation_impressions``) are stored separately.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from avd.db.types import EmbeddingVector


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    # Used only for discovery coverage monitoring (size buckets) — never for quality.
    subscriber_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bucket: Mapped[str] = mapped_column(String(16), default="unknown")
    country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    stats_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    videos: Mapped[list[Video]] = relationship(back_populates="channel")


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    channel_id: Mapped[str] = mapped_column(ForeignKey("channels.id"), index=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(String(16))
    youtube_category_id: Mapped[str | None] = mapped_column(String(8))
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)

    discovery_source: Mapped[str] = mapped_column(String(64), default="unknown")
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    metadata_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Pipeline state: discovered -> metadata_ready -> content_unavailable | content_available
    #                 -> evaluated | evaluation_failed
    status: Mapped[str] = mapped_column(String(32), default="discovered", index=True)
    available_modalities: Mapped[list[Any]] = mapped_column(JSON, default=list)

    genre: Mapped[str | None] = mapped_column(String(32), index=True)
    genre_confidence: Mapped[float | None] = mapped_column(Float)
    genre_method: Mapped[str | None] = mapped_column(String(64))
    topics: Mapped[list[Any]] = mapped_column(JSON, default=list)

    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    embedding_input: Mapped[str | None] = mapped_column(String(32))  # "metadata" | "transcript"

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    channel: Mapped[Channel] = relationship(back_populates="videos")
    content_sources: Mapped[list[ContentSource]] = relationship(back_populates="video")
    evaluations: Mapped[list[Evaluation]] = relationship(back_populates="video")


class VideoStats(Base):
    """Popularity snapshots. Deliberately isolated from the evaluation pipeline."""

    __tablename__ = "video_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    view_count: Mapped[int | None] = mapped_column(Integer)
    like_count: Mapped[int | None] = mapped_column(Integer)
    comment_count: Mapped[int | None] = mapped_column(Integer)


class ContentSource(Base):
    """An authorized piece of content (transcript, audio, ...) for a video."""

    __tablename__ = "content_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(48))
    authorization_basis: Mapped[str] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(64))
    provided_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    modalities: Mapped[list[Any]] = mapped_column(JSON, default=list)
    language: Mapped[str | None] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    transcript: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    media_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | revoked
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    video: Mapped[Video] = relationship(back_populates="content_sources")

    __table_args__ = (UniqueConstraint("video_id", "source_type", "content_hash"),)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(8))  # "B" | "C"
    framework_version: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(32))  # synthetic_demo | llm
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    models: Mapped[list[Any]] = mapped_column(JSON, default=list)
    genre: Mapped[str] = mapped_column(String(32))
    genre_confidence: Mapped[float] = mapped_column(Float)
    input_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    content_source_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16))  # complete | partial | abstained | failed
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage: Mapped[float] = mapped_column(Float, default=0.0)
    group_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    aggregation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    validation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    missing_inputs: Mapped[list[Any]] = mapped_column(JSON, default=list)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Phase 2: reproducibility and review metadata.
    config_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    coverage_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    review_reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    video: Mapped[Video] = relationship(back_populates="evaluations")
    dimensions: Mapped[list[DimensionScoreRow]] = relationship(
        back_populates="evaluation", cascade="all, delete-orphan", order_by="DimensionScoreRow.id"
    )


class DimensionScoreRow(Base):
    __tablename__ = "dimension_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluations.id", ondelete="CASCADE"), index=True)
    dimension: Mapped[str] = mapped_column(String(48))
    group: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(32))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    justification: Mapped[str] = mapped_column(Text, default="")
    method: Mapped[str] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(128))
    missing_inputs: Mapped[list[Any]] = mapped_column(JSON, default=list)
    reason_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    rubric_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rubric_anchor: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    uncertainty: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    evaluation: Mapped[Evaluation] = relationship(back_populates="dimensions")
    evidence: Mapped[list[EvidenceRow]] = relationship(
        back_populates="dimension_score", cascade="all, delete-orphan", order_by="EvidenceRow.id"
    )


class EvidenceRow(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dimension_score_id: Mapped[int] = mapped_column(
        ForeignKey("dimension_scores.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(16), default="supporting")
    content_source_id: Mapped[str | None] = mapped_column(String(36))
    segment_index: Mapped[int | None] = mapped_column(Integer)
    segment_id: Mapped[str | None] = mapped_column(String(48), nullable=True)
    start_seconds: Mapped[float | None] = mapped_column(Float)
    end_seconds: Mapped[float | None] = mapped_column(Float)
    quote: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)

    dimension_score: Mapped[DimensionScoreRow] = relationship(back_populates="evidence")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    preferences: Mapped[UserPreferences | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    interests: Mapped[list[Any]] = mapped_column(JSON, default=list)
    genre_weights: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dimension_weights: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    depth_preference: Mapped[str] = mapped_column(String(16), default="any")
    languages: Mapped[list[Any]] = mapped_column(JSON, default=list)
    max_duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diversity: Mapped[float] = mapped_column(Float, default=0.3)
    min_quality: Mapped[float] = mapped_column(Float, default=0.0)
    small_creator_boost: Mapped[float] = mapped_column(Float, default=0.0)
    include_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    interest_embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="preferences")


class RecommendationImpression(Base):
    """What we showed, with the three separately-stored scores and ranking features."""

    __tablename__ = "recommendation_impressions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    evaluation_id: Mapped[str | None] = mapped_column(String(36))
    position: Mapped[int] = mapped_column(Integer)
    ranker_version: Mapped[str] = mapped_column(String(48))
    intrinsic_quality: Mapped[float | None] = mapped_column(Float)
    relevance: Mapped[float] = mapped_column(Float)
    rank_score: Mapped[float] = mapped_column(Float)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    impression_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    evaluation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # recommendation_useful | recommendation_not_useful | evaluation_incorrect | not_interested
    kind: Mapped[str] = mapped_column(String(40))
    dimension: Mapped[str | None] = mapped_column(String(48), nullable=True)
    suggested_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PipelineJob(Base):
    """Durable job state. The source of truth for scheduling, dedup, retries and budget."""

    __tablename__ = "pipeline_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_type: Mapped[str] = mapped_column(String(32))  # discover | ingest | stage_a | stage_b | ...
    video_id: Mapped[str | None] = mapped_column(String(32), index=True)
    dedup_key: Mapped[str] = mapped_column(String(200), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=4)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    actual_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    correlation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (Index("ix_jobs_claim", "status", "next_attempt_at", "priority"),)


class CostLedger(Base):
    """Every paid or quota-consuming external call (LLM tokens, YouTube API units)."""

    __tablename__ = "cost_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(String(36), index=True)
    video_id: Mapped[str | None] = mapped_column(String(32), index=True)
    provider: Mapped[str] = mapped_column(String(48), index=True)
    model: Mapped[str | None] = mapped_column(String(128))
    operation: Mapped[str] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    quota_units: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    # Phase 2: the ledger doubles as the model-run log (every attempt, including failures).
    model_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(48), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok | cached | error
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)


class LLMCache(Base):
    """Content-addressed cache of validated LLM responses (reproducibility + cost control)."""

    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(48))
    model: Mapped[str] = mapped_column(String(128))
    response: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DiscoveryQuery(Base):
    """Incremental discovery state for a seed query (genre/language stratum)."""

    __tablename__ = "discovery_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String(300))
    genre: Mapped[str] = mapped_column(String(32))
    language: Mapped[str] = mapped_column(String(16))
    order: Mapped[str] = mapped_column(String(16), default="date")
    next_page_token: Mapped[str | None] = mapped_column(String(128))
    published_before: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    total_results: Mapped[int] = mapped_column(Integer, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (UniqueConstraint("query", "language", "order"),)
