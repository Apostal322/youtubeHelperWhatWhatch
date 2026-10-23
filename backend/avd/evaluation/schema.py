"""The evaluation contract.

Every dimension result carries an explicit ``status``. Only ``scored`` results carry a score; every
other status is an *abstention* with a reason. This makes "we don't know" a first-class output
instead of a misleading number.

Scores use an anchored 1-5 scale (see ``rubric.py``) for dimensions, groups and the optional
genre-weighted overall summary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Component versions. Each is stored with every evaluation; FRAMEWORK_VERSION changes whenever any
# of them changes, so results from different framework versions are never silently mixed.
RUBRIC_VERSION = "rubric_v2"  # criterion-specific anchors for all five levels
PROMPT_VERSION = "rubric_prompt_v2"  # nonce-delimited transcript, counter-evidence
AGGREGATION_VERSION = "genre_weighted_mean_renormalized_v2"  # genre mixtures, ambiguity handling
PREPROCESSING_VERSION = (
    "prep_v2"  # normalisation (keeps shared source spans), language id, chunking, sanitisation
)
INJECTION_DETECTOR_VERSION = "injection_v2"  # v2: no flags on talk *about* AI; see docs/phase2.md
# v2: whole-word quotes of >=3 words, negation-aware near matches, duplicates counted once,
# quotes drawing on injection-flagged neighbours rejected (audit 2026-10-23)
EVIDENCE_VALIDATION_VERSION = "evidence_v2"
FRAMEWORK_VERSION = "2026.10.2"

SCORE_MIN = 1.0
SCORE_MAX = 5.0


class DimensionGroup(StrEnum):
    CONTENT = "content"
    PRESENTATION = "presentation"
    TECHNICAL = "technical"


class Dimension(StrEnum):
    # content
    INFORMATION_DENSITY = "information_density"
    DEPTH = "depth"
    FACTUAL_ACCURACY = "factual_accuracy"
    ORIGINALITY = "originality"
    INFORMATIONAL_VALUE = "informational_value"
    # presentation
    CLARITY = "clarity"
    STRUCTURE = "structure"
    PACING = "pacing"
    STORYTELLING = "storytelling"
    TITLE_CONSISTENCY = "title_consistency"
    # technical
    AUDIO_QUALITY = "audio_quality"
    VISUAL_QUALITY = "visual_quality"
    EDITING = "editing"
    VISUAL_READABILITY = "visual_readability"

    @property
    def group(self) -> DimensionGroup:
        return DIMENSION_GROUPS[self]


DIMENSION_GROUPS: dict[Dimension, DimensionGroup] = {
    Dimension.INFORMATION_DENSITY: DimensionGroup.CONTENT,
    Dimension.DEPTH: DimensionGroup.CONTENT,
    Dimension.FACTUAL_ACCURACY: DimensionGroup.CONTENT,
    Dimension.ORIGINALITY: DimensionGroup.CONTENT,
    Dimension.INFORMATIONAL_VALUE: DimensionGroup.CONTENT,
    Dimension.CLARITY: DimensionGroup.PRESENTATION,
    Dimension.STRUCTURE: DimensionGroup.PRESENTATION,
    Dimension.PACING: DimensionGroup.PRESENTATION,
    Dimension.STORYTELLING: DimensionGroup.PRESENTATION,
    Dimension.TITLE_CONSISTENCY: DimensionGroup.PRESENTATION,
    Dimension.AUDIO_QUALITY: DimensionGroup.TECHNICAL,
    Dimension.VISUAL_QUALITY: DimensionGroup.TECHNICAL,
    Dimension.EDITING: DimensionGroup.TECHNICAL,
    Dimension.VISUAL_READABILITY: DimensionGroup.TECHNICAL,
}


class Genre(StrEnum):
    EDUCATION = "education"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    GAMING = "gaming"
    ENTERTAINMENT = "entertainment"
    DOCUMENTARY = "documentary"
    ART = "art"
    MUSIC = "music"
    HOWTO = "howto"
    NEWS_COMMENTARY = "news_commentary"
    # Genre v2: more specific formats (each maps to a v1 parent in GENRE_PARENT).
    LECTURE = "lecture"
    PROGRAMMING = "programming"
    PRODUCT_REVIEW = "product_review"
    INTERVIEW_PODCAST = "interview_podcast"
    COMEDY = "comedy"
    PERSONAL_STORYTELLING = "personal_storytelling"
    OTHER = "other"


# v2 genres refine v1 genres; used to compare against v1-era labels.
GENRE_PARENT: dict[Genre, Genre] = {
    Genre.LECTURE: Genre.EDUCATION,
    Genre.PROGRAMMING: Genre.TECHNOLOGY,
    Genre.PRODUCT_REVIEW: Genre.TECHNOLOGY,
    Genre.COMEDY: Genre.ENTERTAINMENT,
    Genre.PERSONAL_STORYTELLING: Genre.ENTERTAINMENT,
    Genre.INTERVIEW_PODCAST: Genre.OTHER,
}


def expand_genres(genres: list[str]) -> list[str]:
    """Genres plus their v2 refinements (a "technology" filter also matches "programming")."""
    out = list(dict.fromkeys(genres))
    for child, parent in GENRE_PARENT.items():
        if parent.value in out and child.value not in out:
            out.append(child.value)
    return out


def genre_preference(weights: dict[str, float], genre: str | None, default: float = 0.5) -> float:
    """A user's weight for a genre, falling back to its v1 parent (preferences predate v2)."""
    g = genre or "other"
    if g in weights:
        return float(weights[g])
    try:
        parent = GENRE_PARENT.get(Genre(g))
    except ValueError:
        parent = None
    return float(weights.get(parent.value, default)) if parent is not None else default


class GenreStatus(StrEnum):
    CONFIDENT = "confident"  # one genre clearly dominates
    MIXED = "mixed"  # two genres both carry substantial weight -> blended profile
    AMBIGUOUS = "ambiguous"  # weak, spread-out evidence -> neutral (uniform) profile
    UNKNOWN = "unknown"  # no usable evidence -> neutral profile
    OVERRIDDEN = "overridden"  # set by a person


class Modality(StrEnum):
    TRANSCRIPT = "transcript"
    AUDIO = "audio"
    VIDEO = "video"


class DimensionStatus(StrEnum):
    SCORED = "scored"
    NOT_APPLICABLE = "not_applicable"  # genre profile says this dimension does not apply
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # inputs exist but don't support a judgement
    REQUIRES_MEDIA = "requires_media"  # a required authorized modality is unavailable
    FAILED = "failed"  # the evaluator errored or its output failed validation


class EvaluationMode(StrEnum):
    SYNTHETIC_DEMO = "synthetic_demo"  # deterministic heuristic baseline — NOT an AI assessment
    LLM = "llm"
    # The same heuristic baseline applied to real (non-synthetic) content. Recorded at persistence
    # time, because only then is the content's provenance known. Not an AI or human assessment.
    HEURISTIC_BASELINE = "heuristic_baseline"


class ReasonCode(StrEnum):
    """Machine-readable reason for an abstention or failure."""

    GENRE_NOT_APPLICABLE = "genre_not_applicable"
    MISSING_MODALITY = "missing_modality"
    MODEL_ABSTAINED = "model_abstained"
    METHOD_CANNOT_JUDGE = "method_cannot_judge"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    NO_EVALUATOR = "no_evaluator"
    EVALUATOR_ERROR = "evaluator_error"
    MODEL_REFUSED = "model_refused"
    INVALID_OUTPUT = "invalid_output"
    OMITTED_BY_MODEL = "omitted_by_model"
    VALIDATION_FAILED = "validation_failed"


class EvidenceRef(BaseModel):
    """A pointer into an authorized input. Timestamps are copied from our transcript data, never
    from model output. ``segment_id`` = ``<transcript content hash prefix>#<index>`` is stable for a
    given transcript version."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["supporting", "counter"] = "supporting"
    content_source_id: str | None = None
    segment_index: int | None = None
    segment_id: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    quote: str | None = Field(default=None, max_length=600)
    note: str | None = Field(default=None, max_length=600)
    verified: bool = False


class DimensionScore(BaseModel):
    dimension: Dimension
    status: DimensionStatus
    score: float | None = Field(default=None, ge=SCORE_MIN, le=SCORE_MAX)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)
    method: str
    model: str | None = None  # model that actually produced the judgement (served model id)
    missing_inputs: list[str] = Field(default_factory=list)
    reason_code: ReasonCode | None = None  # set for every non-scored status
    rubric_version: str = RUBRIC_VERSION
    rubric_anchor: str | None = None  # anchor text of the scored level (filled by the pipeline)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Raw uncertainty components (model-reported confidence, sample dispersion, evidence
    # verification, adjustments applied). ``confidence`` is derived from these and is uncalibrated
    # until fitted against human labels.
    uncertainty: dict[str, Any] = Field(default_factory=dict)

    @property
    def supporting(self) -> list[EvidenceRef]:
        return [e for e in self.evidence if e.kind == "supporting"]

    @property
    def counter(self) -> list[EvidenceRef]:
        return [e for e in self.evidence if e.kind == "counter"]

    @model_validator(mode="after")
    def _status_consistency(self) -> DimensionScore:
        if self.status == DimensionStatus.SCORED:
            if self.score is None or self.confidence is None:
                raise ValueError("scored dimensions require score and confidence")
            if not self.justification.strip():
                raise ValueError("scored dimensions require a justification")
        elif self.score is not None:
            raise ValueError(f"status {self.status} must not carry a score")
        return self

    @property
    def group(self) -> DimensionGroup:
        return self.dimension.group

    @classmethod
    def abstain(
        cls,
        dimension: Dimension,
        status: DimensionStatus,
        reason: str,
        method: str,
        *,
        model: str | None = None,
        missing_inputs: list[str] | None = None,
        reason_code: ReasonCode | None = None,
    ) -> DimensionScore:
        assert status != DimensionStatus.SCORED
        if reason_code is None:
            reason_code = {
                DimensionStatus.NOT_APPLICABLE: ReasonCode.GENRE_NOT_APPLICABLE,
                DimensionStatus.REQUIRES_MEDIA: ReasonCode.MISSING_MODALITY,
                DimensionStatus.INSUFFICIENT_EVIDENCE: ReasonCode.INSUFFICIENT_MATERIAL,
                DimensionStatus.FAILED: ReasonCode.EVALUATOR_ERROR,
            }[status]
        return cls(
            dimension=dimension,
            status=status,
            justification=reason,
            method=method,
            model=model,
            missing_inputs=missing_inputs or [],
            reason_code=reason_code,
        )


class GenreAssessment(BaseModel):
    genre: Genre
    confidence: float = Field(ge=0.0, le=1.0)
    method: str
    scores: dict[str, float] = Field(default_factory=dict)  # top-4 shares (v1 field)
    distribution: dict[str, float] = Field(default_factory=dict)  # all genres with evidence, sums to 1
    status: GenreStatus = GenreStatus.CONFIDENT
    secondary: Genre | None = None  # the second genre of a mixture


class GroupAggregate(BaseModel):
    group: DimensionGroup
    score: float | None
    confidence: float | None
    coverage: float = Field(ge=0.0, le=1.0)
    scored_dimensions: list[Dimension] = Field(default_factory=list)


class Contribution(BaseModel):
    dimension: Dimension
    effective_weight: float
    score: float
    contribution: float


class AggregationResult(BaseModel):
    profile: Genre
    method: str
    profile_detail: dict[str, Any] = Field(default_factory=dict)  # the exact (possibly blended) weights
    groups: dict[DimensionGroup, GroupAggregate]
    overall: float | None
    overall_confidence: float | None
    coverage: float
    contributions: list[Contribution] = Field(default_factory=list)
    abstain_reason: str | None = None
    notes: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    evidence_claimed: int = 0
    evidence_verified: int = 0
    dimensions_invalidated: list[Dimension] = Field(default_factory=list)
    injection_citations_rejected: int = 0
    popularity_citations_rejected: int = 0
    ungrounded_dimensions: list[str] = Field(default_factory=list)
    injection_findings: list[dict[str, Any]] = Field(default_factory=list)


class UsageRecord(BaseModel):
    provider: str
    model: str
    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class EvaluationResult(BaseModel):
    video_id: str
    stage: str
    framework_version: str = FRAMEWORK_VERSION
    mode: EvaluationMode
    is_synthetic: bool
    models: list[str]
    genre: GenreAssessment
    input_fingerprint: str
    content_source_ids: list[str]
    dimensions: list[DimensionScore]
    aggregation: AggregationResult
    validation: ValidationReport
    missing_inputs: list[str]
    usage: list[UsageRecord] = Field(default_factory=list)
    status: str  # complete | partial | abstained | failed
    # Hash of the evaluator configuration (evaluators, models, prompt/rubric/aggregation versions,
    # sampling and chunking settings). A change forces re-evaluation of otherwise identical inputs.
    config_hash: str = ""
    versions: dict[str, str] = Field(default_factory=dict)
    # Which part of the authorized material evaluators actually received (computed by our code).
    coverage_report: dict[str, Any] = Field(default_factory=dict)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    latency_ms: int | None = None

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage)

    def dimension(self, d: Dimension) -> DimensionScore:
        for ds in self.dimensions:
            if ds.dimension == d:
                return ds
        raise KeyError(d)

    def scored(self) -> dict[Dimension, float]:
        return {
            d.dimension: d.score
            for d in self.dimensions
            if d.status == DimensionStatus.SCORED and d.score is not None
        }

    def summary(self) -> dict[str, Any]:
        return {
            "overall": self.aggregation.overall,
            "groups": {g.value: a.score for g, a in self.aggregation.groups.items()},
            "status": self.status,
            "mode": self.mode.value,
        }
