"""Evaluation persistence and serialization."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from avd.db.models import (
    CostLedger,
    DimensionScoreRow,
    Evaluation,
    EvidenceRow,
    Video,
)
from avd.evaluation.profiles import get_profile
from avd.evaluation.schema import (
    FRAMEWORK_VERSION,
    EvaluationMode,
    EvaluationResult,
    Genre,
)

SYNTHETIC_DISCLAIMER = (
    "Synthetic demo evaluation produced by a deterministic heuristic baseline over synthetic "
    "fixture content. It is NOT an AI assessment and not a real video."
)
LLM_DISCLAIMER = (
    "Automated LLM-based assessment. Scores are model judgements against a published rubric, not "
    "ground truth; their agreement with human raters is measured by the benchmark harness and "
    "confidence values are not yet calibrated."
)

STAGE_RANK = {"B": 1, "C": 2}


def current_evaluation(session: Session, video_id: str) -> Evaluation | None:
    return session.scalar(
        select(Evaluation)
        .where(Evaluation.video_id == video_id, Evaluation.is_current.is_(True))
        .options(selectinload(Evaluation.dimensions).selectinload(DimensionScoreRow.evidence))
        .order_by(Evaluation.created_at.desc())
    )


def is_up_to_date(session: Session, video_id: str, fingerprint: str, stage: str) -> bool:
    """Incremental processing: skip when an evaluation with identical inputs, the same framework
    version and an equal-or-higher stage already exists."""
    ev = current_evaluation(session, video_id)
    return (
        ev is not None
        and ev.input_fingerprint == fingerprint
        and ev.framework_version == FRAMEWORK_VERSION
        and STAGE_RANK.get(ev.stage, 0) >= STAGE_RANK.get(stage, 0)
        and ev.status != "failed"
    )


def persist_evaluation(session: Session, result: EvaluationResult, job_id: str | None = None) -> Evaluation:
    session.execute(
        update(Evaluation)
        .where(Evaluation.video_id == result.video_id, Evaluation.is_current.is_(True))
        .values(is_current=False)
    )
    agg = result.aggregation
    ev = Evaluation(
        video_id=result.video_id,
        stage=result.stage,
        framework_version=result.framework_version,
        mode=result.mode.value,
        is_synthetic=result.is_synthetic,
        models=result.models,
        genre=result.genre.genre.value,
        genre_confidence=result.genre.confidence,
        input_fingerprint=result.input_fingerprint,
        content_source_ids=result.content_source_ids,
        status=result.status,
        overall_score=agg.overall,
        overall_confidence=agg.overall_confidence,
        coverage=agg.coverage,
        group_scores={g.value: a.model_dump(mode="json") for g, a in agg.groups.items()},
        aggregation={
            "method": agg.method,
            "profile": get_profile(agg.profile).as_dict(),
            "contributions": [c.model_dump(mode="json") for c in agg.contributions],
            "abstain_reason": agg.abstain_reason,
            "notes": agg.notes,
            "genre_assessment": result.genre.model_dump(mode="json"),
        },
        validation=result.validation.model_dump(mode="json"),
        missing_inputs=result.missing_inputs,
        cost_usd=result.cost_usd,
        is_current=True,
    )
    for d in result.dimensions:
        row = DimensionScoreRow(
            dimension=d.dimension.value,
            group=d.group.value,
            status=d.status.value,
            score=d.score,
            confidence=d.confidence,
            justification=d.justification,
            method=d.method,
            model=d.model,
            missing_inputs=d.missing_inputs,
        )
        row.evidence = [
            EvidenceRow(
                content_source_id=e.content_source_id,
                segment_index=e.segment_index,
                start_seconds=e.start_seconds,
                end_seconds=e.end_seconds,
                quote=e.quote,
                note=e.note,
                verified=e.verified,
            )
            for e in d.evidence
        ]
        ev.dimensions.append(row)
    session.add(ev)
    for u in result.usage:
        session.add(
            CostLedger(
                job_id=job_id,
                video_id=result.video_id,
                provider=u.provider,
                model=u.model,
                operation=u.operation,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cost_usd=u.cost_usd,
                cached=u.cached,
            )
        )
    video = session.get(Video, result.video_id)
    if video is not None:
        video.status = "evaluated" if result.status != "failed" else "evaluation_failed"
        video.genre = result.genre.genre.value
        video.genre_confidence = result.genre.confidence
        video.genre_method = result.genre.method
    session.flush()
    return ev


def serialize_evaluation(ev: Evaluation, include_evidence: bool = True) -> dict[str, Any]:
    synthetic = ev.is_synthetic or ev.mode == EvaluationMode.SYNTHETIC_DEMO.value
    return {
        "evaluation_id": ev.id,
        "video_id": ev.video_id,
        "stage": ev.stage,
        "framework_version": ev.framework_version,
        "mode": ev.mode,
        "synthetic": synthetic,
        "disclaimer": SYNTHETIC_DISCLAIMER if synthetic else LLM_DISCLAIMER,
        "models": ev.models,
        "genre": ev.genre,
        "genre_confidence": ev.genre_confidence,
        "status": ev.status,
        "overall": {
            "score": ev.overall_score,
            "confidence": ev.overall_confidence,
            "coverage": ev.coverage,
            "abstain_reason": ev.aggregation.get("abstain_reason"),
            "method": ev.aggregation.get("method"),
            "notes": ev.aggregation.get("notes", []),
        },
        "groups": ev.group_scores,
        "aggregation_profile": ev.aggregation.get("profile"),
        "contributions": ev.aggregation.get("contributions", []),
        "dimensions": [
            {
                "dimension": d.dimension,
                "group": d.group,
                "status": d.status,
                "score": d.score,
                "confidence": d.confidence,
                "justification": d.justification,
                "method": d.method,
                "model": d.model,
                "missing_inputs": d.missing_inputs,
                "evidence": [
                    {
                        "segment_index": e.segment_index,
                        "start_seconds": e.start_seconds,
                        "end_seconds": e.end_seconds,
                        "quote": e.quote,
                        "note": e.note,
                        "verified": e.verified,
                        "content_source_id": e.content_source_id,
                    }
                    for e in d.evidence
                ]
                if include_evidence
                else [],
            }
            for d in ev.dimensions
        ],
        "validation": ev.validation,
        "missing_inputs": ev.missing_inputs,
        "cost_usd": ev.cost_usd,
        "created_at": ev.created_at.isoformat() if ev.created_at else None,
    }


def summary(ev: Evaluation | None) -> dict[str, Any] | None:
    if ev is None:
        return None
    return {
        "evaluation_id": ev.id,
        "overall": ev.overall_score,
        "overall_confidence": ev.overall_confidence,
        "coverage": ev.coverage,
        "groups": {g: v.get("score") for g, v in ev.group_scores.items()},
        "status": ev.status,
        "synthetic": ev.is_synthetic,
        "mode": ev.mode,
        "genre": ev.genre,
        "stage": ev.stage,
    }


def genre_or_other(value: str | None) -> Genre:
    try:
        return Genre(value) if value else Genre.OTHER
    except ValueError:
        return Genre.OTHER
