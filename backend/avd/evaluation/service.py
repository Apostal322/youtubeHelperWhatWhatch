"""Evaluation persistence and serialization."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from avd.db.models import (
    DimensionScoreRow,
    Evaluation,
    EvidenceRow,
    FactualClaim,
    Video,
)
from avd.evaluation.profiles import get_profile
from avd.evaluation.schema import (
    FRAMEWORK_VERSION,
    EvaluationMode,
    EvaluationResult,
    Genre,
)
from avd.llm.pricing import LOCAL_PREFIX
from avd.observability import correlation_id

SYNTHETIC_DISCLAIMER = (
    "Synthetic demo evaluation produced by a deterministic heuristic baseline over synthetic "
    "fixture content. It is NOT an AI assessment and not a real video."
)
HEURISTIC_DISCLAIMER = (
    "Heuristic baseline (keyword and structure features) applied to real content. It is NOT an AI "
    "assessment and has not been validated against human judgement."
)
HEURISTIC_MODE_DISCLAIMER = (
    "No AI evaluator is configured: scores come from a keyword and structure heuristic baseline and "
    "have not been validated against human judgement. Synthetic demo videos are labelled individually."
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


def is_up_to_date(
    session: Session, video_id: str, fingerprint: str, stage: str, config_hash: str | None = None
) -> bool:
    """Incremental processing: skip when an evaluation with identical inputs, the same framework
    version, the same evaluator configuration (models, prompts, sampling, chunking) and an
    equal-or-higher stage already exists."""
    ev = current_evaluation(session, video_id)
    return (
        ev is not None
        and ev.input_fingerprint == fingerprint
        and (config_hash is None or ev.config_hash == config_hash)
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
    # "Synthetic" describes the *content*; the evaluator kind is recorded in ``mode``. A heuristic
    # run over real content is a heuristic baseline, not a synthetic demo.
    video_row = session.get(Video, result.video_id)
    content_synthetic = bool(video_row.is_synthetic) if video_row is not None else result.is_synthetic
    mode = result.mode
    if mode == EvaluationMode.SYNTHETIC_DEMO and not content_synthetic:
        mode = EvaluationMode.HEURISTIC_BASELINE
    ev = Evaluation(
        video_id=result.video_id,
        stage=result.stage,
        framework_version=result.framework_version,
        mode=mode.value,
        is_synthetic=content_synthetic,
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
            "profile": agg.profile_detail or get_profile(agg.profile).as_dict(),
            "contributions": [c.model_dump(mode="json") for c in agg.contributions],
            "abstain_reason": agg.abstain_reason,
            "notes": agg.notes,
            "genre_assessment": result.genre.model_dump(mode="json"),
        },
        validation=result.validation.model_dump(mode="json"),
        missing_inputs=result.missing_inputs,
        cost_usd=result.cost_usd,
        is_current=True,
        config_hash=result.config_hash,
        versions=result.versions,
        coverage_report=result.coverage_report,
        needs_review=result.needs_review,
        review_reasons=result.review_reasons,
        latency_ms=result.latency_ms,
        correlation_id=correlation_id.get(),
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
            reason_code=d.reason_code.value if d.reason_code else None,
            rubric_version=d.rubric_version,
            rubric_anchor=d.rubric_anchor,
            evaluated_at=d.evaluated_at,
            uncertainty=d.uncertainty,
        )
        row.evidence = [
            EvidenceRow(
                kind=e.kind,
                content_source_id=e.content_source_id,
                segment_index=e.segment_index,
                segment_id=e.segment_id,
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
    session.flush()
    _persist_claims(session, ev)
    # Model calls are recorded per attempt by LoggingProvider (cost_ledger); only the evaluation
    # total is stored here so failed and retried attempts are never double counted.
    video = session.get(Video, result.video_id)
    if video is not None:
        video.status = "evaluated" if result.status != "failed" else "evaluation_failed"
        video.genre = result.genre.genre.value
        video.genre_confidence = result.genre.confidence
        video.genre_method = result.genre.method
        video.genre_distribution = result.genre.distribution
        video.genre_status = result.genre.status.value
    session.flush()
    return ev


def serialize_evaluation(ev: Evaluation, include_evidence: bool = True) -> dict[str, Any]:
    synthetic = ev.is_synthetic or ev.mode == EvaluationMode.SYNTHETIC_DEMO.value
    state, state_reasons = assessment_state(ev)
    disclaimer = (
        SYNTHETIC_DISCLAIMER
        if synthetic
        else HEURISTIC_DISCLAIMER
        if ev.mode == EvaluationMode.HEURISTIC_BASELINE.value
        else LLM_DISCLAIMER
    )
    return {
        "evaluation_id": ev.id,
        "video_id": ev.video_id,
        "stage": ev.stage,
        "framework_version": ev.framework_version,
        "mode": ev.mode,
        "synthetic": synthetic,
        "disclaimer": disclaimer,
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
                "reason_code": d.reason_code,
                "rubric_version": d.rubric_version,
                "rubric_anchor": d.rubric_anchor,
                "evaluated_at": d.evaluated_at.isoformat() if d.evaluated_at else None,
                "content_fingerprint": ev.input_fingerprint,
                "uncertainty": d.uncertainty,
                "evidence": [
                    {
                        "kind": e.kind,
                        "segment_id": e.segment_id,
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
        "config_hash": ev.config_hash,
        "versions": ev.versions,
        "coverage_report": _public_coverage(ev.coverage_report),
        "needs_review": ev.needs_review,
        "review_reasons": ev.review_reasons,
        "assessment_state": state,
        "assessment_state_reasons": state_reasons,
        "evaluator_source": evaluator_source(ev),
    }


def _persist_claims(session: Session, ev: Evaluation) -> None:
    """Claims become queryable rows: verified ones from the fact checker (with verdict and source
    provenance) and, otherwise, the claims the Stage-B map step extracted (verdict ``unchecked``)."""
    evaluators = (ev.coverage_report or {}).get("evaluators") or {}
    checked = (evaluators.get("fact_check") or {}).get("claims") or []
    seen: set[tuple[int | None, str]] = set()
    for c in checked:
        seen.add((c.get("segment_index"), c.get("quote") or ""))
        session.add(
            FactualClaim(
                evaluation_id=ev.id,
                segment_index=c.get("segment_index"),
                quote=c.get("quote"),
                claim=c["claim"],
                claim_type=c["claim_type"],
                verdict=c.get("verdict", "unchecked"),
                domain_risk=c.get("domain_risk", "standard"),
                sources=c.get("sources", []),
                explanation=c.get("explanation"),
                checked_at=datetime.fromisoformat(c["checked_at"]) if c.get("checked_at") else None,
            )
        )
    for name, r in evaluators.items():
        for c in (r or {}).get("claims_extracted") or []:
            key = (c.get("segment_index"), c.get("quote") or "")
            if name == "fact_check" or key in seen:
                continue
            seen.add(key)
            session.add(
                FactualClaim(
                    evaluation_id=ev.id,
                    segment_index=c.get("segment_index"),
                    quote=c.get("quote"),
                    claim=c.get("claim", ""),
                    claim_type=c.get("claim_type", "factual_assertion"),
                    verdict="unchecked",
                )
            )


def _public_coverage(report: dict[str, Any]) -> dict[str, Any]:
    out = dict(report or {})
    evaluators = {}
    for name, r in (out.get("evaluators") or {}).items():
        evaluators[name] = {k: v for k, v in r.items() if k not in ("claims_extracted", "claims")}
    out["evaluators"] = evaluators
    return out


def evaluator_source(ev: Evaluation) -> str:
    """Who produced the scores: ``local_ai`` (an open-weight model on this deployment's own
    hardware, via Ollama), ``hosted_ai`` (a model API), ``heuristic`` or ``synthetic``. Orthogonal
    to :func:`assessment_state`, which says how much the scores can be trusted."""
    if ev.is_synthetic or ev.mode == EvaluationMode.SYNTHETIC_DEMO.value:
        return "synthetic"
    if ev.mode == EvaluationMode.HEURISTIC_BASELINE.value:
        return "heuristic"
    if any(str(m).startswith(LOCAL_PREFIX) for m in ev.models or []):
        return "local_ai"
    return "hosted_ai"


def assessment_state(ev: Evaluation) -> tuple[str, list[str]]:
    """What kind of assessment this is, in user-facing terms.

    * ``synthetic_demo`` — heuristic baseline over synthetic fixtures; never a real rating.
    * ``failed`` — evaluators failed; no scores should be shown.
    * ``insufficient_evidence`` — nothing (or too little) could be judged; no overall score.
    * ``preliminary`` — scores exist but with caveats (review flags, unverified or missing evidence,
      partial coverage, uncalibrated confidence below 0.5).
    * ``grounded`` — model evaluation whose scored judgements cite verified transcript evidence,
      passed validation and has no review flags. Still not human-validated.
    """
    reasons: list[str] = []
    if ev.is_synthetic or ev.mode == EvaluationMode.SYNTHETIC_DEMO.value:
        return "synthetic_demo", ["heuristic baseline over synthetic content"]
    if ev.mode == EvaluationMode.HEURISTIC_BASELINE.value:
        return "heuristic_baseline", [
            "keyword/structure heuristic on real content; not an AI or human assessment"
        ]
    if ev.status == "failed":
        return "failed", ["evaluation failed"]
    scored = [d for d in ev.dimensions if d.status == "scored"]
    if not scored or ev.status == "abstained":
        return "insufficient_evidence", ["no dimension could be judged"]
    if ev.overall_score is None:
        reasons.append(ev.aggregation.get("abstain_reason") or "no overall score")
    if ev.needs_review:
        reasons.extend(ev.review_reasons)
    if not (ev.validation or {}).get("passed", True):
        reasons.append("evidence validation failed")
    ungrounded = (ev.validation or {}).get("ungrounded_dimensions") or []
    if ungrounded:
        reasons.append(f"scored without evidence: {', '.join(ungrounded)}")
    flagged = ((ev.coverage_report or {}).get("support_audit") or {}).get("flags") or {}
    if flagged:
        reasons.append(f"evidence support flags on: {', '.join(sorted(flagged))}")
    if ev.overall_confidence is not None and ev.overall_confidence < 0.5:
        reasons.append("low (uncalibrated) confidence")
    if reasons:
        return (
            "insufficient_evidence" if ev.overall_score is None and len(scored) < 3 else "preliminary"
        ), reasons
    return "grounded", [
        "scored judgements cite verified transcript evidence and pass the support audit; whether each "
        "quote semantically supports its judgement is not automated; not human-validated"
    ]


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
        "assessment_state": assessment_state(ev)[0],
        "evaluator_source": evaluator_source(ev),
        "needs_review": ev.needs_review,
    }


def genre_or_other(value: str | None) -> Genre:
    try:
        return Genre(value) if value else Genre.OTHER
    except ValueError:
        return Genre.OTHER
