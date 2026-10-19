from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.api.deps import db, embedder, optional_reader
from avd.config import LLMProviderName, Settings, get_settings
from avd.content.store import active_sources
from avd.datasets.blinding import WITHHELD_MESSAGE, blinded_video_ids, is_blinded
from avd.datasets.rights import WITHHELD, video_pending_uses, withhold_quotes
from avd.db.models import Annotation, Evaluation, FactualClaim, Video
from avd.embeddings.base import Embedder
from avd.evaluation.profiles import PROFILES
from avd.evaluation.rubric import RUBRIC
from avd.evaluation.schema import FRAMEWORK_VERSION, Dimension
from avd.evaluation.service import (
    HEURISTIC_MODE_DISCLAIMER,
    LLM_DISCLAIMER,
    current_evaluation,
    serialize_evaluation,
    summary,
)
from avd.recsys.recommender import video_card
from avd.recsys.search import search as search_videos

router = APIRouter(prefix="/v1", dependencies=[Depends(optional_reader)])


def _refuse_if_blinded(session: Session, video_id: str) -> None:
    if is_blinded(session, video_id):
        raise HTTPException(403, WITHHELD_MESSAGE)


@router.get("/meta")
def meta(settings: Settings = Depends(get_settings), emb: Embedder = Depends(embedder)) -> dict[str, Any]:
    return {
        "framework_version": FRAMEWORK_VERSION,
        "demo_mode": settings.demo_mode,
        # The evaluator kind, not the content: real items can be evaluated in heuristic mode too.
        "evaluation_mode": "heuristic_baseline" if settings.demo_mode else "llm",
        "disclaimer": HEURISTIC_MODE_DISCLAIMER if settings.demo_mode else LLM_DISCLAIMER,
        "llm_provider": settings.llm_provider.value,
        # Where model inference runs: "local" (Ollama on the operator's hardware), "hosted", or none.
        "inference": None
        if settings.demo_mode
        else ("local" if settings.llm_provider == LLMProviderName.OLLAMA else "hosted"),
        "models": {}
        if settings.demo_mode
        else {"stage_b": settings.llm_model_stage_b, "stage_c": settings.llm_model_stage_c},
        "embedding_model": emb.model_id,
        "youtube_api_enabled": settings.youtube_api_key is not None,
        "fact_check_corpus": settings.reference_corpus_dir is not None,
        "dimensions": [
            {
                "dimension": d.value,
                "group": d.group.value,
                "question": RUBRIC[d].question,
                "anchors": {"1": RUBRIC[d].anchor_1, "3": RUBRIC[d].anchor_3, "5": RUBRIC[d].anchor_5},
                "requires": sorted(m.value for m in RUBRIC[d].requires),
            }
            for d in Dimension
        ],
        "genre_profiles": {g.value: p.as_dict() for g, p in PROFILES.items()},
    }


@router.get("/videos")
def list_videos(
    genre: str | None = None,
    evaluated: bool = True,
    limit: int = Query(50, le=200),
    offset: int = 0,
    session: Session = Depends(db),
) -> dict[str, Any]:
    q = select(Video).order_by(Video.discovered_at.desc()).limit(limit).offset(offset)
    if genre:
        q = q.where(Video.genre == genre)
    if evaluated:
        q = q.where(Video.status == "evaluated")
    items = []
    blinded = blinded_video_ids(session)
    for v in session.scalars(q).all():
        if v.id in blinded:
            continue
        items.append(
            {"video": video_card(session, v), "intrinsic_quality": summary(current_evaluation(session, v.id))}
        )
    return {"items": items}


@router.get("/videos/{video_id}")
def get_video(video_id: str, session: Session = Depends(db)) -> dict[str, Any]:
    _refuse_if_blinded(session, video_id)
    v = session.get(Video, video_id)
    if v is None:
        raise HTTPException(404, "video not in catalog")
    return {
        "video": video_card(session, v),
        "available_modalities": v.available_modalities,
        "content_sources": [
            {
                "id": s.id,
                "source_type": s.source_type,
                "authorization_basis": s.authorization_basis,
                "license": s.license,
                # Evidence quotes from licensed sources are shown to users; attribution is required.
                "attribution": s.attribution,
                "provenance_url": s.provenance_url,
                "modalities": s.modalities,
                "language": s.language,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in active_sources(session, v.id)
        ],
        "intrinsic_quality": summary(current_evaluation(session, v.id)),
    }


@router.get("/videos/{video_id}/evaluation")
def get_evaluation(video_id: str, session: Session = Depends(db)) -> dict[str, Any]:
    _refuse_if_blinded(session, video_id)
    ev = current_evaluation(session, video_id)
    if ev is None:
        v = session.get(Video, video_id)
        raise HTTPException(
            404,
            "video not in catalog" if v is None else f"no evaluation yet (status: {v.status})",
        )
    payload = serialize_evaluation(ev)
    # Independent human ratings of this video (released only once annotation is unblinded, which
    # _refuse_if_blinded enforces). They are a separate judgement, never merged into the AI scores.
    payload["human_reviews"] = int(
        session.scalar(
            select(func.count())
            .select_from(Annotation)
            .where(Annotation.video_id == video_id, Annotation.status == "submitted")
        )
        or 0
    )
    if "public_quotes" in video_pending_uses(session, video_id):
        payload = withhold_quotes(payload)
    return payload


@router.get("/videos/{video_id}/imported-evaluations")
def imported_evaluations(video_id: str, session: Session = Depends(db)) -> dict[str, Any]:
    """Evaluations imported from other installations' exports: unverified, never merged into
    this instance's scores, rankings or benchmarks."""
    from avd.portability import imported_for_video

    _refuse_if_blinded(session, video_id)
    return {
        "video_id": video_id,
        "status": "imported_unverified",
        "note": "From other installations; not produced or verified here.",
        "items": imported_for_video(session, video_id),
    }


@router.get("/videos/{video_id}/evaluations")
def evaluation_history(video_id: str, session: Session = Depends(db)) -> dict[str, Any]:
    _refuse_if_blinded(session, video_id)
    rows = session.scalars(
        select(Evaluation).where(Evaluation.video_id == video_id).order_by(Evaluation.created_at.desc())
    ).all()
    return {
        "items": [
            {
                **(summary(e) or {}),
                "framework_version": e.framework_version,
                "is_current": e.is_current,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "models": e.models,
            }
            for e in rows
        ]
    }


@router.get("/videos/{video_id}/claims")
def claims(video_id: str, session: Session = Depends(db)) -> dict[str, Any]:
    """Claims from the video's current evaluation(s). ``verdict`` is one of supported,
    contradicted, mixed, outdated (accurate when published, since changed), unverifiable,
    insufficient_evidence, not_applicable, or ``unchecked``
    (extracted but not verified against references). Verdicts come only from licensed reference
    passages, never from model knowledge."""
    _refuse_if_blinded(session, video_id)
    rows = session.execute(
        select(FactualClaim, Evaluation.stage)
        .join(Evaluation, Evaluation.id == FactualClaim.evaluation_id)
        .where(Evaluation.video_id == video_id, Evaluation.is_current.is_(True))
        .order_by(FactualClaim.segment_index)
    ).all()
    quotes_ok = "public_quotes" not in video_pending_uses(session, video_id)
    return {
        "video_id": video_id,
        "quotes_withheld": not quotes_ok,
        "items": [
            {
                "segment_index": c.segment_index,
                "quote": c.quote if quotes_ok or not c.quote else WITHHELD,
                "claim": c.claim,
                "claim_type": c.claim_type,
                "verdict": c.verdict,
                "domain_risk": c.domain_risk,
                "sources": c.sources,
                "explanation": c.explanation,
                "checked_at": c.checked_at.isoformat() if c.checked_at else None,
                "stage": stage,
            }
            for c, stage in rows
        ],
    }


@router.get("/search")
def search(
    q: str = Query(min_length=2, max_length=300),
    genre: list[str] | None = Query(None),
    language: list[str] | None = Query(None),
    min_quality: float | None = Query(None, ge=1, le=5),
    include_synthetic: bool = True,
    limit: int = Query(20, le=100),
    session: Session = Depends(db),
    emb: Embedder = Depends(embedder),
) -> dict[str, Any]:
    return {
        "query": q,
        "items": search_videos(
            session,
            emb,
            q,
            limit=limit,
            genres=genre,
            languages=language,
            min_quality=min_quality,
            include_synthetic=include_synthetic,
        ),
    }
