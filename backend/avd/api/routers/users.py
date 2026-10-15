from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from avd.api.deps import current_user, db, embedder, hash_token
from avd.api.schemas import FeedbackIn, PreferencesIn, TranscriptIn
from avd.config import Settings, get_settings
from avd.db.models import Evaluation, Feedback, RecommendationImpression, User, UserPreferences, Video
from avd.embeddings.base import Embedder
from avd.recsys.recommender import recommend
from avd.services import TranscriptSubmissionError, submit_user_transcript

router = APIRouter(prefix="/v1")


@router.post("/users", status_code=201)
def create_user(session: Session = Depends(db)) -> dict[str, str]:
    """Anonymous account for the extension. The token is shown once; only its hash is stored."""
    token = secrets.token_urlsafe(32)
    user = User(token_hash=hash_token(token))
    user.preferences = UserPreferences()
    session.add(user)
    session.flush()
    return {"user_id": user.id, "token": token}


def _prefs_out(p: UserPreferences) -> dict[str, Any]:
    return {
        "interests": p.interests,
        "genre_weights": p.genre_weights,
        "dimension_weights": p.dimension_weights,
        "depth_preference": p.depth_preference,
        "languages": p.languages,
        "max_duration_minutes": p.max_duration_minutes,
        "diversity": p.diversity,
        "min_quality": p.min_quality,
        "small_creator_boost": p.small_creator_boost,
        "include_synthetic": p.include_synthetic,
    }


@router.get("/users/me/preferences")
def get_preferences(user: User = Depends(current_user), session: Session = Depends(db)) -> dict[str, Any]:
    prefs = session.get(UserPreferences, user.id) or UserPreferences(user_id=user.id)
    return _prefs_out(prefs)


@router.put("/users/me/preferences")
def put_preferences(
    body: PreferencesIn,
    user: User = Depends(current_user),
    session: Session = Depends(db),
    emb: Embedder = Depends(embedder),
) -> dict[str, Any]:
    prefs = session.get(UserPreferences, user.id)
    if prefs is None:
        prefs = UserPreferences(user_id=user.id)
        session.add(prefs)
    for k, v in body.model_dump().items():
        setattr(prefs, k, v)
    if body.interests:
        vecs = emb.embed(body.interests)
        centroid = [sum(c) / len(vecs) for c in zip(*vecs, strict=True)]
        prefs.interest_embedding = centroid
        prefs.embedding_model = emb.model_id
    else:
        prefs.interest_embedding = None
    session.flush()
    return _prefs_out(prefs)


@router.get("/recommendations")
def recommendations(
    limit: int = Query(20, ge=1, le=50),
    user: User = Depends(current_user),
    session: Session = Depends(db),
    emb: Embedder = Depends(embedder),
) -> dict[str, Any]:
    return recommend(session, user.id, emb, limit=limit)


@router.post("/feedback", status_code=201)
def feedback(
    body: FeedbackIn, user: User = Depends(current_user), session: Session = Depends(db)
) -> dict[str, Any]:
    if session.get(Video, body.video_id) is None:
        raise HTTPException(404, "video not in catalog")
    if body.impression_id is not None:
        imp = session.get(RecommendationImpression, body.impression_id)
        if imp is None or imp.user_id != user.id or imp.video_id != body.video_id:
            raise HTTPException(400, "impression does not belong to this user/video")
    if body.evaluation_id is not None:
        ev = session.get(Evaluation, body.evaluation_id)
        if ev is None or ev.video_id != body.video_id:
            raise HTTPException(400, "evaluation does not belong to this video")
    if body.kind == "evaluation_incorrect" and body.evaluation_id is None:
        raise HTTPException(400, "evaluation_id is required when reporting an incorrect evaluation")
    fb = Feedback(
        user_id=user.id,
        video_id=body.video_id,
        impression_id=body.impression_id,
        evaluation_id=body.evaluation_id,
        kind=body.kind,
        dimension=body.dimension.value if body.dimension else None,
        suggested_score=body.suggested_score,
        comment=body.comment,
    )
    session.add(fb)
    session.flush()
    return {"id": fb.id, "status": "recorded"}


@router.post("/videos/{video_id}/transcripts", status_code=202)
def submit_transcript(
    video_id: str,
    body: TranscriptIn,
    user: User = Depends(current_user),
    session: Session = Depends(db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if not 6 <= len(video_id) <= 32:
        raise HTTPException(400, "invalid video id")
    try:
        row, created = submit_user_transcript(
            session,
            user_id=user.id,
            video_id=video_id,
            content=body.content,
            fmt=body.format,
            language=body.language,
            attestation=body.attestation,
            youtube_enabled=settings.youtube_api_key is not None,
        )
    except (TranscriptSubmissionError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "content_source_id": row.id,
        "created": created,
        "status": "queued for evaluation" if created else "identical transcript already on file",
    }
