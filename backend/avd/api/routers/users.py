from __future__ import annotations

import hmac
import logging
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.api.deps import backlog_guard, current_user, db, embedder, hash_token
from avd.api.schemas import FeedbackIn, PreferencesIn, TranscriptIn
from avd.config import Registration, Settings, get_settings
from avd.db.models import Evaluation, Feedback, RecommendationImpression, User, UserPreferences, Video
from avd.embeddings.base import Embedder
from avd.recsys.recommender import recommend
from avd.services import TranscriptSubmissionError, submit_user_transcript

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

# YouTube ids are 11 characters of [A-Za-z0-9_-]; local and test ids are similar but may differ in length.
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{6,32}")

MAX_FEEDBACK_PER_DAY = 500  # per user; far above honest use, bounds spam from one token


def _registration_allowed(settings: Settings, invite: str | None, admin: str | None) -> bool:
    if (
        admin
        and settings.admin_token is not None
        and hmac.compare_digest(admin.encode(), settings.admin_token.get_secret_value().encode())
    ):
        return True
    if settings.registration == Registration.OPEN:
        return True
    if settings.registration == Registration.INVITE and invite and settings.invite_codes is not None:
        codes = [c.strip() for c in settings.invite_codes.get_secret_value().split(",") if c.strip()]
        return any(hmac.compare_digest(invite.encode(), c.encode()) for c in codes)
    return False


@router.post("/users", status_code=201)
def create_user(
    session: Session = Depends(db, scope="function"),
    settings: Settings = Depends(get_settings),
    x_invite_code: str | None = Header(default=None, max_length=200),
    x_admin_token: str | None = Header(default=None, max_length=200),
) -> dict[str, str]:
    """Anonymous account for the extension. The token is shown once; only its hash is stored.
    Local instances register freely (loopback only); servers need an invite code by default."""
    if not _registration_allowed(settings, x_invite_code, x_admin_token):
        raise HTTPException(
            403,
            "registration is closed on this server"
            if settings.registration == Registration.CLOSED
            else "an invite code is required to create an account on this server",
        )
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
        "intent": p.intent or "any",
        "languages": p.languages,
        "min_duration_minutes": p.min_duration_minutes,
        "max_duration_minutes": p.max_duration_minutes,
        "diversity": p.diversity,
        "min_quality": p.min_quality,
        "small_creator_boost": p.small_creator_boost,
        "include_synthetic": p.include_synthetic,
    }


@router.get("/users/me/preferences")
def get_preferences(
    user: User = Depends(current_user), session: Session = Depends(db, scope="function")
) -> dict[str, Any]:
    prefs = session.get(UserPreferences, user.id) or UserPreferences(user_id=user.id)
    return _prefs_out(prefs)


@router.put("/users/me/preferences")
def put_preferences(
    body: PreferencesIn,
    user: User = Depends(current_user),
    session: Session = Depends(db, scope="function"),
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
    session: Session = Depends(db, scope="function"),
    emb: Embedder = Depends(embedder),
) -> dict[str, Any]:
    return recommend(session, user.id, emb, limit=limit)


@router.post("/feedback", status_code=201, response_model=None)
def feedback(
    body: FeedbackIn, user: User = Depends(current_user), session: Session = Depends(db, scope="function")
) -> dict[str, Any] | JSONResponse:
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
    dim = body.dimension.value if body.dimension else None
    # Idempotent: a retried or double-clicked submission does not create a second event.
    dup = session.scalar(
        select(Feedback).where(
            Feedback.user_id == user.id,
            Feedback.video_id == body.video_id,
            Feedback.kind == body.kind,
            Feedback.impression_id.is_(None)
            if body.impression_id is None
            else Feedback.impression_id == body.impression_id,
            Feedback.evaluation_id.is_(None)
            if body.evaluation_id is None
            else Feedback.evaluation_id == body.evaluation_id,
            Feedback.dimension.is_(None) if dim is None else Feedback.dimension == dim,
        )
    )
    if dup is not None:
        return JSONResponse({"id": dup.id, "status": "duplicate"}, status_code=200)
    since = datetime.now(UTC) - timedelta(days=1)
    recent = session.scalar(
        select(func.count())
        .select_from(Feedback)
        .where(Feedback.user_id == user.id, Feedback.created_at >= since)
    )
    if (recent or 0) >= MAX_FEEDBACK_PER_DAY:
        raise HTTPException(429, f"feedback limit reached ({MAX_FEEDBACK_PER_DAY} per day)")
    fb = Feedback(
        user_id=user.id,
        video_id=body.video_id,
        impression_id=body.impression_id,
        evaluation_id=body.evaluation_id,
        kind=body.kind,
        dimension=dim,
        suggested_score=body.suggested_score,
        comment=body.comment,
    )
    session.add(fb)
    session.flush()
    return {"id": fb.id, "status": "recorded"}


@router.post("/videos/{video_id}/transcripts", status_code=202, dependencies=[Depends(backlog_guard)])
def submit_transcript(
    video_id: str,
    body: TranscriptIn,
    user: User = Depends(current_user),
    session: Session = Depends(db, scope="function"),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if not _VIDEO_ID.fullmatch(video_id):
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


@router.put("/videos/{video_id}/media", status_code=202, dependencies=[Depends(backlog_guard)])
async def submit_media(
    video_id: str,
    request: Request,
    x_content_attestation: str = Header(..., min_length=10, max_length=2000),
    user: User = Depends(current_user),
    session: Session = Depends(db, scope="function"),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Submit authorized audio/video as the raw request body (``Content-Type`` must be a supported
    audio/video type). ``X-Content-Attestation`` states your right to submit it, e.g. that you
    are the creator. The file is probed with ffprobe and its audio decoded for analysis; nothing
    is fetched from YouTube."""
    from starlette.concurrency import run_in_threadpool

    from avd.content.media import MediaError, available
    from avd.services import MEDIA_TYPES, submit_user_media

    if not _VIDEO_ID.fullmatch(video_id):
        raise HTTPException(400, "invalid video id")
    if not available():
        raise HTTPException(503, "media processing (ffmpeg) is not installed on this server")
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype not in MEDIA_TYPES:
        raise HTTPException(415, f"unsupported media type; use one of: {', '.join(sorted(MEDIA_TYPES))}")
    limit = settings.max_media_upload_mb * 1024 * 1024
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, f"media larger than {settings.max_media_upload_mb} MB")
    upload_dir = settings.media_cache_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = upload_dir / f"{uuid.uuid4().hex}{MEDIA_TYPES[ctype]}"
    size = 0
    try:
        with tmp.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"media larger than {settings.max_media_upload_mb} MB")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty body")
        row, created = await run_in_threadpool(
            submit_user_media,
            session,
            user_id=user.id,
            video_id=video_id,
            path=tmp,
            attestation=x_content_attestation,
            media_dir=settings.media_cache_dir,
            youtube_enabled=settings.youtube_api_key is not None,
        )
    except MediaError as exc:
        raise HTTPException(422, str(exc)) from exc
    except TranscriptSubmissionError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "content_source_id": row.id,
        "created": created,
        "modalities": row.modalities,
        "status": "queued for evaluation" if created else "identical media already on file",
    }


@router.get("/users/me/export")
def export_me(
    user: User = Depends(current_user), session: Session = Depends(db, scope="function")
) -> dict[str, Any]:
    """Everything this instance stores about your account (preferences, feedback, submissions)."""
    from avd.datactl import user_export

    return user_export(session, user.id)


@router.delete("/users/me")
def delete_me(
    user: User = Depends(current_user), session: Session = Depends(db, scope="function")
) -> dict[str, Any]:
    """Delete the account and everything tied to it: preferences, feedback, recommendation
    impressions and content the user submitted. Submitted transcripts and media are revoked and
    erased. Evaluations derived from them stop being current and lose their quotes,
    justifications and extracted claims (scores and evidence positions are kept for audit);
    cached model responses containing those quotes are deleted. Pseudonymous logs outside the
    database and earlier backups are not affected."""
    from datetime import UTC, datetime

    from sqlalchemy import delete, select

    from avd.db.models import ContentSource, Feedback, RecommendationImpression

    counts: dict[str, int] = {}
    sources = session.scalars(select(ContentSource).where(ContentSource.provided_by_user_id == user.id)).all()
    revoked_ids = set()
    for src in sources:
        revoked_ids.add(src.id)
        src.status = "revoked"
        src.revoked_at = datetime.now(UTC)
        src.transcript = None
        src.provided_by_user_id = None
        src.preprocessing = {**(src.preprocessing or {}), "revocation_reason": "user account deleted"}
    from avd.config import get_settings
    from avd.datactl import erase_derived_content

    session.flush()
    counts["transcripts_revoked"] = len(sources)
    counts.update(erase_derived_content(session, revoked_ids, get_settings().media_cache_dir))
    for model, name in (
        (Feedback, "feedback"),
        (RecommendationImpression, "impressions"),
        (UserPreferences, "preferences"),
    ):
        counts[name] = session.execute(delete(model).where(model.user_id == user.id)).rowcount or 0  # type: ignore[attr-defined]
    session.delete(user)
    session.flush()
    # Counts only: no user id, token or content, so the log never identifies who left.
    log.info("account deleted: %s", counts)
    return {"deleted": True, **counts}
