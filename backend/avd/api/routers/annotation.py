"""Annotation endpoints.

* Raters authenticate with ``Authorization: Bearer rtr_...`` (rater tokens are separate from
  end-user tokens and stored hashed).
* Rater management, QC, exclusion and export require ``X-Admin-Token``.
* ``GET /annotate`` serves the static annotation UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from avd.annotation import service as svc
from avd.api.deps import db, require_admin
from avd.db.models import Rater

router = APIRouter(prefix="/v1/annotation")
ui_router = APIRouter()

_UI = Path(__file__).resolve().parents[2] / "annotation" / "static" / "annotate.html"


def current_rater(authorization: str | None = Header(default=None), session: Session = Depends(db)) -> Rater:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    rater = svc.rater_for_token(session, authorization[7:].strip())
    if rater is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or suspended rater token")
    return rater


def _bad(e: svc.AnnotationError) -> HTTPException:
    return HTTPException(422, str(e))


class RaterIn(BaseModel):
    pseudonym: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    expertise: list[str] = Field(default_factory=list, max_length=20)
    role: str = "rater"


class EvidenceIn(BaseModel):
    segment_index: int
    quote: str | None = Field(default=None, max_length=600)
    kind: str = "supporting"


class ScoreIn(BaseModel):
    dimension: str
    status: str
    score: int | None = None
    confidence: float | None = None
    evidence: list[EvidenceIn] = Field(default_factory=list, max_length=10)
    note: str | None = Field(default=None, max_length=2000)


class SubmitIn(BaseModel):
    scores: list[ScoreIn]
    confidence: float | None = Field(default=None, ge=0, le=1)
    flags: list[str] = Field(default_factory=list, max_length=10)
    notes: str | None = Field(default=None, max_length=4000)
    enjoyment: int | None = Field(default=None, ge=1, le=5)


class PairIn(BaseModel):
    dataset: str
    video_a: str
    video_b: str
    criterion: str
    preference: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = Field(default=None, max_length=2000)
    duration_seconds: float | None = Field(default=None, ge=0)


class AdjudicationIn(BaseModel):
    dataset: str
    video_id: str
    dimension: str
    decision: str
    final_score: int | None = None
    rationale: str = Field(min_length=1, max_length=4000)


class ExcludeIn(BaseModel):
    reason: str = Field(min_length=3, max_length=2000)


# ------------------------------------------------------------------------------------------- admin


@router.post("/raters", status_code=201, dependencies=[Depends(require_admin)])
def create_rater(body: RaterIn, session: Session = Depends(db)) -> dict[str, Any]:
    try:
        rater, token = svc.create_rater(session, body.pseudonym, body.expertise, body.role)
    except svc.AnnotationError as e:
        raise _bad(e) from e
    return {
        "id": rater.id,
        "pseudonym": rater.pseudonym,
        "role": rater.role,
        "status": rater.status,
        "token": token,
    }


@router.get("/qc", dependencies=[Depends(require_admin)])
def qc(dataset: str, session: Session = Depends(db)) -> dict[str, Any]:
    return svc.qc_report(session, dataset)


@router.get("/disagreements", dependencies=[Depends(require_admin)])
def disagreements(dataset: str, session: Session = Depends(db)) -> list[dict[str, Any]]:
    return svc.disagreements(session, dataset)


@router.post("/annotations/{annotation_id}/exclude", dependencies=[Depends(require_admin)])
def exclude(annotation_id: str, body: ExcludeIn, session: Session = Depends(db)) -> dict[str, Any]:
    try:
        ann = svc.exclude_annotation(session, annotation_id, body.reason)
    except svc.AnnotationError as e:
        raise _bad(e) from e
    return {"id": ann.id, "status": ann.status}


@router.get("/export", dependencies=[Depends(require_admin)])
def export(dataset: str, session: Session = Depends(db)) -> dict[str, Any]:
    return {**svc.export_labels(session, dataset), "attributions": svc.source_attributions(session, dataset)}


# ------------------------------------------------------------------------------------------ raters


@router.get("/me")
def me(rater: Rater = Depends(current_rater)) -> dict[str, Any]:
    return {"pseudonym": rater.pseudonym, "role": rater.role, "status": rater.status}


@router.get("/next")
def next_item(
    dataset: str, rater: Rater = Depends(current_rater), session: Session = Depends(db)
) -> dict[str, Any]:
    try:
        ann = svc.next_assignment(session, rater, dataset)
        if ann is None:
            return {"done": True, "rater_status": rater.status}
        return {"done": False, "rater_status": rater.status, **svc.item_payload(session, ann)}
    except svc.AnnotationError as e:
        raise _bad(e) from e


@router.post("/annotations/{annotation_id}/submit")
def submit(
    annotation_id: str, body: SubmitIn, rater: Rater = Depends(current_rater), session: Session = Depends(db)
) -> dict[str, Any]:
    try:
        ann = svc.submit_annotation(
            session,
            rater,
            annotation_id,
            [
                svc.ScoreInput(
                    dimension=s.dimension,
                    status=s.status,
                    score=s.score,
                    confidence=s.confidence,
                    evidence=[e.model_dump() for e in s.evidence],
                    note=s.note,
                )
                for s in body.scores
            ],
            body.confidence,
            body.flags,
            body.notes,
            enjoyment=body.enjoyment,
        )
    except svc.AnnotationError as e:
        raise _bad(e) from e
    return {"id": ann.id, "status": ann.status, "duration_seconds": ann.duration_seconds, "flags": ann.flags}


@router.get("/pairs/next")
def next_pair(
    dataset: str,
    criterion: str = "overall",
    rater: Rater = Depends(current_rater),
    session: Session = Depends(db),
) -> dict[str, Any]:
    try:
        pair = svc.next_pair(session, rater, dataset, criterion)
    except svc.AnnotationError as e:
        raise _bad(e) from e
    if pair is None:
        return {"done": True}
    return {"done": False, "dataset": dataset, "criterion": criterion, "video_a": pair[0], "video_b": pair[1]}


@router.post("/pairs", status_code=201)
def submit_pair(
    body: PairIn, rater: Rater = Depends(current_rater), session: Session = Depends(db)
) -> dict[str, Any]:
    try:
        label = svc.submit_pair(
            session,
            rater,
            body.dataset,
            body.video_a,
            body.video_b,
            body.criterion,
            body.preference,
            body.confidence,
            body.rationale,
            body.duration_seconds,
        )
    except svc.AnnotationError as e:
        raise _bad(e) from e
    return {"id": label.id}


@router.get("/adjudication-queue")
def adjudication_queue(
    dataset: str, rater: Rater = Depends(current_rater), session: Session = Depends(db)
) -> list[dict[str, Any]]:
    if rater.role != "adjudicator":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "adjudicators only")
    return svc.disagreements(session, dataset)


@router.post("/adjudications", status_code=201)
def adjudicate(
    body: AdjudicationIn, rater: Rater = Depends(current_rater), session: Session = Depends(db)
) -> dict[str, Any]:
    try:
        adj = svc.adjudicate(
            session,
            rater,
            body.dataset,
            body.video_id,
            body.dimension,
            body.decision,
            body.final_score,
            body.rationale,
        )
    except svc.AnnotationError as e:
        raise _bad(e) from e
    return {"id": adj.id}


@ui_router.get("/annotate", response_class=HTMLResponse, include_in_schema=False)
def annotate_ui() -> HTMLResponse:
    return HTMLResponse(_UI.read_text(encoding="utf-8"))
