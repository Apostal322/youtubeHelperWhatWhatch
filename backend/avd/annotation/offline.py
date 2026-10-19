"""Offline annotation packets: a rating round without a server that raters connect to.

The coordinator (who holds the dataset and the database) issues each rater a single HTML file
containing their blind assignments. The rater opens it in a browser, without network access,
rates the items, and downloads a ratings file that they send back by any channel. The
coordinator imports it, and every rating goes through the same validation as an online
submission (status rules, evidence quotes verified against the transcript, QC flags).

What the packet contains, and what it never contains:

* the transcript segments of the assigned items, their attribution and licence, the rubric with
  all anchors, and the status guide;
* never model assessments, other raters' work or popularity data (``item_payload``);
* never items whose publisher has not confirmed showing the text to external raters.

A packet carries a random secret, stored hashed, which authenticates the return file in place of
the rater's login token. Timing is reported by the rater's browser, so offline annotations are
flagged ``client_timing``. The ``too_fast`` check still applies to the reported duration.
"""

from __future__ import annotations

import hmac
import json
import secrets
from datetime import UTC, datetime
from importlib import resources
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.annotation.service import (
    RATERS_PER_ITEM,
    REPEAT_AFTER,
    STATUS_GUIDE,
    AnnotationError,
    ScoreInput,
    _available,
    _items,
    _order_key,
    _split,
    _start,
    hash_token,
    item_payload,
    submit_annotation,
)
from avd.db.models import Annotation, AnnotationPacket, Dataset, Rater, RelevanceJudgment

PACKET_FORMAT = "avd-annotation-packet"
RETURN_FORMAT = "avd-annotation-return"
FORMAT_VERSION = "1.0"
FLAGS = ("ambiguous_rubric", "insufficient_material", "transcript_problem", "sensitive_content")
RELEVANCE_GRADES = (0, 1, 2)


def _relevance_queries() -> list[dict[str, str]]:
    from avd.bench.relevance import load_queries

    return [{"query_id": q.query_id, "text": q.text} for q in load_queries()]


def _open_counts(session: Session, dataset_id: str) -> dict[str, int]:
    """Independent ratings submitted or currently assigned, per item (so packets never assign an
    item to more raters than it needs)."""
    rows = session.execute(
        select(Annotation.video_id, func.count())
        .where(
            Annotation.dataset_id == dataset_id,
            Annotation.status.in_(("submitted", "in_progress")),
            Annotation.is_repeat.is_(False),
        )
        .group_by(Annotation.video_id)
    ).all()
    return {vid: int(n) for vid, n in rows}


def issue_packet(
    session: Session, rater: Rater, dataset_id: str, max_items: int = 3, splits: tuple[str, ...] = ("dev",)
) -> tuple[AnnotationPacket, str, list[dict[str, Any]]]:
    """Assign up to ``max_items`` items. A rater who is still onboarding gets the calibration
    items first, and nothing else, until those are returned."""
    if session.get(Dataset, dataset_id) is None:
        raise AnnotationError(f"unknown dataset {dataset_id}")
    if rater.status == "suspended":
        raise AnnotationError("rater is suspended")
    if not 1 <= max_items <= 20:
        raise AnnotationError("a packet holds 1 to 20 items")
    if set(splits) - {"dev", "test"}:
        raise AnnotationError("packets assign dev or test items (calibration is automatic)")
    mine = session.scalars(
        select(Annotation).where(Annotation.rater_id == rater.id, Annotation.dataset_id == dataset_id)
    ).all()
    touched = {a.video_id for a in mine if a.status in ("submitted", "in_progress") and not a.is_repeat}
    anns: list[Annotation] = []
    if rater.status == "onboarding":
        pending = [
            i
            for i in _items(session, dataset_id, ("calibration",))
            if i.video_id not in touched and _available(session, i.video_id)
        ]
        pending.sort(key=lambda i: _order_key(rater.id, i.video_id))
        anns = [_start(session, rater, dataset_id, i.video_id, is_repeat=False) for i in pending]
        if not anns and any(a.status == "in_progress" for a in mine):
            raise AnnotationError("return the calibration packet first")
    if not anns:
        counts = _open_counts(session, dataset_id)
        candidates = [
            i
            for i in _items(session, dataset_id, splits)
            if i.video_id not in touched
            and counts.get(i.video_id, 0) < RATERS_PER_ITEM
            and _available(session, i.video_id)
        ]
        candidates.sort(key=lambda i: (counts.get(i.video_id, 0), _order_key(rater.id, i.video_id)))
        anns = [
            _start(session, rater, dataset_id, i.video_id, is_repeat=False) for i in candidates[:max_items]
        ]
        # Test-retest: one blind repeat of an item the rater already submitted, when there are enough.
        first_pass = [
            a.video_id
            for a in mine
            if a.status == "submitted"
            and not a.is_repeat
            and _split(session, dataset_id, a.video_id) != "calibration"
        ]
        repeated = {a.video_id for a in mine if a.is_repeat}
        eligible = sorted(
            (v for v in first_pass if v not in repeated and _available(session, v)),
            key=lambda v: _order_key(rater.id + ":repeat", v),
        )
        if anns and len(first_pass) >= REPEAT_AFTER and eligible:
            rep = _start(session, rater, dataset_id, eligible[0], is_repeat=True)
            anns.insert(int(_order_key(rater.id, rep.id), 16) % (len(anns) + 1), rep)
    if not anns:
        raise AnnotationError("nothing to assign: every item already has enough ratings or is assigned")
    payloads = [item_payload(session, a) for a in anns]
    queries = _relevance_queries()
    for a, p in zip(anns, payloads, strict=True):
        # Relevance judgments are collected for development items only (the benchmark refuses others).
        p["relevance_queries"] = queries if _split(session, dataset_id, a.video_id) == "dev" else []
    secret = "pkt_" + secrets.token_urlsafe(24)
    packet = AnnotationPacket(
        dataset_id=dataset_id,
        rater_id=rater.id,
        annotation_ids=[a.id for a in anns],
        secret_hash=hash_token(secret),
    )
    session.add(packet)
    session.flush()
    return packet, secret, payloads


def render_packet(packet: AnnotationPacket, rater: Rater, payloads: list[dict[str, Any]], secret: str) -> str:
    """A single self-contained HTML file. Its content security policy forbids every network
    request, so it works offline and cannot send anything anywhere."""
    data = {
        "format": PACKET_FORMAT,
        "format_version": FORMAT_VERSION,
        "packet_id": packet.id,
        "secret": secret,
        "rater": rater.pseudonym,
        "dataset_id": packet.dataset_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "flags": list(FLAGS),
        "status_guide": STATUS_GUIDE,
        "items": payloads,
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
    template = resources.files("avd.annotation").joinpath("static/packet.html").read_text(encoding="utf-8")
    return template.replace("__PACKET_JSON__", blob)


def import_return(session: Session, data: dict[str, Any]) -> dict[str, Any]:
    """Validate and record a returned ratings file. Each item is accepted or rejected on its
    own; rejected items stay assigned, so the rater can correct them and send the file again."""
    if data.get("format") != RETURN_FORMAT or not str(data.get("format_version", "")).startswith("1."):
        raise AnnotationError("not an annotation return file")
    packet = session.get(AnnotationPacket, str(data.get("packet_id")))
    if packet is None:
        raise AnnotationError("unknown packet")
    if not hmac.compare_digest(packet.secret_hash, hash_token(str(data.get("secret", "")))):
        raise AnnotationError("the packet secret does not match")
    if packet.status == "released":
        raise AnnotationError("this packet was released; its items were reassigned")
    rater = session.get(Rater, packet.rater_id)
    if rater is None or rater.status == "suspended":
        raise AnnotationError("rater is unknown or suspended")
    accepted, rejected, already = [], [], []
    relevance_added = 0
    for res in data.get("results") or []:
        ann_id = str(res.get("annotation_id"))
        if ann_id not in packet.annotation_ids:
            rejected.append({"annotation_id": ann_id, "reason": "not part of this packet"})
            continue
        ann = session.get(Annotation, ann_id)
        if ann is None:
            rejected.append({"annotation_id": ann_id, "reason": "assignment no longer exists"})
            continue
        if ann.status != "in_progress":
            already.append(ann_id)
            continue
        try:
            with session.begin_nested():
                scores = [
                    ScoreInput(
                        dimension=str(s.get("dimension")),
                        status=str(s.get("status")),
                        score=s.get("score"),
                        confidence=s.get("confidence"),
                        evidence=list(s.get("evidence") or []),
                        note=s.get("note"),
                    )
                    for s in res.get("scores") or []
                ]
                flags = [f for f in res.get("flags") or [] if f in FLAGS]
                duration = res.get("client_duration_seconds")
                done = submit_annotation(
                    session,
                    rater,
                    ann_id,
                    scores,
                    res.get("confidence"),
                    flags,
                    res.get("notes"),
                    res.get("enjoyment"),
                    client_duration_seconds=float(duration) if duration is not None else 0.0,
                )
                done.flags = sorted({*done.flags, "offline_packet"})
                relevance_added += _record_relevance(session, rater, ann, res.get("relevance") or [])
            accepted.append(ann_id)
        except (AnnotationError, TypeError, ValueError) as exc:
            rejected.append({"annotation_id": ann_id, "reason": str(exc)})
    remaining = [
        a
        for a in packet.annotation_ids
        if (x := session.get(Annotation, a)) is not None and x.status == "in_progress"
    ]
    if not remaining:
        packet.status = "returned"
        packet.returned_at = datetime.now(UTC)
    if rater.status == "onboarding":
        calib = {i.video_id for i in _items(session, packet.dataset_id, ("calibration",))}
        done_calib = {
            a.video_id
            for a in session.scalars(
                select(Annotation).where(Annotation.rater_id == rater.id, Annotation.status == "submitted")
            ).all()
        }
        if calib and calib <= done_calib:
            rater.status = "active"
            rater.onboarded_at = datetime.now(UTC)
    session.flush()
    return {
        "packet_id": packet.id,
        "rater": rater.pseudonym,
        "accepted": accepted,
        "already_imported": already,
        "rejected": rejected,
        "still_open": remaining,
        "relevance_judgments": relevance_added,
        "rater_status": rater.status,
    }


def _record_relevance(session: Session, rater: Rater, ann: Annotation, rows: list[dict[str, Any]]) -> int:
    """Store graded relevance for a development item. Unknown queries or grades reject the item;
    a judgment that already exists (e.g. from a blind repeat) is kept, not overwritten."""
    if not rows:
        return 0
    if _split(session, ann.dataset_id, ann.video_id) != "dev":
        raise AnnotationError("relevance judgments are collected for development items only")
    known = {q["query_id"] for q in _relevance_queries()}
    added = 0
    for row in rows:
        qid, grade = str(row.get("query_id")), row.get("grade")
        if qid not in known:
            raise AnnotationError(f"unknown relevance query {qid!r}")
        if grade is None:  # "can't tell" is not a judgment
            continue
        if grade not in RELEVANCE_GRADES:
            raise AnnotationError(f"relevance grade must be 0, 1 or 2 (got {grade!r})")
        exists = session.scalar(
            select(RelevanceJudgment).where(
                RelevanceJudgment.query_id == qid,
                RelevanceJudgment.video_id == ann.video_id,
                RelevanceJudgment.rater_id == rater.id,
            )
        )
        if exists is None:
            session.add(
                RelevanceJudgment(query_id=qid, video_id=ann.video_id, rater_id=rater.id, grade=int(grade))
            )
            added += 1
    session.flush()
    return added


def export_relevance(session: Session) -> list[dict[str, Any]]:
    """Rows in the ``avd bench-relevance --judgments`` format (pseudonymous rater)."""
    rows = session.execute(
        select(RelevanceJudgment, Rater.pseudonym)
        .join(Rater, Rater.id == RelevanceJudgment.rater_id)
        .order_by(RelevanceJudgment.query_id, RelevanceJudgment.video_id, Rater.pseudonym)
    ).all()
    return [
        {"query_id": j.query_id, "video_id": j.video_id, "rater": name, "grade": j.grade} for j, name in rows
    ]


def release_packet(session: Session, packet_id: str) -> int:
    """Withdraw unreturned assignments (for a rater who stopped), so the items can be reassigned.
    Submitted ratings are kept."""
    packet = session.get(AnnotationPacket, packet_id)
    if packet is None:
        raise AnnotationError("unknown packet")
    n = 0
    for ann_id in packet.annotation_ids:
        ann = session.get(Annotation, ann_id)
        if ann is not None and ann.status == "in_progress":
            session.delete(ann)
            n += 1
    packet.status = "released"
    session.flush()
    return n
