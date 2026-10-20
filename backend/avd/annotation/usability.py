"""A short usability pilot for the offline rating packet, kept apart from every label.

Before volunteers give 1 to 3 hours to full calibration episodes, a 15 to 20 minute trial shows whether
the instructions, the status choices and text-selection citation work, whether partial work
survives a reload, how long reading and rating actually take, and whether consent behaves.

The pilot packet is the ordinary packet page with an *excerpt*: the first ~1,500 words (whole
segments) of the shortest calibration episode, labelled on every screen as a usability excerpt.
It is never a judgement of the episode:

* issuing it writes nothing to the database (no assignment, no packet record), so the return
  file cannot be imported as a rating: ``avd annotate import-packet`` refuses it as an unknown
  packet;
* ``usability_report`` reads a return and reports timing, choices, citation verification against
  the excerpt, unanswered dimensions and the rater's notes. It stores nothing.

Ratings of an excerpt must never be compared with, or substituted for, full-episode labels.
"""

from __future__ import annotations

import re
import secrets
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from avd.annotation.offline import (
    CONSENT_DRAFT_MARKER,
    FLAGS,
    FORMAT_VERSION,
    PACKET_FORMAT,
    RETURN_FORMAT,
    _words,
    consent_digest,
    fill_template,
)
from avd.annotation.service import STATUS_GUIDE, AnnotationError, _available, _items, item_payload
from avd.db.models import Annotation, Dataset

SCOPE = "usability_excerpt"
PACKET_PREFIX = "usability-"
DEFAULT_WORDS = 1500
_SPACE = re.compile(r"\s+")


def _excerpt(segments: list[dict[str, Any]], max_words: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = 0
    for seg in segments:
        if out and n >= max_words:
            break
        out.append(seg)
        n += len(seg["text"].split())
    return out


def issue_usability_packet(
    session: Session, dataset_id: str, label: str, *, consent_text: str, max_words: int = DEFAULT_WORDS
) -> tuple[str, dict[str, Any]]:
    """Packet data for one excerpt of the shortest calibration episode. Nothing is stored."""
    if not consent_text.strip():
        raise AnnotationError("a consent text is required (approved by the project owner)")
    if CONSENT_DRAFT_MARKER in consent_text:
        raise AnnotationError(
            f"the consent text is still marked as a draft; delete the line containing {CONSENT_DRAFT_MARKER!r} "
            "once the project owner has approved it"
        )
    if not 300 <= max_words <= 5000:
        raise AnnotationError("an excerpt holds 300 to 5000 words")
    if session.get(Dataset, dataset_id) is None:
        raise AnnotationError(f"unknown dataset {dataset_id}")
    calibration = [
        i for i in _items(session, dataset_id, ("calibration",)) if _available(session, i.video_id)
    ]
    if not calibration:
        raise AnnotationError("the dataset has no calibration item cleared for external raters")
    item = min(calibration, key=lambda i: (_words(session, i.video_id), i.video_id))
    packet_id = PACKET_PREFIX + secrets.token_hex(8)
    # A transient assignment, never added to the session: item_payload only reads its fields.
    stand_in = Annotation(
        id=f"{packet_id}:{item.video_id}",
        dataset_id=dataset_id,
        video_id=item.video_id,
        started_at=datetime.now(UTC),
    )
    payload = item_payload(session, stand_in)
    segments = _excerpt(payload["segments"], max_words)
    excerpt_words = sum(len(s["text"].split()) for s in segments)
    payload.update(
        segments=segments,
        scope=SCOPE,
        excerpt_words=excerpt_words,
        words=excerpt_words,
        episode_words=payload["words"],
        relevance_queries=[],  # relevance needs the whole episode
    )
    data = {
        "format": PACKET_FORMAT,
        "format_version": FORMAT_VERSION,
        "scope": SCOPE,
        "packet_id": packet_id,
        "secret": "",
        "rater": label,
        "dataset_id": dataset_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "flags": list(FLAGS),
        "consent_text": consent_text.strip(),
        "consent_sha256": consent_digest(consent_text),
        "status_guide": STATUS_GUIDE,
        "items": [payload],
    }
    return packet_id, data


def render_usability_packet(data: dict[str, Any]) -> str:
    return fill_template(data)


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def usability_report(
    session: Session, data: Any, *, consent_text: str, max_words: int = DEFAULT_WORDS
) -> dict[str, Any]:
    """What a pilot return says about the packet's usability. Not a rating; nothing is stored."""
    from avd.content.store import load_bundle

    if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
        raise AnnotationError("malformed ratings file (expected an object with a list of results)")
    if data.get("format") != RETURN_FORMAT or not str(data.get("packet_id", "")).startswith(PACKET_PREFIX):
        raise AnnotationError("not a usability-pilot return (full rating packets go to import-packet)")
    raw_consent = data.get("consent")
    consent: dict[str, Any] = raw_consent if isinstance(raw_consent, dict) else {}
    consent_ok = consent.get("agreed") is True and consent.get("text_sha256") == consent_digest(consent_text)
    if not consent_ok:
        raise AnnotationError("the return does not carry agreement to this consent text; nothing was read")
    items = []
    for res in data.get("results") or []:
        if not isinstance(res, dict):
            continue
        video_id = str(res.get("annotation_id", "")).partition(":")[2]
        bundle = load_bundle(session, video_id)
        if bundle.transcript is None:
            raise AnnotationError(f"unknown item {video_id!r}")
        excerpt = _excerpt(
            [{"text": s.text} for s in bundle.transcript.segments], max_words
        )  # the same cut as when issued
        statuses: Counter[str] = Counter()
        claimed = verified = 0
        unverified: list[str] = []
        raw_scores = res.get("scores")
        scores: list[Any] = raw_scores if isinstance(raw_scores, list) else []
        for s in scores:
            if not isinstance(s, dict):
                continue
            statuses[str(s.get("status") or "unanswered")] += 1
            for e in s.get("evidence") or []:
                if not isinstance(e, dict):
                    continue
                claimed += 1
                ix, quote = e.get("segment_index"), _norm(str(e.get("quote") or ""))
                ok = (
                    isinstance(ix, int)
                    and 0 <= ix < len(excerpt)
                    and bool(quote)
                    and quote in _norm(excerpt[int(ix)]["text"])
                )
                verified += ok
                if not ok:
                    unverified.append(f"{s.get('dimension')}[{ix}]")
        duration = res.get("client_duration_seconds")
        items.append(
            {
                "video_id": video_id,
                "excerpt_words": sum(len(s["text"].split()) for s in excerpt),
                "minutes_reported": round(float(duration) / 60, 1)
                if isinstance(duration, int | float)
                else None,
                "dimensions_answered": sum(n for k, n in statuses.items() if k != "unanswered"),
                "statuses": dict(statuses),
                "citations": {"made": claimed, "found_in_excerpt": verified, "not_found": unverified},
                "flags": [f for f in res.get("flags") or [] if f in FLAGS],
                "notes": res.get("notes"),
                "dimension_notes": {
                    str(s.get("dimension")): s.get("note")
                    for s in scores
                    if isinstance(s, dict) and s.get("note")
                },
            }
        )
    return {
        "scope": SCOPE,
        "is_rating": False,
        "notice": "Usability pilot on an excerpt. Not a rating of the episode; never compare it with or "
        "substitute it for full-episode labels. Nothing was stored.",
        "packet_id": data.get("packet_id"),
        "rater": data.get("rater"),
        "consent": "agreed to this text",
        "items": items,
    }
