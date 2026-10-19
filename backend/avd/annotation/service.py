"""Human annotation workflow for real-world datasets.

Principles (``docs/annotation.md``):

* **Blind and independent.** A rater never sees another rater's scores, the system's evaluation
  or popularity data. Assignment favours the items with the fewest independent ratings.
* **Onboarding first.** New raters only receive calibration items; they become active after
  annotating every calibration item. Calibration annotations never enter agreement metrics.
* **Server-side timing.** Start and submit times are recorded by the server; durations are not
  self-reported.
* **Evidence is verified** against the transcript the rater saw (same matcher as model evidence).
* **QC flags** (too fast, straight-lining, missing evidence, repeat inconsistency) are recorded,
  not silently acted on; exclusion is an explicit, logged admin decision.
* **Disagreement** (score range >= 2, or scored vs. abstained) goes to adjudication by an
  adjudicator who did not rate that item.
* Nothing here generates labels. If no human has annotated, exports are empty.
"""

from __future__ import annotations

import hashlib
import secrets
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.content.models import Transcript
from avd.content.store import load_bundle
from avd.datasets.rights import video_pending_uses
from avd.db.models import (
    Adjudication,
    Annotation,
    AnnotationScore,
    ContentSource,
    Dataset,
    DatasetItem,
    PairwiseLabel,
    Rater,
    Video,
)
from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS
from avd.evaluation.rubric import RUBRIC
from avd.evaluation.schema import RUBRIC_VERSION, EvidenceRef
from avd.evaluation.validation import verify_quote

ANNOTATION_DIMENSIONS = tuple(d.value for d in TEXT_DIMENSIONS)
SCORE_STATUSES = (
    "scored",
    "insufficient_evidence",
    "not_applicable",
    "cannot_determine",
    "requires_subject_expertise",
    "requires_visual_material",
    "ambiguous",
    "abstain",
)
# Shown to raters next to the status selector. No status forces a number.
STATUS_GUIDE = {
    "scored": "You can justify a 1-5 score from the transcript; cite at least one segment.",
    "insufficient_evidence": "The transcript does not contain enough material on this dimension.",
    "not_applicable": "The dimension does not apply to this kind of content.",
    "cannot_determine": "There is material, but you cannot reach a justified judgement.",
    "requires_subject_expertise": "Judging this needs domain expertise you do not have (e.g. checking technical claims).",
    "requires_visual_material": "Judging this needs the video or audio, not just the transcript.",
    "ambiguous": "The rubric question is unclear for this item; say why in the note.",
    "abstain": "You prefer not to rate this dimension for another reason; say why in the note.",
}
PAIR_CRITERIA = ("content", "presentation", "overall")
PAIR_PREFERENCES = ("a", "b", "equal", "insufficient")
RATERS_PER_ITEM = 3
REPEAT_AFTER = 5  # once a rater has this many submissions, every REPEAT_EVERY-th assignment is a repeat
REPEAT_EVERY = 6
MAX_READING_WPM = 1000  # faster than this over the transcript => "too_fast" flag
DISAGREEMENT_RANGE = 2


class AnnotationError(ValueError):
    """Invalid annotation request (maps to HTTP 4xx)."""


def _aware(dt: datetime) -> datetime:
    # SQLite returns naive datetimes; all stored times are UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ------------------------------------------------------------------------------------------ raters


def create_rater(
    session: Session, pseudonym: str, expertise: list[str], role: str = "rater"
) -> tuple[Rater, str]:
    if role not in ("rater", "adjudicator"):
        raise AnnotationError("role must be 'rater' or 'adjudicator'")
    if session.scalar(select(Rater).where(Rater.pseudonym == pseudonym)) is not None:
        raise AnnotationError("pseudonym already exists")
    token = "rtr_" + secrets.token_urlsafe(24)
    rater = Rater(pseudonym=pseudonym, token_hash=hash_token(token), expertise=expertise, role=role)
    session.add(rater)
    session.flush()
    return rater, token  # the token is shown once and only its hash is stored


def rater_for_token(session: Session, token: str) -> Rater | None:
    rater = session.scalar(select(Rater).where(Rater.token_hash == hash_token(token)))
    if rater is None or rater.status == "suspended":
        return None
    return rater


# -------------------------------------------------------------------------------------- assignment


def _items(session: Session, dataset_id: str, splits: tuple[str, ...]) -> list[DatasetItem]:
    return list(
        session.scalars(
            select(DatasetItem).where(DatasetItem.dataset_id == dataset_id, DatasetItem.split.in_(splits))
        ).all()
    )


def _order_key(rater_id: str, video_id: str) -> str:
    # Per-rater pseudo-random order so raters do not all start on the same item.
    return hashlib.sha256(f"{rater_id}:{video_id}".encode()).hexdigest()


def _submitted_counts(session: Session, dataset_id: str) -> dict[str, int]:
    rows = session.execute(
        select(Annotation.video_id, func.count())
        .where(
            Annotation.dataset_id == dataset_id,
            Annotation.status == "submitted",
            Annotation.is_repeat.is_(False),
        )
        .group_by(Annotation.video_id)
    ).all()
    return {vid: int(n) for vid, n in rows}


def next_assignment(session: Session, rater: Rater, dataset_id: str) -> Annotation | None:
    """Return the rater's in-progress annotation, or start a new one on the next item."""
    if session.get(Dataset, dataset_id) is None:
        raise AnnotationError(f"unknown dataset {dataset_id}")
    open_ann = session.scalar(
        select(Annotation).where(
            Annotation.rater_id == rater.id,
            Annotation.dataset_id == dataset_id,
            Annotation.status == "in_progress",
        )
    )
    if open_ann is not None:
        return open_ann
    mine = session.scalars(
        select(Annotation).where(
            Annotation.rater_id == rater.id,
            Annotation.dataset_id == dataset_id,
            Annotation.status == "submitted",
        )
    ).all()
    done = {a.video_id for a in mine}
    if rater.status == "onboarding":
        pending = [
            i
            for i in _items(session, dataset_id, ("calibration",))
            if i.video_id not in done and _available(session, i.video_id)
        ]
        if not pending:
            rater.status = "active"
            rater.onboarded_at = datetime.now(UTC)
        else:
            pending.sort(key=lambda i: _order_key(rater.id, i.video_id))
            return _start(session, rater, dataset_id, pending[0].video_id, is_repeat=False)
    # Repeat (test-retest) check: blind re-annotation of an item the rater already rated.
    first_pass = [
        a.video_id
        for a in mine
        if not a.is_repeat and _split(session, dataset_id, a.video_id) != "calibration"
    ]
    repeated = {a.video_id for a in mine if a.is_repeat}
    eligible_repeat = [v for v in first_pass if v not in repeated and _available(session, v)]
    if len(first_pass) >= REPEAT_AFTER and (len(mine) + 1) % REPEAT_EVERY == 0 and eligible_repeat:
        eligible_repeat.sort(key=lambda v: _order_key(rater.id + ":repeat", v))
        return _start(session, rater, dataset_id, eligible_repeat[0], is_repeat=True)
    counts = _submitted_counts(session, dataset_id)
    candidates = [
        i
        for i in _items(session, dataset_id, ("dev", "test"))
        if i.video_id not in done
        and counts.get(i.video_id, 0) < RATERS_PER_ITEM
        and _available(session, i.video_id)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda i: (counts.get(i.video_id, 0), _order_key(rater.id, i.video_id)))
    return _start(session, rater, dataset_id, candidates[0].video_id, is_repeat=False)


def _split(session: Session, dataset_id: str, video_id: str) -> str | None:
    return session.scalar(
        select(DatasetItem.split).where(
            DatasetItem.dataset_id == dataset_id, DatasetItem.video_id == video_id
        )
    )


def _available(session: Session, video_id: str) -> bool:
    # Revoked content is not annotated, nor content whose publisher has not confirmed that it may
    # be shown to external raters (avd.datasets.rights).
    if "external_annotation" in video_pending_uses(session, video_id):
        return False
    return load_bundle(session, video_id).transcript is not None


def _start(session: Session, rater: Rater, dataset_id: str, video_id: str, is_repeat: bool) -> Annotation:
    ann = Annotation(
        dataset_id=dataset_id,
        video_id=video_id,
        rater_id=rater.id,
        rubric_version=RUBRIC_VERSION,
        status="in_progress",
        is_repeat=is_repeat,
        started_at=datetime.now(UTC),
    )
    session.add(ann)
    session.flush()
    return ann


def item_payload(session: Session, ann: Annotation) -> dict[str, Any]:
    """What a rater sees: the transcript, descriptive metadata, attribution and the rubric.
    Deliberately excluded: other annotations, model evaluations, popularity statistics."""
    if "external_annotation" in video_pending_uses(session, ann.video_id):
        raise AnnotationError("item is not cleared for external annotation (rights confirmation pending)")
    video = session.get(Video, ann.video_id)
    bundle = load_bundle(session, ann.video_id)
    if video is None or bundle.transcript is None:
        raise AnnotationError("item content is unavailable (revoked?)")
    src = session.get(ContentSource, bundle.transcript_source_id) if bundle.transcript_source_id else None
    return {
        "annotation_id": ann.id,
        "dataset_id": ann.dataset_id,
        "video_id": video.id,
        "title": video.title,
        "platform": video.platform,
        "is_calibration": _split(session, ann.dataset_id, ann.video_id) == "calibration",
        "attribution": src.attribution if src else None,
        "license": src.license if src else None,
        "provenance_url": src.provenance_url if src else None,
        "timing_note": _timing_note(src),
        "segments": [
            {"index": i, "start": s.start, "end": s.end, "text": s.text}
            for i, s in enumerate(bundle.transcript.segments)
        ],
        "words": bundle.transcript.word_count,
        "rubric_version": RUBRIC_VERSION,
        "rubric": [
            {
                "dimension": d,
                "question": RUBRIC[dim].question,
                "anchors": {str(k): v for k, v in sorted(RUBRIC[dim].anchors.items())},
                "exclusions": RUBRIC[dim].exclusions,
            }
            for d, dim in ((d.value, d) for d in TEXT_DIMENSIONS)
        ],
        "statuses": list(SCORE_STATUSES),
        "status_guide": STATUS_GUIDE,
        "blinding": (
            "Rate the content, not the speakers or the show. You will not see any automated "
            "assessment of this item; please do not look for one before submitting."
        ),
        "enjoyment_question": (
            "Optional: how much did you personally enjoy this? (1-5). This is recorded separately "
            "and is never used as a quality label."
        ),
        "started_at": ann.started_at.isoformat(),
    }


def _timing_note(src: ContentSource | None) -> str | None:
    if src and (src.preprocessing or {}).get("parser", {}).get("timing") == "source_anchor_spans":
        return "Timestamps are the source's span markers (every ~1 min); several paragraphs share a span."
    return None


# -------------------------------------------------------------------------------------- submission


@dataclass
class ScoreInput:
    dimension: str
    status: str
    score: int | None = None
    confidence: float | None = None
    evidence: list[dict[str, Any]] | None = None
    note: str | None = None


def submit_annotation(
    session: Session,
    rater: Rater,
    annotation_id: str,
    scores: list[ScoreInput],
    confidence: float | None,
    flags: list[str],
    notes: str | None,
    enjoyment: int | None = None,
    client_duration_seconds: float | None = None,
) -> Annotation:
    """``client_duration_seconds`` is used only for offline packets, where the server cannot
    time the work; such annotations carry the ``client_timing`` flag."""
    ann = session.get(Annotation, annotation_id)
    if ann is None or ann.rater_id != rater.id:
        raise AnnotationError("annotation not found")
    if ann.status != "in_progress":
        raise AnnotationError("annotation already submitted")
    seen = {s.dimension for s in scores}
    if len(seen) != len(scores):
        raise AnnotationError("duplicate dimension")
    missing = set(ANNOTATION_DIMENSIONS) - seen
    if missing:
        raise AnnotationError(f"every dimension needs a status; missing: {sorted(missing)}")
    if seen - set(ANNOTATION_DIMENSIONS):
        raise AnnotationError(f"unknown dimensions: {sorted(seen - set(ANNOTATION_DIMENSIONS))}")
    transcript = load_bundle(session, ann.video_id).transcript
    if transcript is None:
        raise AnnotationError("item content is unavailable (revoked?)")
    if enjoyment is not None and not 1 <= enjoyment <= 5:
        raise AnnotationError("enjoyment must be 1-5 (or omitted)")
    qc: list[str] = []
    rows: list[AnnotationScore] = []
    for s in scores:
        if s.status not in SCORE_STATUSES:
            raise AnnotationError(f"{s.dimension}: invalid status {s.status}")
        if (s.status == "scored") != (s.score is not None):
            raise AnnotationError(f"{s.dimension}: a score is required iff status is 'scored'")
        if s.score is not None and not 1 <= s.score <= 5:
            raise AnnotationError(f"{s.dimension}: score must be 1-5")
        if s.confidence is not None and not 0 <= s.confidence <= 1:
            raise AnnotationError(f"{s.dimension}: confidence must be within [0, 1]")
        evidence = [_verify_rater_evidence(e, transcript, s.dimension) for e in (s.evidence or [])]
        if s.status == "scored" and not evidence:
            qc.append(f"missing_evidence:{s.dimension}")
        rows.append(
            AnnotationScore(
                dimension=s.dimension,
                status=s.status,
                score=s.score,
                confidence=s.confidence,
                evidence=evidence,
                note=(s.note or None),
            )
        )
    now = datetime.now(UTC)
    if client_duration_seconds is not None:
        if not 0 <= client_duration_seconds <= 7 * 24 * 3600:
            raise AnnotationError("reported duration is out of range")
        ann.duration_seconds = round(client_duration_seconds, 1)
        qc.append("client_timing")
    else:
        ann.duration_seconds = round((now - _aware(ann.started_at)).total_seconds(), 1)
    min_seconds = transcript.word_count / MAX_READING_WPM * 60
    if ann.duration_seconds < min_seconds:
        qc.append("too_fast")
    scored = [r.score for r in rows if r.score is not None]
    if len(scored) >= 5 and len(set(scored)) == 1:
        qc.append("straight_lining")
    ann.scores = rows
    ann.enjoyment = enjoyment
    ann.submitted_at = now
    ann.status = "submitted"
    ann.confidence = confidence
    ann.flags = sorted(set(flags) | set(qc))
    ann.notes = notes
    session.flush()
    return ann


def _verify_rater_evidence(e: dict[str, Any], transcript: Transcript, dim: str) -> dict[str, Any]:
    try:
        idx = int(e["segment_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationError(f"{dim}: evidence needs an integer segment_index") from exc
    quote = (e.get("quote") or "").strip()
    if not 0 <= idx < len(transcript.segments):
        raise AnnotationError(f"{dim}: segment {idx} does not exist")
    if quote:
        v = verify_quote(EvidenceRef(segment_index=idx, quote=quote[:600]), transcript)
        if v is None:
            raise AnnotationError(f"{dim}: quote not found in segment {idx} or its neighbours")
        idx = v.segment_index if v.segment_index is not None else idx
    seg = transcript.segments[idx]
    kind = e.get("kind", "supporting")
    if kind not in ("supporting", "counter"):
        raise AnnotationError(f"{dim}: evidence kind must be supporting or counter")
    return {
        "segment_index": idx,
        "quote": quote[:600] or None,
        "kind": kind,
        "start": seg.start,
        "end": seg.end,
    }


# ---------------------------------------------------------------------------------------- pairwise


def next_pair(session: Session, rater: Rater, dataset_id: str, criterion: str) -> tuple[str, str] | None:
    """A pair of items in the same split that this rater has both annotated and not yet compared
    on ``criterion``. Order within the pair is randomised per rater to avoid position bias."""
    if criterion not in PAIR_CRITERIA:
        raise AnnotationError(f"criterion must be one of {PAIR_CRITERIA}")
    mine = session.scalars(
        select(Annotation.video_id).where(
            Annotation.rater_id == rater.id,
            Annotation.dataset_id == dataset_id,
            Annotation.status == "submitted",
            Annotation.is_repeat.is_(False),
        )
    ).all()
    by_split: dict[str, list[str]] = {}
    for vid in sorted(set(mine)):
        sp = _split(session, dataset_id, vid)
        if sp and sp != "calibration":
            by_split.setdefault(sp, []).append(vid)
    done = {
        frozenset((a, b))
        for a, b in session.execute(
            select(PairwiseLabel.video_a, PairwiseLabel.video_b).where(
                PairwiseLabel.rater_id == rater.id,
                PairwiseLabel.dataset_id == dataset_id,
                PairwiseLabel.criterion == criterion,
            )
        ).all()
    }
    options = [
        (a, b)
        for vids in by_split.values()
        for i, a in enumerate(vids)
        for b in vids[i + 1 :]
        if frozenset((a, b)) not in done
    ]
    if not options:
        return None
    options.sort(key=lambda p: _order_key(rater.id + criterion, p[0] + p[1]))
    a, b = options[0]
    return (a, b) if int(_order_key(rater.id, a + b)[0], 16) % 2 == 0 else (b, a)


def submit_pair(
    session: Session,
    rater: Rater,
    dataset_id: str,
    video_a: str,
    video_b: str,
    criterion: str,
    preference: str,
    confidence: float | None,
    rationale: str | None,
    duration_seconds: float | None,
) -> PairwiseLabel:
    if criterion not in PAIR_CRITERIA or preference not in PAIR_PREFERENCES:
        raise AnnotationError("invalid criterion or preference")
    if video_a == video_b:
        raise AnnotationError("a pair needs two different items")
    for v in (video_a, video_b):
        if _split(session, dataset_id, v) is None:
            raise AnnotationError(f"{v} is not in {dataset_id}")
    label = PairwiseLabel(
        dataset_id=dataset_id,
        rater_id=rater.id,
        video_a=video_a,
        video_b=video_b,
        criterion=criterion,
        preference=preference,
        confidence=confidence,
        rationale=rationale,
        rubric_version=RUBRIC_VERSION,
        duration_seconds=duration_seconds,
    )
    session.add(label)
    session.flush()
    return label


# ------------------------------------------------------------------------------- QC & adjudication


def _submitted(session: Session, dataset_id: str) -> list[Annotation]:
    return list(
        session.scalars(
            select(Annotation).where(Annotation.dataset_id == dataset_id, Annotation.status == "submitted")
        ).all()
    )


def disagreements(session: Session, dataset_id: str) -> list[dict[str, Any]]:
    """Items x dimensions whose independent (non-repeat, non-calibration) ratings disagree and that
    have no adjudication yet."""
    adjudicated = {
        (a.video_id, a.dimension)
        for a in session.scalars(select(Adjudication).where(Adjudication.dataset_id == dataset_id)).all()
    }
    cells: dict[tuple[str, str], list[tuple[str, AnnotationScore]]] = {}
    for ann in _submitted(session, dataset_id):
        if ann.is_repeat or _split(session, dataset_id, ann.video_id) == "calibration":
            continue
        for sc in ann.scores:
            cells.setdefault((ann.video_id, sc.dimension), []).append((ann.id, sc))
    out = []
    for (vid, dim), entries in sorted(cells.items()):
        if len(entries) < 2 or (vid, dim) in adjudicated:
            continue
        scores = [sc.score for _, sc in entries if sc.score is not None]
        statuses = {sc.status for _, sc in entries}
        reason = None
        if len(scores) >= 2 and max(scores) - min(scores) >= DISAGREEMENT_RANGE:
            reason = f"score_range_{max(scores) - min(scores)}"
        elif "scored" in statuses and len(statuses) > 1:
            reason = "scored_vs_" + "_".join(sorted(statuses - {"scored"}))
        if reason:
            out.append(
                {
                    "video_id": vid,
                    "dimension": dim,
                    "reason": reason,
                    "annotation_ids": [a for a, _ in entries],
                }
            )
    return out


def adjudicate(
    session: Session,
    adjudicator: Rater,
    dataset_id: str,
    video_id: str,
    dimension: str,
    decision: str,
    final_score: int | None,
    rationale: str,
) -> Adjudication:
    if adjudicator.role != "adjudicator":
        raise AnnotationError("only adjudicators can adjudicate")
    if decision not in ("score", "exclude", "ambiguous") or (decision == "score") != (
        final_score is not None
    ):
        raise AnnotationError("decision must be score (with final_score 1-5), exclude or ambiguous")
    if final_score is not None and not 1 <= final_score <= 5:
        raise AnnotationError("final_score must be 1-5")
    if not rationale.strip():
        raise AnnotationError("a rationale is required")
    raters_of_item = set(
        session.scalars(
            select(Annotation.rater_id).where(
                Annotation.dataset_id == dataset_id, Annotation.video_id == video_id
            )
        ).all()
    )
    if adjudicator.id in raters_of_item:
        raise AnnotationError("an adjudicator cannot adjudicate an item they rated")
    pending = {(d["video_id"], d["dimension"]): d for d in disagreements(session, dataset_id)}
    if (video_id, dimension) not in pending:
        raise AnnotationError("no open disagreement for this item and dimension")
    adj = Adjudication(
        dataset_id=dataset_id,
        video_id=video_id,
        dimension=dimension,
        adjudicator_id=adjudicator.id,
        decision=decision,
        final_score=final_score,
        rationale=rationale,
        annotation_ids=pending[(video_id, dimension)]["annotation_ids"],
    )
    session.add(adj)
    session.flush()
    return adj


def exclude_annotation(session: Session, annotation_id: str, reason: str) -> Annotation:
    ann = session.get(Annotation, annotation_id)
    if ann is None:
        raise AnnotationError("annotation not found")
    if not reason.strip():
        raise AnnotationError("an exclusion reason is required")
    ann.status = "excluded"
    ann.exclusion_reason = reason
    session.flush()
    return ann


def qc_report(session: Session, dataset_id: str) -> dict[str, Any]:
    anns = _submitted(session, dataset_id)
    raters = {r.id: r for r in session.scalars(select(Rater)).all()}
    first: dict[tuple[str, str], Annotation] = {}
    repeats: list[tuple[Annotation, Annotation]] = []
    for a in sorted(anns, key=lambda a: _aware(a.submitted_at or a.started_at)):
        key = (a.rater_id, a.video_id)
        if a.is_repeat and key in first:
            repeats.append((first[key], a))
        elif not a.is_repeat:
            first[key] = a
    per_rater: dict[str, dict[str, Any]] = {}
    for a in anns:
        r = per_rater.setdefault(
            raters[a.rater_id].pseudonym if a.rater_id in raters else a.rater_id,
            {"submitted": 0, "durations": [], "flags": {}},
        )
        r["submitted"] += 1
        if a.duration_seconds is not None:
            r["durations"].append(a.duration_seconds)
        for f in a.flags or []:
            key = f.split(":", 1)[0]
            r["flags"][key] = r["flags"].get(key, 0) + 1
    for r in per_rater.values():
        d = r.pop("durations")
        r["median_duration_seconds"] = statistics.median(d) if d else None
    repeat_diffs = []
    for a, b in repeats:
        sa = {s.dimension: s.score for s in a.scores}
        for s in b.scores:
            if s.score is not None and sa.get(s.dimension) is not None:
                repeat_diffs.append(abs(s.score - sa[s.dimension]))  # type: ignore[operator]
    counts = _submitted_counts(session, dataset_id)
    items = _items(session, dataset_id, ("calibration", "dev", "test"))
    return {
        "dataset": dataset_id,
        "items": len(items),
        "items_with_ratings": {
            str(k): sum(1 for i in items if counts.get(i.video_id, 0) == k)
            for k in range(RATERS_PER_ITEM + 1)
        },
        "submitted_annotations": len(anns),
        "raters": per_rater,
        "repeat_pairs": len(repeats),
        "repeat_mean_abs_diff": round(statistics.fmean(repeat_diffs), 3) if repeat_diffs else None,
        "open_disagreements": len(disagreements(session, dataset_id)),
        "pairwise_labels": session.scalar(
            select(func.count()).select_from(PairwiseLabel).where(PairwiseLabel.dataset_id == dataset_id)
        ),
        "note": "No labels exist until human raters submit annotations; nothing here is generated.",
    }


# -------------------------------------------------------------------------------------------- export


def export_labels(session: Session, dataset_id: str) -> dict[str, list[dict[str, Any]]]:
    """Benchmark-ready rows. ``labels``: one row per independent rating (label_source
    ``human_annotation``) plus adjudicated rows (``human_adjudication``). Repeats are exported
    separately for test-retest analysis; calibration items and excluded annotations are omitted."""
    raters = {r.id: r.pseudonym for r in session.scalars(select(Rater)).all()}
    items = {i.video_id: i for i in _items(session, dataset_id, ("calibration", "dev", "test"))}
    labels: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    for a in _submitted(session, dataset_id):
        item = items.get(a.video_id)
        if item is None or item.split == "calibration":
            continue
        for s in a.scores:
            row = {
                "video_id": a.video_id,
                "dimension": s.dimension,
                "status": s.status,
                "score": s.score,
                "confidence": s.confidence,
                "rater_id": raters.get(a.rater_id, a.rater_id),
                "label_source": "human_annotation",
                "rubric_version": a.rubric_version,
                "annotation_id": a.id,
                "split": item.split,
                "group_key": item.group_key,
                "evidence": s.evidence,
                "flags": a.flags,
                "submitted_at": _aware(a.submitted_at).isoformat() if a.submitted_at else None,
            }
            (repeats if a.is_repeat else labels).append(row)
    for adj in session.scalars(select(Adjudication).where(Adjudication.dataset_id == dataset_id)).all():
        item = items.get(adj.video_id)
        labels.append(
            {
                "video_id": adj.video_id,
                "dimension": adj.dimension,
                "status": "scored" if adj.decision == "score" else adj.decision,
                "score": adj.final_score,
                "rater_id": raters.get(adj.adjudicator_id, adj.adjudicator_id),
                "label_source": "human_adjudication",
                "rationale": adj.rationale,
                "split": item.split if item else None,
                "group_key": item.group_key if item else None,
                "submitted_at": _aware(adj.created_at).isoformat(),
            }
        )
    pairs = [
        {
            "pair_id": f"h{p.id}",
            "video_a": p.video_a,
            "video_b": p.video_b,
            "criterion": p.criterion,
            "preferred": p.preference,
            "confidence": p.confidence,
            "rater_id": raters.get(p.rater_id, p.rater_id),
            "label_source": "human_annotation",
            "rubric_version": p.rubric_version,
            "split": items[p.video_a].split if p.video_a in items else None,
            "group_key": items[p.video_a].group_key if p.video_a in items else None,
            "submitted_at": _aware(p.created_at).isoformat(),
        }
        for p in session.scalars(select(PairwiseLabel).where(PairwiseLabel.dataset_id == dataset_id)).all()
    ]
    item_rows = [
        {"video_id": v, "split": i.split, "group_key": i.group_key, "stratum": i.stratum}
        for v, i in sorted(items.items())
    ]
    enjoyment = [
        {
            "video_id": a.video_id,
            "rater_id": raters.get(a.rater_id, a.rater_id),
            "enjoyment": a.enjoyment,
            "label_source": "human_personal_enjoyment",  # not a quality label
            "split": items[a.video_id].split if a.video_id in items else None,
        }
        for a in _submitted(session, dataset_id)
        if a.enjoyment is not None and a.video_id in items and items[a.video_id].split != "calibration"
    ]
    return {"labels": labels, "repeats": repeats, "pairs": pairs, "items": item_rows, "enjoyment": enjoyment}


def source_attributions(session: Session, dataset_id: str) -> list[str]:
    """Attribution lines required by the source licences (shown in the UI and exports)."""
    vids = [i.video_id for i in _items(session, dataset_id, ("calibration", "dev", "test"))]
    rows = session.scalars(select(ContentSource.attribution).where(ContentSource.video_id.in_(vids))).all()
    return sorted({r for r in rows if r})
