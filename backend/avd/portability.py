"""Portable evaluation exports: ``avd-evaluation-export`` format, version 1.

An export carries what another installation or a researcher needs to compare, audit or reproduce
an assessment: the video identifier, every dimension's status, score, confidence and
uncertainty, the evaluator (models with digests, prompt, rubric, framework and config hash), the
source fingerprint, coverage and the evidence *references* (segment, timestamps, verification).

What it does **not** carry by default:

* **Transcript text, never.** Not even with ``include_quotes``.
* **Evidence quotes** are excerpts of someone else's work. They are exported as SHA-256 hashes,
  which anyone holding the same authorized transcript can verify. Quote text is included only
  with ``include_quotes=True`` and only for videos whose every active source permits public
  quotation (project-authored synthetic fixtures, or a licensed dataset whose quotation right
  is not pending review). Creator captions and user-submitted content are never quoted.
* **Personal data**: no user, preference, feedback or rater information.

Integrity: ``integrity.evaluations_sha256`` is the SHA-256 of the canonical JSON of the
``evaluations`` array. It detects corruption and tampering *after* creation. It does **not**
prove who made the file; there is no signature, so authenticity must come from where you got it.

Imports are **never authoritative**. They are validated, hash-checked and stored apart from
local evaluations (``imported_evaluations``), labelled with their declared origin and shown as
"imported, unverified". They never change local scores, rankings or benchmarks.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

FORMAT = "avd-evaluation-export"
FORMAT_VERSION = "1.0"
SUPPORTED_MAJOR = 1
MAX_IMPORT_BYTES = 50 * 1024 * 1024
MAX_EVALUATIONS = 100_000

QUOTABLE_SOURCE_TYPES = {"synthetic_fixture", "licensed_dataset"}


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------------- schema


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceRefOut(_Strict):
    kind: str = Field(max_length=16)
    segment_index: int | None = Field(default=None, ge=0)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    quote_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    quote: str | None = Field(default=None, max_length=2000)
    verified: bool


class DimensionOut(_Strict):
    dimension: str = Field(max_length=48)
    group: str = Field(max_length=24)
    status: str = Field(max_length=32)
    score: float | None = Field(default=None, ge=1, le=5)
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason_code: str | None = Field(default=None, max_length=40)
    rubric_version: str | None = Field(default=None, max_length=32)
    method: str = Field(max_length=64)
    model: str | None = Field(default=None, max_length=128)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRefOut] = Field(default_factory=list, max_length=200)


class EvaluationOut(_Strict):
    video: dict[str, Any]  # platform, video_id, title, is_synthetic
    evaluation_id: str = Field(max_length=64)
    created_at: str | None
    stage: str = Field(max_length=4)
    mode: str = Field(max_length=32)
    evaluator_source: Literal["local_ai", "hosted_ai", "heuristic", "synthetic"]
    status: str = Field(max_length=16)
    framework_version: str = Field(max_length=32)
    versions: dict[str, Any]  # framework, rubric, prompt, ...
    models: list[str] = Field(max_length=20)
    config_hash: str | None = Field(default=None, max_length=64)
    source_fingerprint: str | None = Field(default=None, max_length=128)
    genre: str | None = Field(default=None, max_length=32)
    overall: dict[str, Any]
    coverage: float | None = Field(default=None, ge=0, le=1)
    dimensions: list[DimensionOut] = Field(max_length=64)
    validation: dict[str, Any] = Field(default_factory=dict)
    rights: dict[str, Any]


class ExportFile(_Strict):
    format: Literal["avd-evaluation-export"]
    format_version: str = Field(pattern=r"^\d+\.\d+$")
    created_at: str
    producer: dict[str, Any]
    notice: str
    evaluations: list[EvaluationOut] = Field(max_length=MAX_EVALUATIONS)
    integrity: dict[str, str]


# --------------------------------------------------------------------------------- export


def _quotable(session: Session, video_id: str) -> tuple[bool, str]:
    from avd.datasets.rights import video_pending_uses
    from avd.db.models import ContentSource

    sources = session.scalars(
        select(ContentSource).where(ContentSource.video_id == video_id, ContentSource.status == "active")
    ).all()
    if not sources:
        return False, "no active content source"
    bad = sorted({s.source_type for s in sources if s.source_type not in QUOTABLE_SOURCE_TYPES})
    if bad:
        return False, f"sources not cleared for quotation: {', '.join(bad)}"
    if "public_quotes" in video_pending_uses(session, video_id):
        return False, "public quotation is pending rights review"
    return True, "every source permits public quotation"


def _evaluation_record(session: Session, ev: Any, include_quotes: bool) -> dict[str, Any]:
    from avd.db.models import ContentSource, Video
    from avd.evaluation.service import evaluator_source

    video = session.get(Video, ev.video_id)
    quotable, why = _quotable(session, ev.video_id) if include_quotes else (False, "quotes not requested")
    sources = session.scalars(
        select(ContentSource).where(ContentSource.video_id == ev.video_id, ContentSource.status == "active")
    ).all()
    return {
        "video": {
            "platform": getattr(video, "platform", None) or "youtube",
            "video_id": ev.video_id,
            "title": video.title if video else None,
            "is_synthetic": bool(video.is_synthetic) if video else ev.is_synthetic,
        },
        "evaluation_id": ev.id,
        "created_at": ev.created_at.isoformat() if ev.created_at else None,
        "stage": ev.stage,
        "mode": ev.mode,
        "evaluator_source": evaluator_source(ev),
        "status": ev.status,
        "framework_version": ev.framework_version,
        "versions": ev.versions or {},
        "models": list(ev.models or []),
        "config_hash": ev.config_hash,
        "source_fingerprint": ev.input_fingerprint,
        "genre": ev.genre,
        "overall": {
            "score": ev.overall_score,
            "confidence": ev.overall_confidence,
            "abstain_reason": (ev.aggregation or {}).get("abstain_reason"),
            "method": (ev.aggregation or {}).get("method"),
        },
        "coverage": ev.coverage,
        "dimensions": [
            {
                "dimension": d.dimension,
                "group": d.group,
                "status": d.status,
                "score": d.score,
                "confidence": d.confidence,
                "reason_code": d.reason_code,
                "rubric_version": d.rubric_version,
                "method": d.method,
                "model": d.model,
                "uncertainty": d.uncertainty or {},
                "evidence": [
                    {
                        "kind": e.kind,
                        "segment_index": e.segment_index,
                        "start_seconds": e.start_seconds,
                        "end_seconds": e.end_seconds,
                        "quote_sha256": sha256(e.quote.encode()) if e.quote else None,
                        "quote": e.quote if quotable else None,
                        "verified": bool(e.verified),
                    }
                    for e in d.evidence
                ],
            }
            for d in ev.dimensions
        ],
        "validation": {
            k: v
            for k, v in (ev.validation or {}).items()
            if k in ("passed", "evidence_claimed", "evidence_verified", "ungrounded_dimensions", "warnings")
        },
        "rights": {
            "quotes_included": quotable,
            "quotes_reason": why,
            "transcript_included": False,
            "content_sources": [
                {"source_type": s.source_type, "license": s.license, "content_sha256": s.content_hash}
                for s in sources
            ],
        },
    }


def export_evaluations(
    session: Session,
    *,
    video_ids: list[str] | None = None,
    include_quotes: bool = False,
    include_synthetic: bool = True,
) -> dict[str, Any]:
    """Current evaluations (one per video and stage) as a version-1 export document. Videos
    withheld for blind annotation are skipped, as on the public API."""
    from avd.bench.harness import _git_sha
    from avd.datasets.blinding import blinded_video_ids
    from avd.db.models import Evaluation

    q = (
        select(Evaluation)
        .where(Evaluation.is_current.is_(True))
        .order_by(Evaluation.video_id, Evaluation.stage)
    )
    if video_ids:
        q = q.where(Evaluation.video_id.in_(video_ids))
    blinded = blinded_video_ids(session)
    records = []
    for ev in session.scalars(q):
        if ev.video_id in blinded or (not include_synthetic and ev.is_synthetic):
            continue
        records.append(_evaluation_record(session, ev, include_quotes))
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "producer": {"software": "ai-video-discovery", "git_commit": _git_sha()},
        "notice": (
            "Assessments, not facts: scores are model or heuristic judgements with stated uncertainty, "
            "not validated against human raters unless stated elsewhere. No transcript text is included."
        ),
        "evaluations": records,
        "integrity": {"algorithm": "sha256", "evaluations_sha256": sha256(canonical(records))},
    }


def write_export(doc: dict[str, Any], path: Path) -> str:
    data = json.dumps(doc, indent=2, ensure_ascii=False).encode()
    path.write_bytes(data)
    return sha256(data)


# --------------------------------------------------------------------------------- import


class ImportRefused(ValueError):
    """The file was refused; nothing was imported."""


def validate_export(raw: bytes) -> ExportFile:
    if len(raw) > MAX_IMPORT_BYTES:
        raise ImportRefused(f"file larger than {MAX_IMPORT_BYTES // (1024 * 1024)} MB")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ImportRefused(f"not JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ImportRefused(f"not an {FORMAT} file")
    major = str(data.get("format_version", "")).split(".", 1)[0]
    if major != str(SUPPORTED_MAJOR):
        raise ImportRefused(
            f"unsupported format version {data.get('format_version')} (this build reads {SUPPORTED_MAJOR}.x)"
        )
    try:
        doc = ExportFile.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ImportRefused(f"invalid export: {'.'.join(map(str, first['loc']))}: {first['msg']}") from exc
    if doc.integrity.get("algorithm") != "sha256":
        raise ImportRefused("unknown integrity algorithm")
    actual = sha256(canonical(data["evaluations"]))
    if actual != doc.integrity.get("evaluations_sha256"):
        raise ImportRefused("integrity check failed: the evaluations were modified after export")
    for ev in doc.evaluations:
        for d in ev.dimensions:
            for e in d.evidence:
                if e.quote is not None and (
                    e.quote_sha256 is None or sha256(e.quote.encode()) != e.quote_sha256
                ):
                    raise ImportRefused(f"{ev.evaluation_id}: a quote does not match its hash")
    return doc


def import_evaluations(session: Session, raw: bytes, *, origin: str) -> dict[str, Any]:
    """Store a validated export as *imported, unverified* records. Local evaluations, rankings
    and benchmarks are untouched. Re-importing the same evaluation from the same origin is a no-op."""
    from avd.db.models import ImportedEvaluation

    doc = validate_export(raw)
    file_hash = sha256(raw)
    added = skipped = 0
    for ev in doc.evaluations:
        exists = session.scalar(
            select(ImportedEvaluation).where(
                ImportedEvaluation.origin == origin, ImportedEvaluation.evaluation_id == ev.evaluation_id
            )
        )
        if exists is not None:
            skipped += 1
            continue
        session.add(
            ImportedEvaluation(
                origin=origin[:200],
                evaluation_id=ev.evaluation_id,
                video_id=str(ev.video.get("video_id", ""))[:64],
                format_version=doc.format_version,
                file_sha256=file_hash,
                evaluator_source=ev.evaluator_source,
                models=ev.models,
                payload=ev.model_dump(mode="json"),
            )
        )
        added += 1
    session.flush()
    return {"imported": added, "already_present": skipped, "file_sha256": file_hash, "trusted": False}


def imported_for_video(session: Session, video_id: str) -> list[dict[str, Any]]:
    from avd.db.models import ImportedEvaluation

    rows = session.scalars(
        select(ImportedEvaluation)
        .where(ImportedEvaluation.video_id == video_id)
        .order_by(ImportedEvaluation.imported_at)
    ).all()
    return [
        {
            "origin": r.origin,
            "imported_at": r.imported_at.isoformat() if r.imported_at else None,
            "status": "imported_unverified",
            "evaluator_source": r.evaluator_source,
            "models": r.models,
            "overall": r.payload.get("overall"),
            "dimensions": [
                {k: d.get(k) for k in ("dimension", "status", "score", "confidence")}
                for d in r.payload.get("dimensions", [])
            ],
            "versions": r.payload.get("versions"),
            "file_sha256": r.file_sha256,
        }
        for r in rows
    ]
