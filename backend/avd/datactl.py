"""Data ownership for operators: backup, restore and selective clearing (``avd data ...``).

Everything an instance knows lives in its own database and media directory. Nothing is sent to
the project or anyone else. SQLite backups use SQLite's online backup API, so they are consistent
while the instance runs. PostgreSQL installations use ``pg_dump``/``pg_restore``
(``docs/self-hosting.md``).

A backup contains whatever the instance stores, which can include authorized transcripts. Keep
backups under the same access rules as the instance. Sharing them is not covered by the rights
the content was obtained under; share an evaluation export instead (``avd evaluations export``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session


class DataError(RuntimeError):
    pass


def sqlite_path(database_url: str) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise DataError(
            "this command handles SQLite; for PostgreSQL use pg_dump/pg_restore (docs/self-hosting.md)"
        )
    return Path(database_url.removeprefix("sqlite:///"))


def backup_sqlite(database_url: str, dest: Path) -> dict[str, Any]:
    src = sqlite_path(database_url)
    if not src.exists():
        raise DataError(f"no database at {src}")
    if dest.exists():
        raise DataError(f"{dest} already exists; choose a new file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(src) as a, sqlite3.connect(dest) as b:
        a.backup(b)
    dest.chmod(0o600)
    return {"source": str(src), "backup": str(dest), "bytes": dest.stat().st_size}


def restore_sqlite(backup: Path, database_url: str) -> dict[str, Any]:
    """Restore into a database file that does not exist yet (never overwrites: move the old one
    away first, so a mistaken restore cannot destroy data)."""
    dest = sqlite_path(database_url)
    if dest.exists():
        raise DataError(f"{dest} already exists; move it away before restoring")
    with backup.open("rb") as f:
        if f.read(16) != b"SQLite format 3\x00":
            raise DataError(f"{backup} is not a SQLite database")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(backup) as a, sqlite3.connect(dest) as b:
        a.backup(b)
    return {"restored": str(dest), "bytes": dest.stat().st_size}


def clear(
    session: Session,
    *,
    evaluations: bool = False,
    model_cache: bool = False,
    preferences: bool = False,
    feedback: bool = False,
    imported: bool = False,
) -> dict[str, int]:
    """Selectively delete local data. Evaluations are deleted outright (with their dimension
    scores and evidence); their videos return to "not evaluated" and are re-evaluated on demand.
    Catalog entries and authorized content sources are kept."""
    from avd.db.models import (
        Evaluation,
        Feedback,
        ImportedEvaluation,
        LLMCache,
        RecommendationImpression,
        UserPreferences,
        Video,
    )

    out: dict[str, int] = {}

    def count(model: Any) -> int:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)

    if evaluations:
        out["evaluations"] = count(Evaluation)
        for ev in session.scalars(select(Evaluation)):
            session.delete(ev)  # ORM cascade removes dimension scores and evidence
        session.execute(update(Video).where(Video.status == "evaluated").values(status="metadata_ready"))
    if model_cache:
        out["model_cache"] = count(LLMCache)
        session.execute(delete(LLMCache))
    if preferences:
        out["preferences"] = count(UserPreferences)
        session.execute(delete(UserPreferences))  # recreated with defaults on next use
    if feedback:
        out["feedback"] = count(Feedback) + count(RecommendationImpression)
        session.execute(delete(Feedback))
        session.execute(delete(RecommendationImpression))
    if imported:
        out["imported_evaluations"] = count(ImportedEvaluation)
        session.execute(delete(ImportedEvaluation))
    session.flush()
    return out


def user_export(session: Session, user_id: str) -> dict[str, Any]:
    """Everything stored about one account, for that account's owner (data portability)."""
    from datetime import UTC, datetime

    from avd.db.models import ContentSource, Feedback, UserPreferences

    prefs = session.get(UserPreferences, user_id)
    pref_out = (
        {
            c: getattr(prefs, c)
            for c in (
                "interests",
                "genre_weights",
                "dimension_weights",
                "languages",
                "depth_preference",
                "intent",
                "min_duration_minutes",
                "max_duration_minutes",
                "diversity",
                "min_quality",
                "small_creator_boost",
                "include_synthetic",
            )
            if hasattr(prefs, c)
        }
        if prefs
        else {}
    )
    fb = session.scalars(select(Feedback).where(Feedback.user_id == user_id)).all()
    subs = session.scalars(select(ContentSource).where(ContentSource.provided_by_user_id == user_id)).all()
    return {
        "format": "avd-user-data-export",
        "format_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "user_id": user_id,
        "preferences": pref_out,
        "feedback": [
            {
                "video_id": f.video_id,
                "kind": f.kind,
                "dimension": f.dimension,
                "suggested_score": f.suggested_score,
                "comment": f.comment,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in fb
        ],
        "submitted_content": [
            {
                "video_id": s.video_id,
                "source_type": s.source_type,
                "status": s.status,
                "content_sha256": s.content_hash,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in subs
        ],
        "note": "Your submitted transcripts are listed by hash; their text is yours and not repeated here.",
    }


def _without_text(value: Any) -> Any:
    """``value`` with every string removed (numbers, booleans and structure stay)."""
    if isinstance(value, dict):
        return {k: _without_text(v) for k, v in value.items() if not isinstance(v, str)}
    if isinstance(value, list):
        return [_without_text(v) for v in value if not isinstance(v, str)]
    return value


def erase_derived_content(session: Session, source_ids: set[str], media_dir: Path) -> dict[str, int]:
    """After content sources are revoked (account deletion), erase what was derived from them.

    For every evaluation that used one of the sources, current or not: evidence quotes and
    justifications are emptied and extracted claims deleted (scores, statuses and evidence
    positions stay, for audit); the evaluation stops being current. Cached model responses that
    contain an erased quote are deleted. Media files are deleted unless another active source
    still uses them (content-addressed files can be shared), and only inside ``media_dir``. The
    video loses its transcript-derived embedding and, without a current evaluation, its
    "evaluated" status. Backups made earlier are not touched.
    """
    import json

    from avd.db.models import (
        ContentSource,
        DimensionScoreRow,
        Evaluation,
        EvidenceRow,
        FactualClaim,
        LLMCache,
        Video,
    )

    counts = {
        "evaluations_withdrawn": 0,
        "quotes_erased": 0,
        "claims_deleted": 0,
        "cache_entries": 0,
        "media_files": 0,
    }
    if not source_ids:
        return counts
    evaluations = [
        ev
        for ev in session.scalars(select(Evaluation)).all()
        if source_ids & set(ev.content_source_ids or [])
    ]
    erased_quotes: set[str] = set()
    videos = set()
    for ev in evaluations:
        videos.add(ev.video_id)
        if ev.is_current:
            ev.is_current = False
            counts["evaluations_withdrawn"] += 1
        for dim in session.scalars(
            select(DimensionScoreRow).where(DimensionScoreRow.evaluation_id == ev.id)
        ).all():
            dim.justification = ""
            for e in session.scalars(
                select(EvidenceRow).where(EvidenceRow.dimension_score_id == dim.id)
            ).all():
                if e.quote:
                    erased_quotes.add(e.quote)
                    counts["quotes_erased"] += 1
                e.quote = None
                e.note = None
        counts["claims_deleted"] += (
            session.execute(delete(FactualClaim).where(FactualClaim.evaluation_id == ev.id)).rowcount or 0  # type: ignore[attr-defined]
        )
        # Copies of claims, model-reported issues and warnings: keep only counts and flags.
        ev.coverage_report = _without_text(ev.coverage_report or {})
        ev.validation = _without_text(ev.validation or {})
    # Cached responses for these videos' prompts (part summaries, claims, judgments) paraphrase the
    # revoked content, so they go whole. Rows cached before responses recorded their video are
    # matched on erased quotes instead.
    affected = videos | {
        s.video_id for s in session.scalars(select(ContentSource).where(ContentSource.id.in_(source_ids)))
    }
    probes = [q for q in erased_quotes if len(q) >= 20]
    for row in session.scalars(select(LLMCache)).all():
        resp = row.response or {}
        text = json.dumps(resp, ensure_ascii=False) if probes else ""
        if resp.get("video_id") in affected or any(q in text for q in probes):
            session.delete(row)
            counts["cache_entries"] += 1
    root = media_dir.resolve()
    revoked = session.scalars(select(ContentSource).where(ContentSource.id.in_(source_ids))).all()
    # Files are content-addressed, so an active source can share either the upload or the
    # audio derived from it (the same clip submitted for two videos).
    in_use = {
        p
        for src in session.scalars(select(ContentSource).where(ContentSource.status == "active")).all()
        for p in (src.media_uri, (src.preprocessing or {}).get("audio_wav"))
        if p
    }
    for src in revoked:
        paths = [p for p in (src.media_uri, (src.preprocessing or {}).get("audio_wav")) if p]
        for raw in paths:
            path = Path(raw).resolve()
            if raw in in_use or not path.is_relative_to(root):
                continue
            if path.is_file():
                path.unlink()
                counts["media_files"] += 1
        src.media_uri = None
        src.preprocessing = {k: v for k, v in (src.preprocessing or {}).items() if k != "audio_wav"}
        videos.add(src.video_id)
    session.flush()
    for vid in videos:
        video = session.get(Video, vid)
        if video is None:
            continue
        video.embedding = None
        video.embedding_content_hash = None
        if (
            session.scalar(
                select(Evaluation.id).where(Evaluation.video_id == vid, Evaluation.is_current.is_(True))
            )
            is None
        ):
            active = session.scalar(
                select(func.count())
                .select_from(ContentSource)
                .where(ContentSource.video_id == vid, ContentSource.status == "active")
            )
            video.status = "content_available" if active else "content_unavailable"
    session.flush()
    return counts
