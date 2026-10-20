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
