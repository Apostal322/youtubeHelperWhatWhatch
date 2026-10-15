"""Operator endpoints (require ``X-Admin-Token``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.api.deps import db, require_admin
from avd.api.schemas import DiscoverIn
from avd.config import Settings, get_settings
from avd.db.models import Channel, CostLedger, Evaluation, PipelineJob, Video
from avd.workers import queue
from avd.workers.budget import BudgetManager, llm_spent_today, youtube_units_today

router = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_admin)])


@router.post("/seed-demo")
def seed_demo(session: Session = Depends(db), settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    from avd.services import seed_demo as _seed

    return {"seeded": _seed(session, settings)}


@router.post("/discover")
def discover(
    body: DiscoverIn, session: Session = Depends(db), settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    if settings.youtube_api_key is None:
        return {"enqueued": 0, "note": "YouTube Data API not configured (AVD_YOUTUBE_API_KEY)"}
    from avd.services import plan_discovery

    return {"enqueued": plan_discovery(session, body.n_queries)}


@router.post("/reevaluate-stale")
def reevaluate_stale(session: Session = Depends(db)) -> dict[str, Any]:
    job, created = queue.enqueue(
        session, "reevaluate_stale", f"reevaluate_stale:{datetime.now(UTC):%Y%m%d%H%M}", priority=2.0
    )
    return {"job_id": job.id, "created": created}


@router.post("/worker/run")
def run_worker(max_jobs: int = Query(200, le=5000)) -> dict[str, Any]:
    """Drain the queue synchronously (inline mode / demos). Use Celery workers in production."""
    from avd.workers.context import build_context
    from avd.workers.runner import run_until_idle

    return {"outcomes": run_until_idle(build_context(), max_jobs=max_jobs)}


@router.get("/jobs")
def jobs(
    status: str | None = None, limit: int = Query(50, le=500), session: Session = Depends(db)
) -> dict[str, Any]:
    q = select(PipelineJob).order_by(PipelineJob.updated_at.desc()).limit(limit)
    if status:
        q = q.where(PipelineJob.status == status)
    return {
        "stats": queue.stats(session),
        "jobs": [
            {
                "id": j.id,
                "type": j.job_type,
                "video_id": j.video_id,
                "status": j.status,
                "priority": j.priority,
                "attempts": j.attempts,
                "estimated_cost_usd": j.estimated_cost_usd,
                "actual_cost_usd": j.actual_cost_usd,
                "last_error": (j.last_error or "")[:500],
                "result": j.result,
                "updated_at": j.updated_at.isoformat() if j.updated_at else None,
            }
            for j in session.scalars(q).all()
        ],
    }


@router.get("/costs")
def costs(session: Session = Depends(db), settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    by_model = session.execute(
        select(
            CostLedger.provider,
            CostLedger.model,
            CostLedger.operation,
            func.count(),
            func.sum(CostLedger.input_tokens),
            func.sum(CostLedger.output_tokens),
            func.sum(CostLedger.quota_units),
            func.sum(CostLedger.cost_usd),
        ).group_by(CostLedger.provider, CostLedger.model, CostLedger.operation)
    ).all()
    n_eval = session.scalar(select(func.count()).select_from(Evaluation)) or 0
    total = session.scalar(select(func.coalesce(func.sum(Evaluation.cost_usd), 0.0))) or 0.0
    return {
        "today": {
            "llm_usd": round(llm_spent_today(session), 6),
            "llm_budget_usd": settings.daily_llm_budget_usd,
            "llm_remaining_usd": round(BudgetManager(settings).remaining(session), 6),
            "youtube_quota_units": youtube_units_today(session),
            "youtube_quota_limit": settings.youtube_daily_quota_units,
        },
        "evaluations": n_eval,
        "mean_cost_per_evaluation_usd": round(total / n_eval, 6) if n_eval else None,
        "ledger": [
            {
                "provider": p,
                "model": m,
                "operation": o,
                "calls": n,
                "input_tokens": it or 0,
                "output_tokens": ot or 0,
                "quota_units": qu or 0,
                "cost_usd": round(c or 0.0, 6),
            }
            for p, m, o, n, it, ot, qu, c in by_model
        ],
    }


@router.get("/coverage")
def coverage(session: Session = Depends(db)) -> dict[str, Any]:
    """Catalog vs. evaluated coverage across genres, channel-size strata and languages."""

    def counts(col: Any, evaluated: bool) -> dict[str, int]:
        q = select(col, func.count()).select_from(Video).join(Channel, Channel.id == Video.channel_id)
        if evaluated:
            q = q.where(Video.status == "evaluated")
        return {str(k or "unknown"): n for k, n in session.execute(q.group_by(col)).all()}

    return {
        dim: {"catalog": counts(col, False), "evaluated": counts(col, True)}
        for dim, col in (
            ("genre", Video.genre),
            ("channel_size", Channel.size_bucket),
            ("language", Video.language),
            ("status", Video.status),
        )
    }
