"""Job execution shared by the inline worker and Celery tasks."""

from __future__ import annotations

import time
import traceback
from datetime import timedelta
from typing import Any

from avd.db.models import PipelineJob
from avd.llm.base import LLMBudgetExceeded, LLMError
from avd.logging import get_logger
from avd.observability import METRICS, job_context
from avd.workers import queue
from avd.workers.budget import BudgetExceeded, next_pacific_midnight, utc_midnight
from avd.workers.context import WorkerContext
from avd.workers.handlers import HANDLERS
from avd.youtube.client import QuotaExceeded, YouTubeAPIError

log = get_logger(__name__)


def _classify(exc: Exception) -> tuple[str, bool]:
    """-> (kind, retryable). kind in {defer_quota, defer_budget, error}."""
    if isinstance(exc, QuotaExceeded):
        return "defer_quota", True
    if isinstance(exc, BudgetExceeded):
        return "defer_budget", True
    if isinstance(exc, LLMBudgetExceeded):
        return ("defer_budget", True) if exc.scope == "daily" else ("error", False)
    if isinstance(exc, LLMError):
        return "error", exc.retryable
    if isinstance(exc, YouTubeAPIError):
        return "error", exc.retryable
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "error", True
    return "error", False


def execute_job(ctx: WorkerContext, job_id: str) -> dict[str, Any]:
    """Run one claimed job in its own transaction and record the outcome."""
    with ctx.session_factory() as session:
        job = session.get(PipelineJob, job_id)
        if job is None or job.status != "running":
            return {"job_id": job_id, "outcome": "not_running"}
        handler = HANDLERS.get(job.job_type)
        started = time.monotonic()
        try:
            if handler is None:
                raise ValueError(f"unknown job type {job.job_type}")
            with job_context(job.id, job.correlation_id):
                res = handler(ctx, session, job)
            queue.complete(session, job, res.result, skipped=res.skipped, cost=res.cost_usd)
            session.commit()
            outcome = "skipped" if res.skipped else "succeeded"
        except Exception as exc:
            session.rollback()
            job = session.get(PipelineJob, job_id)
            assert job is not None
            kind, retryable = _classify(exc)
            if kind == "defer_quota":
                queue.defer(session, job, next_pacific_midnight(), f"YouTube quota: {exc}")
                outcome = "deferred"
            elif kind == "defer_budget":
                queue.defer(session, job, utc_midnight() + timedelta(days=1), f"budget: {exc}")
                outcome = "deferred"
            else:
                log.warning(
                    "job failed",
                    extra={
                        "job_id": job_id,
                        "job_type": job.job_type,
                        "retryable": retryable,
                        "error": str(exc),
                    },
                )
                queue.fail(
                    session,
                    job,
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
                    retryable=retryable,
                    base_seconds=ctx.settings.job_retry_base_seconds,
                )
                outcome = "retry" if job.status == "queued" else "dead"
            session.commit()
        METRICS.inc("avd_jobs_total", job_type=job.job_type, outcome=outcome)
        METRICS.observe("avd_job_duration_seconds", time.monotonic() - started, job_type=job.job_type)
        log.info(
            "job finished",
            extra={
                "job_id": job_id,
                "job_type": job.job_type,
                "outcome": outcome,
                "duration_s": round(time.monotonic() - started, 3),
            },
        )
        return {"job_id": job_id, "job_type": job.job_type, "outcome": outcome}


def claim_batch(
    ctx: WorkerContext, limit: int | None = None, job_types: list[str] | None = None
) -> list[str]:
    with ctx.session_factory() as session:
        jobs = queue.claim(
            session,
            limit=limit or ctx.settings.worker_batch_size,
            lease_seconds=ctx.settings.worker_lease_seconds,
            budget=ctx.budget,
            job_types=job_types,
        )
        ids = [j.id for j in jobs]
        session.commit()
    return ids


def run_until_idle(
    ctx: WorkerContext, max_jobs: int | None = None, job_types: list[str] | None = None
) -> dict[str, int]:
    """Drain the queue (inline worker). Returns outcome counts."""
    counts: dict[str, int] = {}
    processed = 0
    while max_jobs is None or processed < max_jobs:
        ids = claim_batch(ctx, job_types=job_types)
        if not ids:
            break
        for jid in ids:
            out = execute_job(ctx, jid)
            counts[out["outcome"]] = counts.get(out["outcome"], 0) + 1
            processed += 1
    return counts


def run_forever(ctx: WorkerContext, poll_seconds: float = 5.0) -> None:  # pragma: no cover
    log.info("inline worker started")
    while True:
        counts = run_until_idle(ctx, max_jobs=100)
        if not counts:
            time.sleep(poll_seconds)
