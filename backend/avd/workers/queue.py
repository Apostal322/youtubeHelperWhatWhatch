"""Durable, database-backed job queue.

The ``pipeline_jobs`` table is the source of truth for scheduling, whichever executor runs the
jobs (the inline worker or Celery):

* **Deduplication** — unique ``dedup_key`` (e.g. ``stage_b:<video>:<framework_version>``).
* **Priority** — claimed highest priority first.
* **Leases & failure recovery** — a claimed job holds a lease; if the worker dies, the lease
  expires and the job is claimable again.
* **Retries** — exponential backoff with jitter up to ``max_attempts``, then ``dead``.
* **Budget-aware claiming** — jobs with an estimated cost are only claimed if today's remaining
  LLM budget covers them; otherwise they stay queued (deferred, not failed).
* **Concurrency** — ``SELECT ... FOR UPDATE SKIP LOCKED`` on PostgreSQL.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from avd.db.models import PipelineJob
from avd.observability import correlation_id
from avd.workers.budget import BudgetManager

TERMINAL = ("succeeded", "skipped", "dead", "cancelled")


def _now() -> datetime:
    return datetime.now(UTC)


def enqueue(
    session: Session,
    job_type: str,
    dedup_key: str,
    *,
    video_id: str | None = None,
    payload: dict[str, Any] | None = None,
    priority: float = 0.0,
    estimated_cost_usd: float = 0.0,
    run_at: datetime | None = None,
    max_attempts: int = 4,
    requeue_terminal: bool = False,
) -> tuple[PipelineJob, bool]:
    """Insert a job unless one with the same dedup key exists. Returns (job, created)."""
    existing = session.scalar(select(PipelineJob).where(PipelineJob.dedup_key == dedup_key))
    if existing is not None:
        if requeue_terminal and existing.status in TERMINAL:
            existing.status = "queued"
            existing.attempts = 0
            existing.next_attempt_at = run_at or _now()
            existing.priority = priority
            existing.payload = payload or existing.payload
            existing.last_error = None
            return existing, True
        if existing.status == "queued" and priority > existing.priority:
            existing.priority = priority
        return existing, False
    job = PipelineJob(
        job_type=job_type,
        dedup_key=dedup_key,
        video_id=video_id,
        payload=payload or {},
        priority=priority,
        estimated_cost_usd=estimated_cost_usd,
        next_attempt_at=run_at or _now(),
        max_attempts=max_attempts,
        # Ties the job (and its model calls) to the API request or parent job that created it.
        correlation_id=correlation_id.get(),
    )
    try:
        with session.begin_nested():
            session.add(job)
    except IntegrityError:  # lost a race with a concurrent enqueue
        found = session.scalar(select(PipelineJob).where(PipelineJob.dedup_key == dedup_key))
        assert found is not None
        return found, False
    return job, True


BUDGET_DEFERRED = "deferred: daily LLM budget exhausted"
# A claim whose lease expired before any worker started it (a backed-up broker) is not a failed
# attempt. Re-dispatching it stays free up to this many times, then the job goes dead: no worker
# is taking work at all.
MAX_UNDELIVERED = 24


def used_attempts(job: PipelineJob) -> int:
    """Attempts that a worker actually started (claims that never reached one are not counted)."""
    return job.attempts - int((job.payload or {}).get("undelivered", 0))


def started(job: PipelineJob) -> bool:
    return (job.payload or {}).get("started_attempt") == job.attempts


def claim(
    session: Session,
    *,
    limit: int,
    lease_seconds: int,
    budget: BudgetManager | None = None,
    job_types: list[str] | None = None,
) -> list[PipelineJob]:
    now = _now()
    remaining = budget.remaining(session) if budget is not None else float("inf")
    q = (
        select(PipelineJob)
        .where(
            or_(
                (PipelineJob.status == "queued") & (PipelineJob.next_attempt_at <= now),
                # failure recovery: a worker died while holding the lease
                (PipelineJob.status == "running") & (PipelineJob.lease_expires_at < now),
            )
        )
        .order_by(PipelineJob.priority.desc(), PipelineJob.created_at)
        .limit(limit * 4)
    )
    if job_types:
        q = q.where(PipelineJob.job_type.in_(job_types))
    if remaining != float("inf"):
        # Jobs the budget cannot pay for would otherwise fill the candidate window, so free jobs
        # behind them were never claimed (with limit=1 the window is only 4 jobs).
        affordable = PipelineJob.estimated_cost_usd <= max(remaining, 0.0)
        session.execute(
            update(PipelineJob)
            .where(
                PipelineJob.status == "queued",
                PipelineJob.next_attempt_at <= now,
                ~affordable,
                or_(PipelineJob.last_error.is_(None), PipelineJob.last_error != BUDGET_DEFERRED),
            )
            .values(last_error=BUDGET_DEFERRED)
            .execution_options(synchronize_session=False)
        )
        q = q.where(affordable)
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        q = q.with_for_update(skip_locked=True)
    candidates = session.scalars(q).all()
    claimed: list[PipelineJob] = []
    committed_cost = 0.0
    for job in candidates:
        if len(claimed) >= limit:
            break
        if job.estimated_cost_usd > 0 and committed_cost + job.estimated_cost_usd > remaining:
            job.last_error = BUDGET_DEFERRED
            continue
        undelivered = job.status == "running" and not started(job)
        if job.status == "running":
            n = int((job.payload or {}).get("undelivered", 0))
            if undelivered and n >= MAX_UNDELIVERED:
                job.status = "dead"
                job.last_error = f"no worker started this job after {n} dispatches"
                continue
            if not undelivered and used_attempts(job) >= job.max_attempts:
                job.status = "dead"
                job.last_error = (job.last_error or "") + " | lease expired on final attempt"
                continue
        # Claim with a conditional update: of two workers that read the same job (SQLite has no
        # SKIP LOCKED), only the one whose update still sees the old state wins.
        won = session.execute(  # type: ignore[attr-defined]  # rowcount: a CursorResult at runtime
            update(PipelineJob)
            .where(
                PipelineJob.id == job.id,
                PipelineJob.status == job.status,
                PipelineJob.attempts == job.attempts,
            )
            .values(
                status="running",
                attempts=job.attempts + 1,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
            .execution_options(synchronize_session=False)
        ).rowcount
        if won != 1:
            continue
        session.refresh(job)
        if undelivered:
            job.payload = {
                **(job.payload or {}),
                "undelivered": int((job.payload or {}).get("undelivered", 0)) + 1,
            }
        committed_cost += job.estimated_cost_usd
        claimed.append(job)
    session.flush()
    return claimed


def complete(
    session: Session, job: PipelineJob, result: dict[str, Any], *, skipped: bool = False, cost: float = 0.0
) -> None:
    job.status = "skipped" if skipped else "succeeded"
    job.result = result
    job.actual_cost_usd = cost
    job.lease_expires_at = None
    job.last_error = None


def fail(session: Session, job: PipelineJob, error: str, *, retryable: bool, base_seconds: float) -> None:
    job.last_error = error[:4000]
    job.lease_expires_at = None
    if retryable and used_attempts(job) < job.max_attempts:
        delay = base_seconds * (2 ** (used_attempts(job) - 1)) * (1 + random.random() * 0.25)
        job.status = "queued"
        job.next_attempt_at = _now() + timedelta(seconds=delay)
    else:
        job.status = "dead"


def defer(session: Session, job: PipelineJob, until: datetime, reason: str) -> None:
    """Re-queue without consuming an attempt (quota/budget exhaustion is not a job failure)."""
    job.status = "queued"
    job.attempts = max(0, job.attempts - 1)
    job.next_attempt_at = until
    job.lease_expires_at = None
    job.last_error = f"deferred: {reason}"


def cancel(session: Session, *, job_ids: list[str] | None = None, job_types: list[str] | None = None) -> int:
    """Cancel queued jobs (all, by id or by type). A job that is already running finishes its
    current attempt; stop or pause the worker to halt it. Cancelled jobs can be requeued."""
    q = select(PipelineJob).where(PipelineJob.status == "queued")
    if job_ids:
        q = q.where(PipelineJob.id.in_(job_ids))
    if job_types:
        q = q.where(PipelineJob.job_type.in_(job_types))
    n = 0
    for job in session.scalars(q):
        job.status = "cancelled"
        job.last_error = "cancelled by operator"
        n += 1
    session.flush()
    return n


def stats(session: Session) -> dict[str, Any]:
    rows = session.execute(
        select(PipelineJob.job_type, PipelineJob.status, func.count()).group_by(
            PipelineJob.job_type, PipelineJob.status
        )
    ).all()
    out: dict[str, dict[str, int]] = {}
    for jt, st, n in rows:
        out.setdefault(jt, {})[st] = n
    return out
