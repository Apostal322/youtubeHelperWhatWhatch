"""Celery transport for the DB-backed job queue.

* ``avd.dispatch`` (scheduled by beat every few seconds) claims due jobs from ``pipeline_jobs`` —
  priority- and budget-aware — and fans them out as ``avd.execute_job`` tasks.
* ``avd.execute_job`` runs one claimed job. ``acks_late`` + lease expiry give at-least-once
  execution: if a worker dies mid-job, the lease expires and the next dispatch re-claims it.
  Handlers are idempotent (dedup keys, input fingerprints), so re-execution is safe.
* ``avd.plan_discovery`` (beat) enqueues discovery for the least-recently-run seed queries when
  the YouTube Data API is configured.
* LLM-heavy job types are routed to a separate ``deep`` queue so their concurrency and rate can
  be scaled independently from lightweight work.
"""

from __future__ import annotations

from typing import Any

from celery import Celery

from avd.config import get_settings
from avd.logging import configure_logging

settings = get_settings()
broker = settings.redis_url or "memory://"

app = Celery("avd", broker=broker)
app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    task_default_queue="light",
    task_routes={"avd.execute_deep_job": {"queue": "deep"}},
    beat_schedule={
        "dispatch": {"task": "avd.dispatch", "schedule": 5.0},
        "plan-discovery": {"task": "avd.plan_discovery", "schedule": 3600.0},
        "reevaluate-stale": {"task": "avd.enqueue_reevaluate_stale", "schedule": 6 * 3600.0},
    },
)

DEEP_JOB_TYPES = {"stage_b", "stage_c"}
_ctx: Any = None


def _context() -> Any:
    global _ctx
    if _ctx is None:
        from avd.workers.context import build_context

        configure_logging(settings.log_level, settings.log_json)
        _ctx = build_context()
    return _ctx


@app.task(name="avd.execute_job")
def execute_job_task(job_id: str) -> dict[str, Any]:
    from avd.workers.runner import execute_job

    return execute_job(_context(), job_id)


@app.task(name="avd.execute_deep_job")
def execute_deep_job_task(job_id: str) -> dict[str, Any]:
    from avd.workers.runner import execute_job

    return execute_job(_context(), job_id)


@app.task(name="avd.dispatch")
def dispatch_task(limit: int = 32) -> int:
    from avd.db.models import PipelineJob
    from avd.workers.runner import claim_batch

    ctx = _context()
    ids = claim_batch(ctx, limit=limit)
    with ctx.session_factory() as s:
        types = {j.id: j.job_type for j in s.query(PipelineJob).filter(PipelineJob.id.in_(ids))}
    for jid in ids:
        task = execute_deep_job_task if types.get(jid) in DEEP_JOB_TYPES else execute_job_task
        task.delay(jid)
    return len(ids)


@app.task(name="avd.plan_discovery")
def plan_discovery_task(n_queries: int = 4) -> int:
    ctx = _context()
    if ctx.youtube_factory is None:
        return 0
    from avd.services import plan_discovery

    with ctx.session_factory() as s:
        n = plan_discovery(s, n_queries)
        s.commit()
    return n


@app.task(name="avd.enqueue_reevaluate_stale")
def enqueue_reevaluate_stale_task() -> None:
    from datetime import UTC, datetime

    from avd.workers.queue import enqueue

    with _context().session_factory() as s:
        enqueue(s, "reevaluate_stale", f"reevaluate_stale:{datetime.now(UTC):%Y%m%d%H}", priority=2.0)
        s.commit()
