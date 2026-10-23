"""In-process background worker for local mode (`avd start`): no Redis, no Celery.

One daemon thread drains the same durable job queue that Celery workers use, so local and
self-hosted installations share every scheduling, retry, budget and evaluation rule. It is
bounded by design: one job at a time (model calls are further limited by
``llm_max_concurrency``), it can be paused, and it stops cleanly between jobs on shutdown. An
interrupted job is not lost: on the next start its lease is released and it runs again.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from sqlalchemy import select

from avd.db.models import PipelineJob

log = logging.getLogger(__name__)


def release_stale_leases(session_factory: Any) -> int:
    """Single-process local mode only: jobs left ``running`` by a previous process cannot still be
    running, so they are re-queued without consuming an attempt (stopping the app is not the
    job's fault). Interruptions are counted, though: a job that keeps taking the process down
    (out of memory, a crashing runtime) goes dead instead of looping forever."""
    with session_factory() as s:
        n = 0
        for job in s.scalars(select(PipelineJob).where(PipelineJob.status == "running")).all():
            interrupted = int((job.payload or {}).get("interrupted", 0)) + 1
            job.payload = {**(job.payload or {}), "interrupted": interrupted}
            job.lease_expires_at = None
            if interrupted >= job.max_attempts:
                job.status = "dead"
                job.last_error = f"the process stopped {interrupted} times while running this job"
            else:
                job.status = "queued"
                job.attempts = max(0, job.attempts - 1)
            n += 1
        s.commit()
    return n


class EmbeddedWorker:
    def __init__(self, ctx: Any, poll_seconds: float = 3.0) -> None:
        self.ctx = ctx
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self.processed: dict[str, int] = {}
        self.current_job: str | None = None
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def start(self) -> None:
        released = release_stale_leases(self.ctx.session_factory)
        if released:
            log.info("re-queued %d jobs interrupted by a previous shutdown", released)
        self._thread = threading.Thread(target=self._loop, name="avd-embedded-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        from avd.workers.runner import claim_batch, execute_job

        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(self.poll_seconds)
                continue
            try:
                ids = claim_batch(self.ctx, limit=1)
            except Exception as exc:  # database unavailable: keep the API up, retry later
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("embedded worker could not claim jobs")
                self._stop.wait(self.poll_seconds * 5)
                continue
            if not ids:
                self._stop.wait(self.poll_seconds)
                continue
            self.current_job = ids[0]
            try:
                out = execute_job(self.ctx, ids[0])
                self.processed[out["outcome"]] = self.processed.get(out["outcome"], 0) + 1
            except Exception as exc:  # execute_job records failures itself; this is a last resort
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("embedded worker job failed")
                self._release(ids[0])
            finally:
                self.current_job = None

    def _release(self, job_id: str) -> None:
        """A job whose execution crashed (e.g. the database failed mid-write) is put back in the
        queue now, rather than staying locked until its lease expires. The attempt still counts,
        so a job that keeps crashing ends up dead instead of looping."""
        try:
            with self.ctx.session_factory() as s:
                job = s.get(PipelineJob, job_id)
                if job is not None and job.status == "running":
                    job.status = "queued" if job.attempts < job.max_attempts else "dead"
                    job.lease_expires_at = None
                    job.last_error = f"worker crash: {self.last_error}"[:4000]
                    s.commit()
        except Exception:  # the database is still failing: the lease will expire on its own
            log.exception("could not release job %s", job_id)

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "paused": self.paused,
            "current_job": self.current_job,
            "processed": dict(self.processed),
            "last_error": self.last_error,
        }
