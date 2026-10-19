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

from sqlalchemy import update

from avd.db.models import PipelineJob

log = logging.getLogger(__name__)


def release_stale_leases(session_factory: Any) -> int:
    """Single-process local mode only: jobs left ``running`` by a previous process cannot still be
    running, so they are re-queued without consuming an attempt."""
    with session_factory() as s:
        n = s.execute(
            update(PipelineJob)
            .where(PipelineJob.status == "running")
            .values(status="queued", lease_expires_at=None, attempts=PipelineJob.attempts - 1)
        ).rowcount
        s.commit()
    return int(n or 0)


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
            finally:
                self.current_job = None

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "paused": self.paused,
            "current_job": self.current_job,
            "processed": dict(self.processed),
            "last_error": self.last_error,
        }
