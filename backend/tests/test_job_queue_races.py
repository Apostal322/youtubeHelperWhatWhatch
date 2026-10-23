"""Job claiming under concurrency, budgets and crashes (audit 2026-10-23)."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from avd.db.models import PipelineJob
from avd.workers import handlers, queue, runner
from avd.workers.budget import BudgetManager
from avd.workers.context import build_context
from avd.workers.embedded import release_stale_leases
from avd.workers.runner import claim_batch, execute_job
from tests.conftest import DB, _make_db


def test_two_sqlite_workers_cannot_claim_the_same_job(tmp_path: Path) -> None:
    db = _make_db(tmp_path, None)  # SQLite has no SKIP LOCKED: this is the case at risk
    with db.sf() as s:
        queue.enqueue(s, "stage_b", "k1")
        s.commit()
    got: dict[str, list[str]] = {}

    def first() -> None:
        with db.sf() as s:
            got["a"] = [j.id for j in queue.claim(s, limit=1, lease_seconds=600)]
            time.sleep(1.0)  # still uncommitted when the second worker reads the queue
            s.commit()

    t = threading.Thread(target=first)
    t.start()
    time.sleep(0.3)
    with db.sf() as s:
        got["b"] = [j.id for j in queue.claim(s, limit=1, lease_seconds=600)]
        s.commit()
    t.join()
    assert got["a"] and not got["b"]  # before: both workers claimed it


def test_jobs_over_budget_do_not_block_free_jobs(db: DB) -> None:
    db.settings.daily_llm_budget_usd = 0.10
    with db.sf() as s:
        for i in range(4):
            queue.enqueue(s, "stage_b", f"b{i}", priority=12.0, estimated_cost_usd=0.50)
        queue.enqueue(s, "stage_a", "a1", priority=10.0)
        s.commit()
    with db.sf() as s:
        got = queue.claim(s, limit=1, lease_seconds=600, budget=BudgetManager(db.settings))
        s.commit()
    assert [j.dedup_key for j in got] == ["a1"]  # before: nothing, on every tick


def test_a_job_that_keeps_crashing_the_process_eventually_dies(db: DB) -> None:
    with db.sf() as s:
        job, _ = queue.enqueue(s, "stage_b", "k", max_attempts=2)
        s.commit()
        jid = job.id
    for _ in range(3):
        with db.sf() as s:
            queue.claim(s, limit=1, lease_seconds=600)  # ... and the process is killed mid-job
            s.commit()
        release_stale_leases(db.sf)  # the next start
    with db.sf() as s:
        j = s.get(PipelineJob, jid)
        assert (
            j is not None and j.status == "dead" and "stopped" in (j.last_error or "")
        )  # before: queued forever


def test_one_restart_does_not_use_up_an_attempt(db: DB) -> None:
    with db.sf() as s:
        job, _ = queue.enqueue(s, "stage_b", "k", max_attempts=3)
        s.commit()
        jid = job.id
    with db.sf() as s:
        queue.claim(s, limit=1, lease_seconds=600)
        s.commit()
    assert release_stale_leases(db.sf) == 1
    with db.sf() as s:
        j = s.get(PipelineJob, jid)
        assert j is not None and j.status == "queued" and j.attempts == 0


def test_a_late_delivery_of_an_expired_claim_does_not_run_the_job_again(
    db: DB, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = build_context(settings=db.settings, session_factory=db.sf)
    calls: list[int] = []

    def slow(ctx: Any, session: Any, job: PipelineJob) -> handlers.HandlerResult:
        calls.append(job.attempts)
        return handlers.HandlerResult({"ok": 1})

    monkeypatch.setitem(runner.HANDLERS, "slow", slow)
    with db.sf() as s:
        job, _ = queue.enqueue(s, "slow", "k1")
        s.commit()
        jid = job.id
    offset = [0.0]
    monkeypatch.setattr(queue, "_now", lambda: datetime.now(UTC) + timedelta(seconds=offset[0]))
    assert claim_batch(ctx) == [jid]  # dispatch 1: the task waits in the broker ...
    offset[0] += ctx.settings.worker_lease_seconds + 1
    assert claim_batch(ctx) == [jid]  # ... the lease expires and dispatch 2 claims it again
    assert execute_job(ctx, jid, attempt=1)["outcome"] == "stale_dispatch"
    assert execute_job(ctx, jid, attempt=2)["outcome"] == "succeeded"
    assert calls == [2]  # before: the handler ran once per delivery


def test_a_job_running_longer_than_its_lease_is_not_claimed_again(
    db: DB, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.settings.worker_lease_seconds = 1
    ctx = build_context(settings=db.settings, session_factory=db.sf)
    started, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def slow(ctx: Any, session: Any, job: PipelineJob) -> handlers.HandlerResult:
        calls.append(job.attempts)
        started.set()
        release.wait(10)
        return handlers.HandlerResult({"ok": 1})

    monkeypatch.setitem(runner.HANDLERS, "slow", slow)
    with db.sf() as s:
        job, _ = queue.enqueue(s, "slow", "long")
        s.commit()
        jid = job.id
    assert claim_batch(ctx) == [jid]
    t = threading.Thread(target=execute_job, args=(ctx, jid, 1))
    t.start()
    assert started.wait(10)
    time.sleep(2.5)  # well past the 1 s lease while the handler is still working (a slow local model)
    second = claim_batch(ctx)
    release.set()
    t.join(10)
    assert second == []  # before: re-claimed, and the new delivery ran the evaluation a second time
    with db.sf() as s:
        j = s.get(PipelineJob, jid)
        assert j is not None and j.status == "succeeded" and calls == [1]


def test_a_dispatch_that_never_reached_a_worker_does_not_use_up_attempts(
    db: DB, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = build_context(settings=db.settings, session_factory=db.sf)
    monkeypatch.setitem(runner.HANDLERS, "slow", lambda c, s, j: handlers.HandlerResult({"ok": 1}))
    with db.sf() as s:
        job, _ = queue.enqueue(s, "slow", "backlog", max_attempts=2)
        s.commit()
        jid = job.id
    offset = [0.0]
    monkeypatch.setattr(queue, "_now", lambda: datetime.now(UTC) + timedelta(seconds=offset[0]))
    for _ in range(4):  # the broker is backed up: each dispatch's lease expires before a worker starts it
        assert claim_batch(ctx) == [jid]
        offset[0] += ctx.settings.worker_lease_seconds + 1
    with db.sf() as s:
        j = s.get(PipelineJob, jid)
        assert j is not None and j.status == "running"  # before: dead, though it never ran
        attempt = j.attempts
    assert execute_job(ctx, jid, attempt)["outcome"] == "succeeded"  # the latest delivery runs it


def test_a_job_no_worker_ever_starts_eventually_dies(db: DB, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = build_context(settings=db.settings, session_factory=db.sf)
    with db.sf() as s:
        job, _ = queue.enqueue(s, "slow", "nobody")
        s.commit()
        jid = job.id
    offset = [0.0]
    monkeypatch.setattr(queue, "_now", lambda: datetime.now(UTC) + timedelta(seconds=offset[0]))
    for _ in range(queue.MAX_UNDELIVERED + 2):
        claim_batch(ctx)
        offset[0] += ctx.settings.worker_lease_seconds + 1
    with db.sf() as s:
        j = s.get(PipelineJob, jid)
        assert j is not None and j.status == "dead" and "no worker started" in (j.last_error or "")


def test_a_worker_that_died_mid_job_still_uses_up_its_attempts(
    db: DB, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = build_context(settings=db.settings, session_factory=db.sf)
    with db.sf() as s:
        job, _ = queue.enqueue(s, "slow", "crashy", max_attempts=2)
        s.commit()
        jid = job.id
    offset = [0.0]
    monkeypatch.setattr(queue, "_now", lambda: datetime.now(UTC) + timedelta(seconds=offset[0]))
    for _ in range(3):
        if claim_batch(ctx):
            with db.sf() as s:  # the worker starts the attempt, then is lost before finishing it
                j = s.get(PipelineJob, jid)
                assert j is not None
                j.payload = {**(j.payload or {}), "started_attempt": j.attempts}
                s.commit()
        offset[0] += ctx.settings.worker_lease_seconds + 1
    with db.sf() as s:
        j = s.get(PipelineJob, jid)
        assert j is not None and j.status == "dead" and "final attempt" in (j.last_error or "")
