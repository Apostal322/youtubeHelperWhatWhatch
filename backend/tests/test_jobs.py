from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update

from avd.content.models import SourceType
from avd.db.models import ContentSource, CostLedger, Evaluation, PipelineJob, Video, VideoStats
from avd.evaluation import schema as schema_mod
from avd.ingest import import_licensed_dataset
from avd.services import seed_demo, submit_user_transcript
from avd.workers import queue
from avd.workers.budget import BudgetManager
from avd.workers.context import build_context
from avd.workers.prioritizer import CoverageCounts, priority_for
from avd.workers.runner import execute_job, run_until_idle
from tests.conftest import DB, ScriptedProvider


def test_enqueue_deduplicates(db: DB) -> None:
    with db.sf() as s:
        j1, c1 = queue.enqueue(s, "stage_a", "k1", priority=1)
        j2, c2 = queue.enqueue(s, "stage_a", "k1", priority=5)
        s.commit()
        assert c1 and not c2 and j1.id == j2.id
        assert j2.priority == 5  # a duplicate may raise the priority of a queued job
        assert s.scalar(select(func.count()).select_from(PipelineJob)) == 1


def test_claim_orders_by_priority_and_sets_lease(db: DB) -> None:
    with db.sf() as s:
        for i, p in enumerate([1.0, 9.0, 5.0]):
            queue.enqueue(s, "stage_a", f"k{i}", priority=p)
        s.commit()
        jobs = queue.claim(s, limit=2, lease_seconds=60)
        assert [j.priority for j in jobs] == [9.0, 5.0]
        assert all(j.status == "running" and j.attempts == 1 and j.lease_expires_at for j in jobs)


def test_retry_backoff_then_dead(db: DB) -> None:
    with db.sf() as s:
        job, _ = queue.enqueue(s, "stage_a", "k", max_attempts=2)
        s.commit()
        queue.claim(s, limit=1, lease_seconds=60)
        queue.fail(s, job, "boom", retryable=True, base_seconds=10)
        assert job.status == "queued" and job.next_attempt_at > datetime.now(UTC) - timedelta(seconds=1)
        job.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
        s.commit()
        queue.claim(s, limit=1, lease_seconds=60)
        queue.fail(s, job, "boom again", retryable=True, base_seconds=10)
        assert job.status == "dead"


def test_non_retryable_failure_goes_dead_immediately(db: DB) -> None:
    with db.sf() as s:
        job, _ = queue.enqueue(s, "stage_a", "k")
        s.commit()
        queue.claim(s, limit=1, lease_seconds=60)
        queue.fail(s, job, "bad input", retryable=False, base_seconds=10)
        assert job.status == "dead"


def test_expired_lease_is_reclaimed(db: DB) -> None:
    with db.sf() as s:
        job, _ = queue.enqueue(s, "stage_a", "k")
        s.commit()
        queue.claim(s, limit=1, lease_seconds=60)
        s.commit()
        # Simulate a worker crash: the lease expires without completion.
        s.execute(update(PipelineJob).values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=5)))
        s.commit()
        again = queue.claim(s, limit=1, lease_seconds=60)
        assert [j.id for j in again] == [job.id] and again[0].attempts == 2


def test_budget_defers_expensive_jobs(db: DB) -> None:
    settings = db.settings.model_copy(update={"daily_llm_budget_usd": 1.0})
    with db.sf() as s:
        s.add(CostLedger(provider="anthropic", model="m", operation="x", cost_usd=0.95))
        queue.enqueue(s, "stage_b", "expensive", estimated_cost_usd=0.2, priority=9)
        queue.enqueue(s, "stage_a", "cheap", priority=1)
        s.commit()
        claimed = queue.claim(s, limit=5, lease_seconds=60, budget=BudgetManager(settings))
        assert [j.dedup_key for j in claimed] == ["cheap"]
        exp = s.scalar(select(PipelineJob).where(PipelineJob.dedup_key == "expensive"))
        assert exp is not None and exp.status == "queued" and "budget" in (exp.last_error or "")


def test_defer_does_not_consume_attempts(db: DB) -> None:
    with db.sf() as s:
        job, _ = queue.enqueue(s, "discover", "d")
        s.commit()
        queue.claim(s, limit=1, lease_seconds=60)
        queue.defer(s, job, datetime.now(UTC) + timedelta(hours=1), "quota")
        assert job.status == "queued" and job.attempts == 0


def test_end_to_end_demo_processing_and_incremental_skip(db: DB) -> None:
    with db.sf() as s:
        n = seed_demo(s, db.settings)
        s.commit()
    ctx = build_context(settings=db.settings, session_factory=db.sf)
    out = run_until_idle(ctx)
    assert out == {"succeeded": 2 * n}
    with db.sf() as s:
        evals = s.scalars(select(Evaluation)).all()
        assert len(evals) == n and all(e.is_synthetic and e.mode == "synthetic_demo" for e in evals)
        assert all(
            v.status == "evaluated" and v.embedding_input == "transcript" for v in s.scalars(select(Video))
        )
        # Incremental processing: re-queuing identical inputs creates no new evaluations.
        for v in s.scalars(select(Video)).all():
            queue.enqueue(s, "stage_b", f"rerun:{v.id}", video_id=v.id)
        s.commit()
    assert run_until_idle(ctx) == {"skipped": n}
    with db.sf() as s:
        assert s.scalar(select(func.count()).select_from(Evaluation)) == n


def test_framework_version_bump_triggers_reevaluation(db: DB, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    ctx = build_context(settings=db.settings, session_factory=db.sf)
    run_until_idle(ctx)
    new_version = schema_mod.FRAMEWORK_VERSION + "-next"
    for mod in (
        "avd.evaluation.schema",
        "avd.evaluation.inputs",
        "avd.evaluation.service",
        "avd.workers.handlers",
        "avd.services",
    ):
        monkeypatch.setattr(f"{mod}.FRAMEWORK_VERSION", new_version, raising=False)
    with db.sf() as s:
        queue.enqueue(s, "reevaluate_stale", "stale-1")
        s.commit()
    run_until_idle(ctx)
    with db.sf() as s:
        current = s.scalars(select(Evaluation).where(Evaluation.is_current.is_(True))).all()
        assert current and all(e.framework_version == new_version for e in current)
        assert s.scalar(
            select(func.count()).select_from(Evaluation).where(Evaluation.is_current.is_(False))
        ) == len(current)


def test_user_transcript_submission_creates_source_and_reevaluates(db: DB) -> None:
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    ctx = build_context(settings=db.settings, session_factory=db.sf)
    run_until_idle(ctx)
    from avd.db.models import User

    with db.sf() as s:
        user = User(token_hash="h")
        s.add(user)
        s.flush()
        text = " ".join(["Hash tables map keys to buckets because hashing spreads keys evenly."] * 12)
        row, created = submit_user_transcript(
            s,
            user_id=user.id,
            video_id="syn_tech_001",
            content=text,
            fmt="text",
            language="en",
            attestation="I am the creator of this video and own the transcript.",
            youtube_enabled=False,
        )
        s.commit()
        assert created and row.source_type == SourceType.USER_PROVIDED_TRANSCRIPT.value
        assert "timing_estimated" in row.authorization_basis
        source_id = row.id
    run_until_idle(ctx)
    with db.sf() as s:
        evs = s.scalars(select(Evaluation).where(Evaluation.video_id == "syn_tech_001")).all()
        assert len(evs) == 2  # new authorized content -> new fingerprint -> re-evaluated, history kept
        current = [e for e in evs if e.is_current]
        assert len(current) == 1 and current[0].content_source_ids == [source_id]


def test_licensed_dataset_import_rejects_unlicensed_rows(db: DB, tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "ds.jsonl"
    seg = [{"start": 0, "end": 5, "text": "Licensed lecture text about thermodynamics and entropy."}]
    rows = [
        {
            "video_id": "lic_000001",
            "license": "CC-BY-4.0",
            "authorization_basis": "Dataset X, CC-BY",
            "segments": seg,
        },
        {"video_id": "lic_000002", "segments": seg},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with db.sf() as s:
        stats, ids = import_licensed_dataset(s, p)
        s.commit()
        assert stats == {"imported": 1, "duplicates": 0, "rejected": 1} and ids == ["lic_000001"]
        assert s.scalar(select(ContentSource.license)) == "CC-BY-4.0"


def test_llm_transient_failure_is_retried_not_persisted(db: DB) -> None:
    from avd.llm.base import LLMTransientError

    provider = ScriptedProvider(lambda req, i: LLMTransientError("overloaded"))
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    ctx = build_context(settings=db.settings, session_factory=db.sf, llm=provider)  # type: ignore[arg-type]
    run_until_idle(ctx, job_types=["stage_a"])
    with db.sf() as s:
        jobs = queue.claim(s, limit=1, lease_seconds=60, job_types=["stage_b"])
        s.commit()
    out = execute_job(ctx, jobs[0].id)
    assert out["outcome"] == "retry"
    with db.sf() as s:
        assert s.scalar(select(func.count()).select_from(Evaluation)) == 0


class TestPrioritizer:
    def test_priority_ignores_popularity_and_balances_strata(self) -> None:
        counts = CoverageCounts(
            by_genre={"science": 5}, by_size={"micro": 5, "huge": 5}, by_language={"en": 5}
        )
        kw = {
            "genre": "science",
            "language": "en",
            "has_transcript": True,
            "has_audio": False,
            "counts": counts,
        }
        micro, _ = priority_for("stage_b", size_bucket="micro", **kw)  # type: ignore[arg-type]
        huge, _ = priority_for("stage_b", size_bucket="huge", **kw)  # type: ignore[arg-type]
        assert micro == huge  # equal coverage -> equal treatment regardless of channel size

    def test_under_covered_strata_rise(self) -> None:
        counts = CoverageCounts(
            by_genre={"science": 50, "art": 0}, by_size={"micro": 0}, by_language={"en": 50}
        )
        kw = {
            "size_bucket": "micro",
            "language": "en",
            "has_transcript": True,
            "has_audio": False,
            "counts": counts,
        }
        art, _ = priority_for("stage_b", genre="art", **kw)  # type: ignore[arg-type]
        sci, _ = priority_for("stage_b", genre="science", **kw)  # type: ignore[arg-type]
        assert art > sci

    def test_video_priority_unchanged_by_view_counts(self, db: DB) -> None:
        from avd.workers.prioritizer import priority_for_video

        with db.sf() as s:
            seed_demo(s, db.settings)
            s.commit()
            v = s.get(Video, "syn_sci_001")
            assert v is not None
            before = priority_for_video(s, v, "stage_b")
            s.execute(update(VideoStats).values(view_count=10**9, like_count=10**8))
            assert priority_for_video(s, v, "stage_b") == before
