"""Job outcomes when evaluation fails, and recovery afterwards (audit 2026-10-23).

Model calls here come from a scripted test double; nothing is real inference.
"""

from __future__ import annotations

from sqlalchemy import func, select

from avd import datactl
from avd.db.models import Evaluation, PipelineJob, Video
from avd.llm.base import LLMConfigurationError
from avd.services import enqueue_stage_a, seed_demo
from avd.workers.context import build_context
from avd.workers.runner import run_until_idle
from tests.conftest import DB, ScriptedProvider

VID = "syn_tech_001"


def _seeded(db: DB) -> None:
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    run_until_idle(build_context(settings=db.settings, session_factory=db.sf))


def test_unavailable_model_keeps_the_previous_evaluation_and_fails_the_job(db: DB) -> None:
    _seeded(db)
    with db.sf() as s:
        good = s.scalar(select(Evaluation).where(Evaluation.video_id == VID, Evaluation.is_current.is_(True)))
        assert good is not None and good.status != "failed"
        good_id = good.id
    missing = ScriptedProvider(lambda r, i: LLMConfigurationError("model 'qwen3.5:4b' is not installed"))
    with db.sf() as s:
        enqueue_stage_a(s, VID, force=True)
        s.commit()
    run_until_idle(build_context(settings=db.settings, session_factory=db.sf, llm=missing))  # type: ignore[arg-type]
    with db.sf() as s:
        current = s.scalars(
            select(Evaluation).where(Evaluation.video_id == VID, Evaluation.is_current.is_(True))
        ).all()
        # Before: the good evaluation was demoted and a failed one became current; the job "succeeded".
        assert [e.id for e in current] == [good_id]
        assert s.get(Video, VID).status == "evaluated"  # type: ignore[union-attr]
        job = s.scalar(
            select(PipelineJob)
            .where(PipelineJob.video_id == VID, PipelineJob.job_type == "stage_b")
            .order_by(PipelineJob.created_at.desc())
        )
        assert job is not None and job.status == "dead"
        assert "not installed" in (job.last_error or "")


def test_a_failed_first_evaluation_is_kept_for_diagnosis_and_the_job_fails(db: DB) -> None:
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    missing = ScriptedProvider(lambda r, i: LLMConfigurationError("model 'qwen3.5:4b' is not installed"))
    run_until_idle(build_context(settings=db.settings, session_factory=db.sf, llm=missing))  # type: ignore[arg-type]
    with db.sf() as s:
        ev = s.scalar(select(Evaluation).where(Evaluation.video_id == VID, Evaluation.is_current.is_(True)))
        assert ev is not None and ev.status == "failed"
        assert s.get(Video, VID).status == "evaluation_failed"  # type: ignore[union-attr]
        statuses = s.scalars(
            select(PipelineJob.status).where(PipelineJob.job_type == "stage_b", PipelineJob.video_id == VID)
        ).all()
        assert statuses == ["dead"]  # not "succeeded"


def test_cleared_evaluations_are_recomputed_when_stage_a_is_forced(db: DB) -> None:
    _seeded(db)
    with db.sf() as s:
        datactl.clear(s, evaluations=True)
        for vid in s.scalars(select(Video.id)).all():
            enqueue_stage_a(s, vid, force=True)
        s.commit()
    run_until_idle(build_context(settings=db.settings, session_factory=db.sf))
    with db.sf() as s:
        # Before: the finished Stage B jobs were never re-queued, so nothing was re-evaluated.
        assert s.scalar(select(func.count()).select_from(Evaluation)) == s.scalar(
            select(func.count()).select_from(Video)
        )


def test_a_model_that_leaves_most_dimensions_out_does_not_replace_a_full_evaluation(db: DB) -> None:
    from avd.content.store import load_bundle
    from avd.db.models import DimensionScoreRow
    from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS

    _seeded(db)
    with db.sf() as s:
        good = s.scalar(select(Evaluation).where(Evaluation.video_id == VID, Evaluation.is_current.is_(True)))
        assert good is not None and good.overall_score is not None
        good_id = good.id
        quote = load_bundle(s, VID).transcript.segments[0].text[:60]  # type: ignore[union-attr]

    def two_dimensions(req: object, i: int) -> dict[str, object]:
        return {
            "content_summary": "",
            "detected_issues": [],
            "judgments": [
                {
                    "dimension": d.value,
                    "assessable": True,
                    "score": 4,
                    "confidence": 0.8,
                    "justification": "ok",
                    "evidence": [{"segment_index": 0, "quote": quote}],
                    "counter_evidence": [],
                    "unassessable_reason": "",
                }
                for d in TEXT_DIMENSIONS[:2]  # a small local model that skipped the rest
            ],
        }

    with db.sf() as s:
        enqueue_stage_a(s, VID, force=True)
        s.commit()
    run_until_idle(
        build_context(settings=db.settings, session_factory=db.sf, llm=ScriptedProvider(two_dimensions))
    )  # type: ignore[arg-type]
    with db.sf() as s:
        current = s.scalars(
            select(Evaluation).where(Evaluation.video_id == VID, Evaluation.is_current.is_(True))
        ).all()
        assert [e.id for e in current] == [good_id]  # before: the partial evaluation replaced it
        job = s.scalar(
            select(PipelineJob)
            .where(PipelineJob.video_id == VID, PipelineJob.job_type == "stage_b")
            .order_by(PipelineJob.created_at.desc())
        )
        assert job is not None and job.status != "succeeded" and "incomplete" in (job.last_error or "")
        assert s.scalar(
            select(func.count())
            .select_from(DimensionScoreRow)
            .where(DimensionScoreRow.evaluation_id == good_id)
        )
