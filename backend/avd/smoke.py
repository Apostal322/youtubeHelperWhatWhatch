"""Explicitly-authorized live smoke test of the real LLM evaluation path.

``avd llm-smoke`` evaluates ONE transcript through the production wrapper chain (logging, cache,
spend guard, rate limit, retries) on an isolated temporary database and prints a JSON report.

Safety:
* refuses (exit 2) when no provider/credentials are configured, naming the blocked step;
* estimates the cost first and refuses if it exceeds ``AVD_SMOKE_TEST_MAX_COST_USD`` (default
  $0.25) unless ``--max-cost`` raises it explicitly;
* the per-job and daily budgets of the temporary run are set to that cap, so the spend guard
  stops the run before exceeding it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from avd.config import LLMProviderName, Settings
from avd.content.models import SourceType, Transcript
from avd.content.parsing import parse_timed_text
from avd.content.store import add_transcript_source
from avd.db.models import Base, CostLedger, Evaluation
from avd.db.session import make_engine
from avd.ingest import load_fixture_files, seed_fixture_video, upsert_channel
from avd.llm.pricing import estimate_cost
from avd.observability import job_context


class SmokeBlocked(RuntimeError):
    pass


def credentials_status(settings: Settings) -> tuple[bool, str]:
    if settings.llm_provider == LLMProviderName.HEURISTIC:
        return False, "AVD_LLM_PROVIDER=heuristic (demo mode); set it to 'anthropic' or 'openai_compatible'"
    if settings.llm_provider == LLMProviderName.ANTHROPIC:
        if (
            settings.anthropic_api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ):
            return True, "anthropic credentials present"
        return False, "no Anthropic credentials (AVD_ANTHROPIC_API_KEY / ANTHROPIC_API_KEY unset)"
    if settings.openai_base_url:
        return True, f"OpenAI-compatible endpoint {settings.openai_base_url}"
    return False, "no OpenAI-compatible endpoint configured"


def run_smoke(
    settings: Settings,
    *,
    fixture_video: str | None = "syn_sci_001",
    transcript_file: Path | None = None,
    title: str = "Smoke test transcript",
    max_cost: float | None = None,
) -> dict[str, Any]:
    ok, why = credentials_status(settings)
    if not ok:
        raise SmokeBlocked(f"live LLM evaluation blocked: {why}")
    cap = max_cost if max_cost is not None else settings.smoke_test_max_cost_usd
    tmp = Path(tempfile.mkdtemp(prefix="avd-smoke-"))
    run_settings = settings.model_copy(
        update={
            "database_url": f"sqlite:///{tmp / 'smoke.db'}",
            "media_cache_dir": tmp / "media",
            "redis_url": None,
            "daily_llm_budget_usd": cap,
            "max_job_cost_usd": cap,
        }
    )
    engine = make_engine(run_settings.database_url)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)

    with sf() as s:
        if transcript_file is not None:
            raw = transcript_file.read_text(encoding="utf-8")
            t = (
                Transcript.model_validate_json(raw)
                if transcript_file.suffix == ".json"
                else parse_timed_text(raw)
            )
            from avd.db.models import Video

            upsert_channel(s, "smoke_channel", "")
            video_id = "smoke_video_1"
            s.add(
                Video(
                    id=video_id,
                    channel_id="smoke_channel",
                    title=title,
                    discovery_source="smoke",
                    status="metadata_ready",
                )
            )
            s.flush()
            add_transcript_source(
                s,
                video_id=video_id,
                transcript=t,
                source_type=SourceType.USER_PROVIDED_TRANSCRIPT,
                authorization_basis=f"Operator-supplied file {transcript_file.name} for an authorized smoke test",
            )
            chars = len(t.text)
        else:
            fx = next(
                f for f in load_fixture_files(run_settings.fixtures_dir) if f["video_id"] == fixture_video
            )
            video = seed_fixture_video(s, fx, run_settings.media_cache_dir)
            video_id = video.id
            chars = sum(len(seg["text"]) for seg in fx["content"]["transcript"]["segments"])
        s.commit()

    est = estimate_cost(run_settings.llm_model_stage_b, chars * 2 + 12000, 4000)
    if est > cap:
        raise SmokeBlocked(
            f"estimated cost ${est:.4f} exceeds the smoke-test cap ${cap:.2f}; raise --max-cost to authorize"
        )

    from avd.workers.context import build_context
    from avd.workers.handlers import handle_stage_a, handle_stage_b
    from avd.workers.queue import claim, enqueue

    ctx = build_context(settings=run_settings, session_factory=sf)
    started = time.monotonic()
    with sf() as s:
        for job_type, handler in (("stage_a", handle_stage_a), ("stage_b", handle_stage_b)):
            job, _ = enqueue(s, job_type, f"smoke:{job_type}:{video_id}", video_id=video_id)
            s.commit()
            claim(s, limit=1, lease_seconds=600, job_types=[job_type])
            with job_context(job.id):
                handler(ctx, s, job)
            s.commit()
        ev = (
            s.query(Evaluation)
            .filter(Evaluation.video_id == video_id, Evaluation.is_current.is_(True))
            .one_or_none()
        )
        calls = s.query(CostLedger).filter(CostLedger.provider != "youtube_data_api").all()
        out: dict[str, Any] = {
            "provider": run_settings.llm_provider.value,
            "requested_model": run_settings.llm_model_stage_b,
            "estimated_cost_usd": round(est, 5),
            "wall_time_s": round(time.monotonic() - started, 2),
            "model_calls": [
                {
                    "status": c.status,
                    "served_model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": c.cost_usd,
                    "latency_ms": c.latency_ms,
                    "attempts": c.attempts,
                    "error_type": c.error_type,
                }
                for c in calls
            ],
            "total_cost_usd": round(sum(c.cost_usd for c in calls), 6),
        }
        if ev is None:
            out["evaluation"] = None
        else:
            out["evaluation"] = {
                "status": ev.status,
                "mode": ev.mode,
                "overall": ev.overall_score,
                "validation": ev.validation,
                "needs_review": ev.needs_review,
                "review_reasons": ev.review_reasons,
                "dimensions": [
                    {
                        "dimension": d.dimension,
                        "status": d.status,
                        "score": d.score,
                        "evidence_verified": sum(1 for e in d.evidence if e.verified),
                    }
                    for d in ev.dimensions
                ],
            }
    return out


def main(args: Any) -> int:
    from avd.config import get_settings

    try:
        report = run_smoke(
            get_settings(),
            fixture_video=args.fixture,
            transcript_file=Path(args.transcript) if args.transcript else None,
            max_cost=args.max_cost,
        )
    except SmokeBlocked as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0
