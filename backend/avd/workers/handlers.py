"""Job handlers: discovery -> metadata ingestion -> Stage A -> Stage B -> (Stage C).

Stage A (lightweight): genre from metadata, topic keywords, metadata embedding, content
    availability assessment across content providers. No LLM calls.
Stage B (deep): acquire authorized content, refine genre with the transcript, run the evaluation
    pipeline, persist, re-embed with the transcript, decide on escalation.
Stage C (selective, expensive): stronger model + self-consistency + retrieval fact checking, only
    when it is expected to improve reliability and the budget allows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.content.store import active_sources, load_bundle
from avd.db.models import Channel, DiscoveryQuery, Evaluation, PipelineJob, Video
from avd.embeddings import documents as emb_docs
from avd.evaluation.genre import apply_override
from avd.evaluation.inputs import build_evaluation_input
from avd.evaluation.schema import FRAMEWORK_VERSION, EvaluationResult, Genre
from avd.evaluation.service import is_up_to_date, persist_evaluation
from avd.evaluation.topics import extract_topics
from avd.ingest import apply_channel_metadata, apply_video_metadata, register_discovered
from avd.llm.pricing import estimate_cost
from avd.logging import get_logger
from avd.workers.context import WorkerContext
from avd.workers.prioritizer import coverage_counts, priority_for_video
from avd.workers.queue import enqueue

log = get_logger(__name__)

FACT_CRITICAL = {
    Genre.SCIENCE.value,
    Genre.EDUCATION.value,
    Genre.LECTURE.value,
    Genre.PRODUCT_REVIEW.value,
    Genre.NEWS_COMMENTARY.value,
    Genre.DOCUMENTARY.value,
}
CHANNEL_STATS_TTL = timedelta(days=7)


@dataclass
class HandlerResult:
    result: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    cost_usd: float = 0.0


Handler = Callable[[WorkerContext, Session, PipelineJob], HandlerResult]


def _content_key(session: Session, video_id: str) -> str:
    hashes = sorted(r.content_hash for r in active_sources(session, video_id))
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:12] if hashes else "pending"


# ----------------------------------------------------------------------------- discovery


def handle_discover(ctx: WorkerContext, session: Session, job: PipelineJob) -> HandlerResult:
    q = session.get(DiscoveryQuery, job.payload["query_id"])
    if q is None or not q.enabled:
        return HandlerResult({"reason": "query missing or disabled"}, skipped=True)
    yt = ctx.youtube(job.id)
    ids, next_token = yt.search(q.query, language=q.language, order=q.order, page_token=q.next_page_token)
    new = [vid for vid in ids if register_discovered(session, vid, None, f"search:{q.id}")]
    q.last_run_at = datetime.now(UTC)
    q.next_page_token = next_token  # None -> next run restarts from the newest uploads
    q.total_results += len(new)
    if new:
        key = hashlib.sha256(",".join(sorted(new)).encode()).hexdigest()[:16]
        enqueue(session, "ingest_metadata", f"ingest:{key}", payload={"video_ids": new}, priority=8.0)
    return HandlerResult({"found": len(ids), "new": len(new)})


def handle_ingest_metadata(ctx: WorkerContext, session: Session, job: PipelineJob) -> HandlerResult:
    ids: list[str] = job.payload["video_ids"]
    yt = ctx.youtube(job.id)
    metas = yt.videos(ids)  # batched: 50 ids per quota unit
    now = datetime.now(UTC)
    need_stats = []
    for cid in {m.channel_id for m in metas}:
        ch = session.get(Channel, cid)
        fetched = ch.stats_fetched_at if ch else None
        if fetched is not None and fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=UTC)
        if fetched is None or now - fetched > CHANNEL_STATS_TTL:
            need_stats.append(cid)
    channels = yt.channels(need_stats) if need_stats else []
    for c in channels:
        apply_channel_metadata(session, c)
    for m in metas:
        apply_video_metadata(session, m, discovery_source="youtube_data_api")
        enqueue(
            session,
            "stage_a",
            f"stage_a:{m.video_id}:{FRAMEWORK_VERSION}",
            video_id=m.video_id,
            priority=10.0,
        )
    returned = {m.video_id for m in metas}
    for vid in set(ids) - returned:
        v = session.get(Video, vid)
        if v is not None:
            v.status = "unavailable"  # private, deleted or region-blocked
    return HandlerResult({"ingested": len(metas), "unavailable": len(set(ids) - returned)})


# ----------------------------------------------------------------------------- stage A


def handle_stage_a(ctx: WorkerContext, session: Session, job: PipelineJob) -> HandlerResult:
    video = session.get(Video, job.video_id)
    if video is None:
        return HandlerResult({"reason": "video not found"}, skipped=True)
    g = apply_override(
        ctx.classifier.classify(
            video.title, video.description or "", list(video.tags or []), video.youtube_category_id
        ),
        video.genre_override,
        video.genre_override_by,
    )
    # Stage A never overwrites a transcript-informed (Stage B) classification.
    if video.genre_method is None or video.genre_method == g.method or video.genre_override:
        video.genre, video.genre_confidence, video.genre_method = g.genre.value, g.confidence, g.method
        video.genre_distribution, video.genre_status = g.distribution, g.status.value
    video.topics = extract_topics(video.title, video.description or "", list(video.tags or []))
    if video.embedding is None or video.embedding_model != ctx.embedder.model_id:
        emb_docs.apply(video, emb_docs.embed_video(ctx.embedder, video, None))

    modalities: set[str] = set()
    provider_errors = []
    for p in ctx.providers:
        try:
            modalities |= {m.value for m in p.check(session, video)}
        except Exception as exc:
            provider_errors.append(f"{p.name}: {exc}")
            log.warning("content provider check failed", extra={"provider": p.name, "video_id": video.id})
    video.available_modalities = sorted(modalities)
    if "transcript" not in modalities:
        if video.status not in ("evaluated",):
            video.status = "content_unavailable"
        return HandlerResult({"modalities": sorted(modalities), "provider_errors": provider_errors})
    if video.status not in ("evaluated",):
        video.status = "content_available"

    priority, parts = priority_for_video(session, video, "stage_b")
    est = 0.0
    if ctx.llm is not None:
        chars = sum(len(str(r.transcript or "")) for r in active_sources(session, video.id)[:1])
        est = estimate_cost(ctx.settings.llm_model_stage_b, chars + 8000)
    _, created = enqueue(
        session,
        "stage_b",
        f"stage_b:{video.id}:{FRAMEWORK_VERSION}:{_content_key(session, video.id)}",
        video_id=video.id,
        priority=priority,
        estimated_cost_usd=est,
        payload={"priority_parts": parts},
        max_attempts=ctx.settings.job_max_attempts,
        # e.g. after a manual genre correction: re-run Stage B even though the content is unchanged
        requeue_terminal=bool((job.payload or {}).get("force_stage_b")),
    )
    return HandlerResult(
        {"modalities": sorted(modalities), "stage_b_enqueued": created, "priority": priority}
    )


# ----------------------------------------------------------------------------- stage B / C


def should_escalate(ctx: WorkerContext, session: Session, result: EvaluationResult) -> tuple[bool, str]:
    if ctx.llm is None:
        return False, "demo mode: no Stage-C model configured"
    if result.status in ("failed", "abstained"):
        return False, f"stage B {result.status}; escalation would not have usable inputs"
    frac = ctx.budget.remaining_fraction(session)
    if frac < ctx.settings.stage_c_min_remaining_budget_fraction:
        return False, f"budget: only {frac:.0%} of today's LLM budget remains"
    conf = result.aggregation.overall_confidence
    if conf is None or conf < 0.5:
        return True, f"low overall confidence ({conf})"
    if result.genre.genre.value in FACT_CRITICAL and ctx.retriever is not None:
        return True, "fact-critical genre and a reference corpus is configured"
    return False, "stage B confidence adequate"


def _evaluate(ctx: WorkerContext, session: Session, job: PipelineJob, stage: str) -> HandlerResult:
    video = session.get(Video, job.video_id)
    if video is None:
        return HandlerResult({"reason": "video not found"}, skipped=True)
    for p in ctx.providers:
        try:
            p.acquire(session, video)
        except Exception as exc:
            log.warning(
                "content acquisition failed",
                extra={"provider": p.name, "video_id": video.id, "error": str(exc)},
            )
    bundle = load_bundle(session, video.id)
    if bundle.transcript is None:
        video.status = "content_unavailable"
        return HandlerResult({"reason": "no authorized transcript"}, skipped=True)
    genre = apply_override(
        ctx.classifier.classify(
            video.title,
            video.description or "",
            list(video.tags or []),
            video.youtube_category_id,
            transcript_text=bundle.transcript.text,
        ),
        video.genre_override,
        video.genre_override_by,
    )
    inp = build_evaluation_input(video, bundle, genre)
    if is_up_to_date(session, video.id, inp.fingerprint(), stage, ctx.pipeline(stage).config_hash()):
        return HandlerResult({"reason": "up to date (same inputs and framework version)"}, skipped=True)
    # Release write locks before slow external calls (LLM providers write to the cache table).
    session.commit()

    result = ctx.pipeline(stage).run(inp)
    ev = persist_evaluation(session, result, job.id)
    if emb_docs.is_stale(ctx.embedder, video, bundle.transcript):
        emb_docs.apply(video, emb_docs.embed_video(ctx.embedder, video, bundle.transcript))

    out: dict[str, Any] = {
        "evaluation_id": ev.id,
        "status": result.status,
        "overall": result.aggregation.overall,
        "mode": result.mode.value,
    }
    if stage == "B":
        escalate, reason = should_escalate(ctx, session, result)
        out["stage_c"] = {"escalate": escalate, "reason": reason}
        if escalate:
            priority, _ = priority_for_video(session, video, "stage_c")
            enqueue(
                session,
                "stage_c",
                f"stage_c:{video.id}:{FRAMEWORK_VERSION}:{result.input_fingerprint[:12]}",
                video_id=video.id,
                priority=priority,
                estimated_cost_usd=estimate_cost(
                    ctx.settings.llm_model_stage_c, len(bundle.transcript.text) + 8000
                )
                * ctx.settings.llm_self_consistency_samples_stage_c,
                payload={"reason": reason},
            )
    return HandlerResult(out, cost_usd=result.cost_usd)


def handle_stage_b(ctx: WorkerContext, session: Session, job: PipelineJob) -> HandlerResult:
    return _evaluate(ctx, session, job, "B")


def handle_stage_c(ctx: WorkerContext, session: Session, job: PipelineJob) -> HandlerResult:
    if ctx.llm is None:
        return HandlerResult({"reason": "no LLM configured"}, skipped=True)
    return _evaluate(ctx, session, job, "C")


def handle_reevaluate_stale(ctx: WorkerContext, session: Session, job: PipelineJob) -> HandlerResult:
    """Re-queue Stage A for videos whose current evaluation predates FRAMEWORK_VERSION."""
    stale = session.scalars(
        select(Evaluation.video_id).where(
            Evaluation.is_current.is_(True), Evaluation.framework_version != FRAMEWORK_VERSION
        )
    ).all()
    counts = coverage_counts(session)
    n = 0
    for vid in stale:
        video = session.get(Video, vid)
        if video is None:
            continue
        prio, _ = priority_for_video(session, video, "stage_a", counts)
        _, created = enqueue(
            session, "stage_a", f"stage_a:{vid}:{FRAMEWORK_VERSION}", video_id=vid, priority=prio
        )
        n += int(created)
    return HandlerResult({"requeued": n})


HANDLERS: dict[str, Handler] = {
    "discover": handle_discover,
    "ingest_metadata": handle_ingest_metadata,
    "stage_a": handle_stage_a,
    "stage_b": handle_stage_b,
    "stage_c": handle_stage_c,
    "reevaluate_stale": handle_reevaluate_stale,
}
