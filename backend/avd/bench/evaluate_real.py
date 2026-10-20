"""Reproducible, budget-capped evaluation of authorized real items with a configured LLM.

``avd evaluate-real`` runs the *production* job path (import → Stage A → Stage B, optional
Stage C) on dataset items in an isolated, persistent run database, so a run is:

* **Pre-flighted.** Before any paid call: split guard, rights check, prompt/rubric/model
  validation, credentials check, and a projected cost from the measured call plan (a dry-run
  provider records every request the pipeline would make). The run refuses to start when the
  *expected* cost exceeds the budget.
* **Hard-capped.** A run-level guard reserves each call's worst-case cost before it is sent, so
  the ceiling holds with concurrency and retries, and across resumes (spend is read back from the
  run's cost ledger).
* **Resumable.** Items whose current evaluation already matches the run's pipeline config are
  skipped; completed model calls are served from the run's response cache, so an interrupted run
  never pays twice for the same call.
* **Recorded.** Per item: calls (ok / cached / error by type), parse failures, refusals, evidence
  claimed vs verified, ungrounded dimensions, abstentions, tokens, cost and latency; per run: the
  model, prompt key, rubric version, config hash, git commit and settings.

A dry run (``--dry-run``) makes no API call and needs no credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from avd.bench.harness import _git_sha
from avd.bench.real import GATES_PATH, LOCK_PATH, SplitGuardError, _git_dirty, load_gates
from avd.bench.throughput import CHARS_PER_TOKEN, OUTPUT_TOKENS, DryRunProvider
from avd.config import REPO_ROOT, LLMProviderName, Settings, get_settings
from avd.content.store import load_bundle
from avd.datasets.fetch import SourceFetcher
from avd.datasets.importer import import_manifest
from avd.datasets.pilot import load_manifest
from avd.datasets.rights import pending_uses
from avd.db.models import Base, CostLedger, Evaluation
from avd.db.session import make_engine
from avd.evaluation.prompts import get_prompt
from avd.llm.base import LLMError, LLMProvider
from avd.llm.pricing import DEFAULT_PRICES, cost_usd, register_local_model
from avd.llm.providers import RunBudget, build_llm_provider
from avd.services import enqueue_stage_a
from avd.workers.context import build_context
from avd.workers.runner import claim_batch, execute_job

RUNS_DIR = REPO_ROOT / "benchmarks" / "runs"
HELD_OUT_NAMES = {"real_holdout"}


class PreflightError(RuntimeError):
    """The run must not start; the message says exactly what is missing."""


@dataclass
class RunConfig:
    manifest: Path
    split: str = "dev"
    run_id: str = ""
    out_dir: Path | None = None
    budget_usd: float | None = None
    dry_run: bool = False
    max_items: int | None = None
    concurrency: int = 1
    timeout_seconds: float = 180.0
    max_retries: int = 2
    model: str | None = None
    prompt: str | None = None
    rubric_version: str | None = None
    stage_c: bool = False
    confirm_test: bool = False
    offline: bool = False
    cache_dir: Path | None = None
    provider: str | None = None  # overrides AVD_LLM_PROVIDER for this run (e.g. "ollama")
    output_token_assumption: dict[str, int] = field(default_factory=lambda: dict(OUTPUT_TOKENS))

    def resolved_out(self) -> Path:
        return self.out_dir or RUNS_DIR / self.run_id


def _is_local(settings: Settings, raw: LLMProvider | None) -> bool:
    return (raw.name if raw is not None else settings.llm_provider) == LLMProviderName.OLLAMA


def _settings(cfg: RunConfig, base: Settings, out: Path) -> Settings:
    prompt = get_prompt(cfg.prompt)
    if cfg.provider and cfg.provider != base.llm_provider:
        # Re-validate (not model_copy) so provider-dependent defaults, e.g. local context sizing,
        # are applied; only explicitly set fields are carried over.
        base = Settings(
            **{k: getattr(base, k) for k in base.model_fields_set if k != "llm_provider"},
            llm_provider=LLMProviderName(cfg.provider),
        )
    local = base.llm_provider == LLMProviderName.OLLAMA
    if local:
        for m in (cfg.model or base.llm_model_stage_b, cfg.model or base.llm_model_stage_c):
            register_local_model(m)
    # The run ceiling governs; the daily/job budgets are set to it so they never bind first. Local
    # calls cost $0, so the daily budget only needs to be positive for Stage C's remaining-fraction
    # check; the run ceiling (0 unless given) still refuses any call that would cost money.
    daily = max(cfg.budget_usd or 0.0, 1.0) if local else (cfg.budget_usd or 0.0)
    return base.model_copy(
        update={
            "database_url": f"sqlite:///{out / 'run.db'}",
            "media_cache_dir": out / "media",
            "redis_url": None,
            "llm_prompt": prompt.key,
            "llm_model_stage_b": cfg.model or base.llm_model_stage_b,
            "llm_model_stage_c": cfg.model or base.llm_model_stage_c,
            "llm_timeout_seconds": cfg.timeout_seconds,
            "llm_max_retries": cfg.max_retries,
            "llm_max_concurrency": max(1, cfg.concurrency),
            "daily_llm_budget_usd": daily,
            "max_job_cost_usd": daily,
            # frac <= 1 always, so 1.01 disables escalation unless Stage C was asked for.
            "stage_c_min_remaining_budget_fraction": 0.3 if cfg.stage_c else 1.01,
        }
    )


def _select(cfg: RunConfig, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    items = sorted((it for it in manifest["items"] if it["split"] == cfg.split), key=lambda it: it["item_id"])
    excluded = []
    keep = []
    for it in items:
        if "internal_analysis" in pending_uses(it["source"]):
            excluded.append(
                {"item_id": it["item_id"], "reason": "internal analysis not cleared (rights review)"}
            )
        else:
            keep.append(it)
    if cfg.max_items is not None:
        keep = keep[: cfg.max_items]
    return keep, excluded


def _guard_split(cfg: RunConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    held_out = cfg.split == "test" or manifest["name"] in HELD_OUT_NAMES
    gates = load_gates(GATES_PATH, LOCK_PATH)
    if held_out:
        if not cfg.confirm_test:
            raise SplitGuardError(
                f"{manifest['name']}@{manifest['version']} split {cfg.split} is held out: "
                "pass --confirm-test only for the single, locked confirmatory run"
            )
        if not gates["lock_valid"]:
            raise SplitGuardError("release gates are not locked (avd gates lock); refusing a held-out run")
    return {"held_out": held_out, "gates_sha256": gates["sha256"], "gates_lock_valid": gates["lock_valid"]}


def _input_for(s: Any, vid: str) -> Any:
    """The same input the Stage B handler builds (classifier + override + bundle)."""
    from avd.db.models import Video
    from avd.evaluation.genre import KeywordGenreClassifier, apply_override
    from avd.evaluation.inputs import build_evaluation_input

    video = s.get(Video, vid)
    bundle = load_bundle(s, vid)
    genre = apply_override(
        KeywordGenreClassifier().classify(
            video.title,
            video.description or "",
            list(video.tags or []),
            video.youtube_category_id,
            transcript_text=bundle.transcript.text if bundle.transcript else None,
        ),
        video.genre_override,
        video.genre_override_by,
    )
    return build_evaluation_input(video, bundle, genre)


def plan_calls(
    cfg: RunConfig, settings: Settings, sf: sessionmaker[Any], video_ids: list[str]
) -> list[dict[str, Any]]:
    """Measured call plan: run the real pipeline with a dry-run provider (no API, no persistence)."""
    from avd.evaluation.pipeline import build_pipeline

    plans = []
    stages = ("B", "C") if cfg.stage_c else ("B",)
    with sf() as s:
        for vid in video_ids:
            entry: dict[str, Any] = {"video_id": vid, "stages": {}}
            for stage in stages:
                dry = DryRunProvider()
                pipe = build_pipeline(
                    stage, settings.model_copy(update={"llm_provider": LLMProviderName.ANTHROPIC}), dry
                )
                pipe.run(_input_for(s, vid))
                if any("output contract" in c.user for c in dry.calls):
                    raise RuntimeError("dry-run answer failed validation; the plan would include repairs")
                expected = worst = 0.0
                tin = 0
                for c in dry.calls:
                    chars = len(c.system) + len(c.user)
                    out_tokens = cfg.output_token_assumption.get(
                        c.schema_name, cfg.output_token_assumption["default"]
                    )
                    expected += cost_usd(c.model, int(chars / CHARS_PER_TOKEN), min(out_tokens, c.max_tokens))
                    worst += cost_usd(c.model, int(chars / RunBudget.CHARS_PER_TOKEN_WORST) + 1, c.max_tokens)
                    tin += int(chars / CHARS_PER_TOKEN)
                entry["stages"][stage] = {
                    "calls": len(dry.calls),
                    "by_kind": dict(Counter(c.schema_name for c in dry.calls)),
                    "est_input_tokens": tin,
                    "expected_usd": round(expected, 4),
                    "worst_case_usd": round(worst, 4),
                    "model": dry.calls[0].model if dry.calls else None,
                }
            plans.append(entry)
    return plans


def _check_live_config(cfg: RunConfig, settings: Settings, raw: LLMProvider | None) -> None:
    model = settings.llm_model_stage_b
    local = _is_local(settings, raw)
    if not local and model not in DEFAULT_PRICES and model not in settings.llm_price_overrides:
        raise PreflightError(
            f"model {model!r} has no known price; set AVD_LLM_PRICE_OVERRIDES so the budget can be enforced"
        )
    prompt = get_prompt(settings.llm_prompt)
    if not prompt.compatible_with(model, local=local):
        raise PreflightError(f"prompt {prompt.key} is not registered as compatible with {model}")
    if cfg.rubric_version and cfg.rubric_version != prompt.rubric_version:
        raise PreflightError(f"prompt {prompt.key} uses {prompt.rubric_version}, not {cfg.rubric_version}")
    if raw is not None:
        return
    if local:
        return  # $0 per call: no budget needed; reachability is checked by _local_runtime

    if settings.llm_provider == LLMProviderName.HEURISTIC:
        raise PreflightError("AVD_LLM_PROVIDER is heuristic: configure anthropic or openai_compatible")
    if settings.llm_provider == LLMProviderName.ANTHROPIC and settings.anthropic_api_key is None:
        raise PreflightError(
            "AVD_ANTHROPIC_API_KEY is not set (add it as an environment secret, never in code)"
        )
    if settings.llm_provider == LLMProviderName.OPENAI_COMPATIBLE and settings.openai_api_key is None:
        host = settings.openai_base_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise PreflightError("AVD_OPENAI_API_KEY is not set for a remote OpenAI-compatible endpoint")
    if not cfg.budget_usd or cfg.budget_usd <= 0:
        raise PreflightError("--budget-usd is required for a live run (an explicitly approved ceiling)")


def _item_record(s: Any, vid: str, hashes: set[str]) -> dict[str, Any]:
    ledger = s.scalars(select(CostLedger).where(CostLedger.video_id == vid)).all()
    status = Counter(r.status for r in ledger)
    errors = Counter(r.error_type for r in ledger if r.error_type)
    ev = s.scalar(
        select(Evaluation)
        .where(Evaluation.video_id == vid, Evaluation.is_current.is_(True))
        .order_by(Evaluation.created_at.desc())
    )
    rec: dict[str, Any] = {
        "video_id": vid,
        "calls": {
            "ok": status.get("ok", 0),
            "cached": status.get("cached", 0),
            "error": status.get("error", 0),
        },
        "errors": dict(errors),
        "parse_failures": errors.get("LLMOutputError", 0),
        "refusals": errors.get("LLMRefusalError", 0),
        "budget_refusals": errors.get("LLMBudgetExceeded", 0),
        "input_tokens": sum(r.input_tokens for r in ledger),
        "output_tokens": sum(r.output_tokens for r in ledger),
        "cost_usd": round(sum(r.cost_usd for r in ledger), 6),
        "latency_ms": [r.latency_ms for r in ledger if r.status == "ok" and r.latency_ms is not None],
        "models_served": sorted({r.model for r in ledger if r.model and r.status == "ok"}),
        "evaluation": None,
    }
    if ev is not None:
        dims = ev.dimensions
        v = ev.validation or {}
        rec["evaluation"] = {
            "id": ev.id,
            "stage": ev.stage,
            "mode": ev.mode,
            "status": ev.status,
            "config_hash": ev.config_hash,
            "matches_run_config": ev.config_hash in hashes,
            "overall": ev.overall_score,
            "scores": {d.dimension: d.score for d in dims if d.status == "scored" and d.score is not None},
            "scored": sum(1 for d in dims if d.status == "scored"),
            "abstained": dict(Counter(f"{d.status}:{d.reason_code}" for d in dims if d.status != "scored")),
            "evidence_claimed": v.get("evidence_claimed", 0),
            "evidence_verified": v.get("evidence_verified", 0),
            "ungrounded_dimensions": v.get("ungrounded_dimensions", []),
            "validation_passed": v.get("passed"),
            "needs_review": ev.needs_review,
        }
    return rec


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    lat = [x for r in records for x in r["latency_ms"]]
    evs = [r["evaluation"] for r in records if r["evaluation"]]
    claimed = sum(e["evidence_claimed"] for e in evs)
    verified = sum(e["evidence_verified"] for e in evs)
    llm_evs = [e for e in evs if e["mode"] == "llm"]
    return {
        "items": len(records),
        "evaluated_by_llm": len(llm_evs),
        "evaluation_status": dict(Counter(e["status"] for e in llm_evs)),
        "calls": {k: sum(r["calls"][k] for r in records) for k in ("ok", "cached", "error")},
        "errors": dict(sum((Counter(r["errors"]) for r in records), Counter())),
        "parse_failures": sum(r["parse_failures"] for r in records),
        "refusals": sum(r["refusals"] for r in records),
        "budget_refusals": sum(r["budget_refusals"] for r in records),
        "evidence": {
            "claimed": claimed,
            "verified": verified,
            "invalid_rate": round(1 - verified / claimed, 4) if claimed else None,
            "ungrounded_scored_dimensions": sum(len(e["ungrounded_dimensions"]) for e in llm_evs),
        },
        "abstentions": dict(sum((Counter(e["abstained"]) for e in llm_evs), Counter())),
        "tokens": {
            "input": sum(r["input_tokens"] for r in records),
            "output": sum(r["output_tokens"] for r in records),
        },
        "cost_usd": round(sum(r["cost_usd"] for r in records), 4),
        "latency_ms": {
            "p50": statistics.median(lat) if lat else None,
            "p95": sorted(lat)[max(0, int(0.95 * len(lat)) - 1)] if lat else None,
            "n": len(lat),
        },
    }


def run(
    cfg: RunConfig, base_settings: Settings | None = None, raw_provider: LLMProvider | None = None
) -> dict[str, Any]:
    base = base_settings or get_settings()
    if not cfg.run_id:
        cfg.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.resolved_out()
    (out / "items").mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cfg.manifest)
    guard = _guard_split(cfg, manifest)
    items, excluded = _select(cfg, manifest)
    if not items:
        raise PreflightError(f"no items in split {cfg.split!r} are cleared for analysis")
    settings = _settings(cfg, base, out)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    fetcher = SourceFetcher(cfg.cache_dir or base.dataset_cache_dir, offline=cfg.offline)
    with sf() as s:
        import_manifest(s, cfg.manifest, fetcher)
        s.commit()
    ids = [it["item_id"] for it in items]
    plan = plan_calls(cfg, settings, sf, ids)
    stages = ("B", "C") if cfg.stage_c else ("B",)
    expected = round(sum(p["stages"][st]["expected_usd"] for p in plan for st in stages), 4)
    worst = round(sum(p["stages"][st]["worst_case_usd"] for p in plan for st in stages), 4)
    prompt = get_prompt(settings.llm_prompt)
    meta: dict[str, Any] = {
        "run_id": cfg.run_id,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": f"{manifest['name']}@{manifest['version']}",
        "manifest_items_sha256": manifest["items_sha256"],
        "manifest_file_sha256": hashlib.sha256(cfg.manifest.read_bytes()).hexdigest(),
        "split": cfg.split,
        "items": ids,
        "excluded": excluded,
        "provider": "dry_run"
        if cfg.dry_run
        else (raw_provider.name if raw_provider else settings.llm_provider.value),
        "model": settings.llm_model_stage_b,
        "prompt": prompt.key,
        "rubric_version": prompt.rubric_version,
        "stage_c": cfg.stage_c,
        "concurrency": cfg.concurrency,
        "timeout_seconds": cfg.timeout_seconds,
        "max_retries": cfg.max_retries,
        "budget_usd": cfg.budget_usd,
        "projection": {
            "expected_usd": expected,
            "worst_case_usd": worst,
            "assumptions": {
                "chars_per_token_expected": CHARS_PER_TOKEN,
                "chars_per_token_worst": RunBudget.CHARS_PER_TOKEN_WORST,
                "output_tokens_expected": cfg.output_token_assumption,
                "output_tokens_worst": "max_tokens of each request",
            },
        },
        "git_commit": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        **guard,
    }
    (out / "plan.json").write_text(json.dumps({"meta": meta, "plan": plan}, indent=2))
    if cfg.dry_run:
        return {"meta": meta, "plan": plan, "dry_run": True}

    _check_live_config(cfg, settings, raw_provider)
    local = _is_local(settings, raw_provider)
    if local:
        from avd.llm.local import runtime_snapshot

        try:
            meta["local_runtime"] = runtime_snapshot(settings, raw_provider)
        except LLMError as exc:
            raise PreflightError(f"local model not usable: {exc}") from exc
        (out / "plan.json").write_text(json.dumps({"meta": meta, "plan": plan}, indent=2))
        if cfg.budget_usd is None:
            cfg.budget_usd = 0.0  # a $0 ceiling: any call that would cost money is refused
    assert cfg.budget_usd is not None
    if expected > cfg.budget_usd:
        raise PreflightError(
            f"expected cost ${expected:.2f} exceeds the approved budget ${cfg.budget_usd:.2f}; "
            "reduce --max-items or get approval for a larger budget"
        )
    with sf() as s:
        already = float(s.scalar(select(func.coalesce(func.sum(CostLedger.cost_usd), 0.0))) or 0.0)
    budget = RunBudget(cfg.budget_usd, already_spent_usd=already)
    llm = build_llm_provider(settings, sf, raw=raw_provider, run_budget=budget)
    ctx = build_context(settings=settings, session_factory=sf, llm=llm)
    hashes = {ctx.pipeline(st).config_hash() for st in ("B", "C")}

    todo = []
    with sf() as s:
        for vid in ids:
            rec = _item_record(s, vid, hashes)
            done = (
                rec["evaluation"]
                and rec["evaluation"]["mode"] == "llm"
                and rec["evaluation"]["matches_run_config"]
            )
            if not done:
                todo.append(vid)
                enqueue_stage_a(s, vid, force=True)
        s.commit()

    t0 = time.monotonic()
    claim_lock = threading.Lock()  # SQLite has no row locks: serialise claiming, not execution

    def worker() -> None:
        while True:
            with claim_lock:
                jids = claim_batch(ctx, limit=1)
            if not jids:
                return
            execute_job(ctx, jids[0])

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, cfg.concurrency))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    with sf() as s:
        records = [_item_record(s, vid, hashes) for vid in ids]
    for r in records:
        (out / "items" / f"{r['video_id']}.json").write_text(json.dumps(r, indent=2))
    report = {
        "meta": {
            **meta,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "runtime_seconds": round(time.monotonic() - t0, 2),
            "resumed_items_skipped": len(ids) - len(todo),
            "spent_before_this_session_usd": round(already, 6),
            "budget_refusals_this_session": budget.refused,
            "pipeline_config_hashes": sorted(hashes),
        },
        "summary": _summary(records),
        "items": records,
        "interpretation": (
            "Operational evidence only: schema validity, grounding, abstention, cost and latency. "
            "Agreement with people requires human labels (avd bench-real --labels)."
        ),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(render_markdown(report))
    return report


def render_markdown(r: dict[str, Any]) -> str:
    m, s = r["meta"], r["summary"]
    lines = [
        f"# Real evaluation run {m['run_id']}",
        "",
        f"> {r['interpretation']}",
        "",
        "| key | value |",
        "|---|---|",
    ]
    for k in (
        "dataset",
        "split",
        "provider",
        "model",
        "prompt",
        "rubric_version",
        "stage_c",
        "budget_usd",
        "git_commit",
        "git_dirty",
        "runtime_seconds",
        "resumed_items_skipped",
    ):
        lines.append(f"| {k} | `{m.get(k)}` |")
    lines.append(
        f"| projection (expected / worst case) | ${m['projection']['expected_usd']} / ${m['projection']['worst_case_usd']} |"
    )
    lines += [
        "",
        "## Results",
        "",
        f"* Items: {s['items']}; evaluated by the LLM: {s['evaluated_by_llm']}; status {s['evaluation_status']}",
        f"* Calls: {s['calls']}; errors by type: {s['errors']}",
        f"* Parse failures: {s['parse_failures']}; refusals: {s['refusals']}; budget refusals: {s['budget_refusals']}",
        f"* Evidence: {s['evidence']}",
        f"* Abstentions (status:reason): {s['abstentions']}",
        f"* Tokens: {s['tokens']}; cost: ${s['cost_usd']}; call latency ms: {s['latency_ms']}",
        "",
        "## Per item",
        "",
        "| item | status | scored | evidence verified/claimed | ungrounded | calls ok/err | cost $ |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for it in r["items"]:
        e = it["evaluation"] or {}
        lines.append(
            f"| {it['video_id']} | {e.get('status', '—')} ({e.get('mode', '—')}) | {e.get('scored', '—')} | "
            f"{e.get('evidence_verified', 0)}/{e.get('evidence_claimed', 0)} | {len(e.get('ungrounded_dimensions', []))} | "
            f"{it['calls']['ok']}/{it['calls']['error']} | {it['cost_usd']} |"
        )
    return "\n".join(lines) + "\n"


def add_parser(sub: Any) -> None:
    p = sub.add_parser(
        "evaluate-real", help="budget-capped, resumable live evaluation of authorized real items"
    )
    p.add_argument(
        "--manifest", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    )
    p.add_argument("--split", default="dev", choices=["calibration", "dev", "test"])
    p.add_argument("--run-id", default="", help="reuse an id to resume an interrupted run")
    p.add_argument("--out", default=None)
    p.add_argument("--budget-usd", type=float, default=None, help="approved hard ceiling (required live)")
    p.add_argument("--dry-run", action="store_true", help="plan and project cost; no API calls")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--model", default=None, help="defaults to AVD_LLM_MODEL_STAGE_B")
    p.add_argument("--prompt", default=None, help="registered prompt key (avd prompts)")
    p.add_argument("--rubric-version", default=None)
    p.add_argument("--stage-c", action="store_true", help="allow Stage C escalation (3 more calls per item)")
    p.add_argument("--confirm-test", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--cache-dir", default=None)
    p.add_argument(
        "--provider",
        default=None,
        choices=[v.value for v in LLMProviderName if v != LLMProviderName.HEURISTIC],
        help="override AVD_LLM_PROVIDER (ollama: local, $0, no budget needed)",
    )
    p.set_defaults(fn=main)


def main(args: argparse.Namespace) -> None:
    cfg = RunConfig(
        manifest=Path(args.manifest),
        split=args.split,
        run_id=args.run_id,
        out_dir=Path(args.out) if args.out else None,
        budget_usd=args.budget_usd,
        dry_run=args.dry_run,
        max_items=args.max_items,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        model=args.model,
        prompt=args.prompt,
        rubric_version=args.rubric_version,
        stage_c=args.stage_c,
        confirm_test=args.confirm_test,
        offline=args.offline,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        provider=args.provider,
    )
    try:
        r = run(cfg)
    except (PreflightError, SplitGuardError) as exc:
        raise SystemExit(f"refused: {exc}") from exc
    m = r["meta"]
    print(
        json.dumps(
            {
                "run_id": m["run_id"],
                "out": str(cfg.resolved_out()),
                "dry_run": r.get("dry_run", False),
                "items": len(m["items"]),
                "projection": m["projection"],
                "summary": r.get("summary"),
            },
            indent=2,
        )
    )
