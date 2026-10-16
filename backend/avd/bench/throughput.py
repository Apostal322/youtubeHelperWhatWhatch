"""Throughput and cost experiment with scale projections.

Two kinds of numbers are kept apart:

* **Measured** — wall-clock latency of the heuristic pipeline and of the embedders on the real
  pilot transcripts in this environment, and the exact **LLM call plan** the production LLM
  pipeline issues per item (number of calls, map/reduce chunking, prompt characters). The call
  plan is recorded by a dry-run provider that returns schema-valid *empty* answers; no model API
  is called.
* **Projected** — dollars, tokens and worker counts for 1k / 10k / 100k videos per day. These
  rest on stated assumptions: ~4 characters per token (no tokenizer), output tokens per call
  type, the Stage C escalation rate, LLM latency, and that pilot-length (~1 h) items are
  representative. Real usage must be measured with a live, budget-capped run
  (``avd llm-smoke``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from avd.bench import stats as S
from avd.bench.harness import _git_sha
from avd.config import REPO_ROOT, LLMProviderName, get_settings
from avd.content.processing import normalize_transcript
from avd.datasets.fetch import SourceFetcher
from avd.datasets.pilot import load_manifest
from avd.datasets.sources import SOURCES, parse
from avd.embeddings.base import HashingEmbedder
from avd.evaluation.genre import KeywordGenreClassifier
from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.pipeline import build_pipeline
from avd.evaluation.schema import Modality
from avd.llm.base import LLMRequest, LLMResponse
from avd.llm.pricing import cost_usd, price_for

CHARS_PER_TOKEN = 4.0  # assumption: planning approximation, not a tokenizer
# Assumption. On Opus 5.5 thinking is always on and billed as output, so these may be low; the
# report therefore also shows a 3x output-token sensitivity.
OUTPUT_TOKENS = {"chunk_notes": 1200, "rubric_evaluation": 2500, "default": 1500}
OUTPUT_SENSITIVITY = 3.0


class DryRunProvider:
    """Records each request and returns the smallest schema-valid answer (nothing assessable)."""

    name = "dry_run"

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        props = request.json_schema.get("properties", {})
        if "judgments" in props:
            dims = props["judgments"]["items"]["properties"]["dimension"]["enum"]
            data: dict[str, Any] = {
                "content_summary": "",
                "detected_issues": [],
                "judgments": [
                    {
                        "dimension": d,
                        "assessable": False,
                        "score": 0,
                        "confidence": 0.0,
                        "justification": "dry run",
                        "evidence": [],
                        "counter_evidence": [],
                    }
                    for d in dims
                ],
            }
        else:
            data = {k: ([] if v.get("type") == "array" else "") for k, v in props.items()}
        return LLMResponse(
            data=data,
            provider=self.name,
            requested_model=request.model,
            served_model=request.model,
            input_tokens=0,
            output_tokens=0,
            stop_reason="end_turn",
        )


def _items(manifest_path: Path) -> list[tuple[str, Any]]:
    m = load_manifest(manifest_path)
    fetcher = SourceFetcher(get_settings().dataset_cache_dir, offline=True)
    out = []
    for it in m["items"]:
        src = SOURCES[it["source"]]
        t, _ = parse(src, it["path"], fetcher.get(src, it["path"], it["raw_sha256"]).decode("utf-8"))
        out.append((it["item_id"], normalize_transcript(t)[0]))
    return out


def _input(vid: str, transcript: Any) -> EvaluationInput:
    genre = KeywordGenreClassifier().classify("", "", [], None, transcript.text)
    return EvaluationInput(
        video_id=vid,
        title="",
        description="",
        duration_seconds=int(transcript.duration),
        language="en",
        tags=(),
        genre=genre,
        transcript=transcript,
        transcript_source_id="s",
        audio_path=None,
        audio_sha256=None,
        audio_source_id=None,
        modalities=frozenset({Modality.TRANSCRIPT}),
        content_source_ids=("s",),
        title_is_original=False,
    )


def run(manifest_path: Path, stage_c_rate: float = 0.2, llm_latency_s: float = 60.0) -> dict[str, Any]:
    settings = get_settings()
    items = _items(manifest_path)
    # ---- measured: heuristic pipeline latency and embedder throughput
    heuristic = build_pipeline(
        "B", settings.model_copy(update={"llm_provider": LLMProviderName.HEURISTIC}), None
    )
    lat = []
    for vid, t in items:
        t0 = time.perf_counter()
        heuristic.run(_input(vid, t))
        lat.append((time.perf_counter() - t0) * 1000)
    hashing = HashingEmbedder()
    t0 = time.perf_counter()
    for _, t in items:
        hashing.embed([t.text[:20000]])
    hash_ms = (time.perf_counter() - t0) * 1000 / len(items)
    # ---- measured: LLM call plan (dry run, no API)
    plans: dict[str, list[dict[str, Any]]] = {}
    for stage in ("B", "C"):
        stage_plans = []
        for vid, t in items:
            dry = DryRunProvider()
            build_pipeline(
                stage, settings.model_copy(update={"llm_provider": LLMProviderName.ANTHROPIC}), dry
            ).run(_input(vid, t))
            # The dry-run answers must be valid: a repair call would inflate the plan.
            if any("output contract" in c.user for c in dry.calls):
                raise RuntimeError("dry-run response failed validation; the call plan would include repairs")
            by_kind: dict[str, dict[str, float]] = {}
            for c in dry.calls:
                k = by_kind.setdefault(c.schema_name, {"calls": 0, "prompt_chars": 0})
                k["calls"] += 1
                k["prompt_chars"] += len(c.system) + len(c.user)
            stage_plans.append(
                {
                    "video_id": vid,
                    "words": t.word_count,
                    "calls": len(dry.calls),
                    "by_kind": by_kind,
                    "model": dry.calls[0].model if dry.calls else None,
                }
            )
        plans[stage] = stage_plans

    def cost_of(plan: dict[str, Any], out_mult: float = 1.0) -> tuple[float, float, float]:
        calls = sum(k["calls"] for k in plan["by_kind"].values())
        tin = sum(k["prompt_chars"] for k in plan["by_kind"].values()) / CHARS_PER_TOKEN
        tout = out_mult * sum(
            k["calls"] * OUTPUT_TOKENS.get(name, OUTPUT_TOKENS["default"])
            for name, k in plan["by_kind"].items()
        )
        if not plan["model"] or not calls:
            return tin, tout, 0.0
        # Priced per call so long-prompt tiers apply to the prompt size of each request.
        return tin, tout, calls * cost_usd(plan["model"], int(tin / calls), int(tout / calls))

    per_stage = {}
    for stage, ps in plans.items():
        costs = [cost_of(p) for p in ps]
        per_stage[stage] = {
            "model": ps[0]["model"],
            "calls_per_video": S.percentiles([p["calls"] for p in ps]),
            "calls_mean": round(statistics.fmean(p["calls"] for p in ps), 2),
            "call_kinds": sorted({k for p in ps for k in p["by_kind"]}),
            "est_input_tokens_mean": round(statistics.fmean(c[0] for c in costs)),
            "assumed_output_tokens_mean": round(statistics.fmean(c[1] for c in costs)),
            "est_cost_usd_mean": round(statistics.fmean(c[2] for c in costs), 4),
            "est_cost_usd_p95": S.percentiles([c[2] for c in costs])["p95"],
            "est_cost_usd_mean_output_x3": round(
                statistics.fmean(cost_of(p, OUTPUT_SENSITIVITY)[2] for p in ps), 4
            ),
        }
    per_video = per_stage["B"]["est_cost_usd_mean"] + stage_c_rate * per_stage["C"]["est_cost_usd_mean"]
    per_video_x3 = (
        per_stage["B"]["est_cost_usd_mean_output_x3"]
        + stage_c_rate * per_stage["C"]["est_cost_usd_mean_output_x3"]
    )
    calls_per_video = per_stage["B"]["calls_mean"] + stage_c_rate * per_stage["C"]["calls_mean"]
    tokens_per_video = (
        per_stage["B"]["est_input_tokens_mean"]
        + per_stage["B"]["assumed_output_tokens_mean"]
        + stage_c_rate
        * (per_stage["C"]["est_input_tokens_mean"] + per_stage["C"]["assumed_output_tokens_mean"])
    )
    projections = []
    for n in (1_000, 10_000, 100_000):
        calls_per_min = n * calls_per_video / 1440
        projections.append(
            {
                "videos_per_day": n,
                "llm_usd_per_day": round(n * per_video, 0),
                "llm_usd_per_day_output_x3": round(n * per_video_x3, 0),
                "llm_usd_per_month": round(n * per_video * 30, 0),
                "tokens_per_day_millions": round(n * tokens_per_video / 1e6, 1),
                "llm_calls_per_minute": round(calls_per_min, 1),
                "concurrent_llm_requests": round(calls_per_min * llm_latency_s / 60, 1),
                "heuristic_cpu_seconds_per_day": round(n * statistics.fmean(lat) / 1000, 0),
            }
        )
    return {
        "meta": {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_sha(),
            "dataset": "real_pilot@v1 (29 long-form episodes, median ~11k words)",
            "assumptions": {
                "chars_per_token": CHARS_PER_TOKEN,
                "output_tokens_per_call": OUTPUT_TOKENS,
                "stage_c_escalation_rate": stage_c_rate,
                "llm_latency_seconds_per_call": llm_latency_s,
                "prices_usd_per_mtok": {
                    m: price_for(m) for m in {per_stage["B"]["model"], per_stage["C"]["model"]} if m
                },
                "representativeness": "pilot items are ~1 h podcasts; typical YouTube videos are shorter, so per-video cost is likely lower",
            },
        },
        "measured": {
            "heuristic_pipeline_ms_per_video": S.percentiles(lat),
            "hashing_embed_ms_per_video": round(hash_ms, 2),
            "minilm_embed_ms_per_video": "805 (retrieval experiment, 24 windows/doc, CPU)",
            "llm_call_plan": per_stage,
        },
        "projected": projections,
    }


def render_markdown(r: dict[str, Any]) -> str:
    m, me = r["meta"], r["measured"]
    out = [
        "# Throughput and cost: measured call plan, projected scale",
        "",
        f"Commit `{m['git_commit']}`; data: {m['dataset']}.",
        "",
        "> **No live LLM calls were made.** The LLM call plan is *measured* by running the production",
        "> pipeline with a dry-run provider; the dollar and token figures are *projections* from",
        "> the stated assumptions below.",
        "",
        "## Measured (this environment)",
        "",
        f"* Heuristic pipeline per video (ms): {me['heuristic_pipeline_ms_per_video']}",
        f"* Hashing embedding per video: {me['hashing_embed_ms_per_video']} ms; MiniLM ONNX: {me['minilm_embed_ms_per_video']} ms",
        "",
        "| stage | model | LLM calls per video (mean; p50/p95) | call kinds | est. input tokens | assumed output tokens | est. $ per video (mean; p95) |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for st, x in me["llm_call_plan"].items():
        out.append(
            f"| {st} | {x['model']} | {x['calls_mean']}; {x['calls_per_video']} | {', '.join(x['call_kinds'])} | "
            f"{x['est_input_tokens_mean']:,} | {x['assumed_output_tokens_mean']:,} | {x['est_cost_usd_mean']}; {x['est_cost_usd_p95']} |"
        )
    out += [
        "",
        "## Projected (assumptions, not measurements)",
        "",
        f"Assumptions: `{json.dumps(m['assumptions'])}`",
        "",
        "| videos/day | LLM $/day | $/day if output ×3 | LLM $/month | tokens/day (M) | LLM calls/min | concurrent requests | heuristic CPU s/day |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in r["projected"]:
        out.append(
            f"| {p['videos_per_day']:,} | {p['llm_usd_per_day']:,.0f} | {p['llm_usd_per_day_output_x3']:,.0f} | {p['llm_usd_per_month']:,.0f} | {p['tokens_per_day_millions']} | "
            f"{p['llm_calls_per_minute']} | {p['concurrent_llm_requests']} | {p['heuristic_cpu_seconds_per_day']:,.0f} |"
        )
    out.append("")
    return "\n".join(out)


def add_parser(sub: Any) -> None:
    p = sub.add_parser(
        "bench-throughput", help="measured LLM call plan + projected cost at scale (no API calls)"
    )
    p.add_argument(
        "--manifest", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    )
    p.add_argument("--stage-c-rate", type=float, default=0.2)
    p.add_argument("--llm-latency", type=float, default=60.0)
    p.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "reports"))
    p.add_argument("--tag", default="throughput-real_pilot_v1")
    p.set_defaults(fn=main)


def main(args: argparse.Namespace) -> None:
    report = run(Path(args.manifest), args.stage_c_rate, args.llm_latency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    (out / f"{args.tag}.md").write_text(render_markdown(report))
    print(render_markdown(report))
