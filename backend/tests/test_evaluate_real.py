"""Phase 3: `avd evaluate-real` — pre-flight, hard budget ceiling, resume, guards, concurrency.

No live inference: a scripted judge answers from the prompt it receives (it cites real segments
it was shown), so grounding validation runs for real.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from avd.bench import evaluate_real as er
from avd.bench.real import SplitGuardError
from avd.datasets.pilot import describe_item, finalize_manifest
from avd.datasets.sources import CHANGELOG, TALKPYTHON
from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS
from avd.llm.base import LLMRequest
from tests.conftest import ScriptedProvider, make_settings
from tests.test_e2e_scenario import CACHING_EPISODE, PACKAGING_TALK, TRACING_EPISODE, _write

_SEG = re.compile(r"^\[(\d+)\] \(\d[^)]*\) (.+)$", re.M)


def grounded_judge(request: LLMRequest, _i: int) -> dict[str, Any]:
    """Scores 3 everywhere and cites the first segment it was shown, verbatim."""
    m = _SEG.search(request.user)
    assert m, "prompt must contain numbered segments"
    seg, text = int(m.group(1)), " ".join(m.group(2).split()[:8])
    if request.schema_name != "rubric_evaluation":
        props = request.json_schema.get("properties", {})
        return {k: ([] if v.get("type") == "array" else "") for k, v in props.items()}
    return {
        "content_summary": "summary",
        "detected_issues": [],
        "judgments": [
            {
                "dimension": d.value,
                "assessable": True,
                "score": 3,
                "confidence": 0.6,
                "justification": "Cited segment supports a middling judgement.",
                "evidence": [{"segment_index": seg, "quote": text}],
                "counter_evidence": [],
                "unassessable_reason": "",
            }
            for d in TEXT_DIMENSIONS
        ],
    }


@pytest.fixture
def manifest(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    items = []
    for source, path, body in (
        (CHANGELOG, "podcast/the-changelog-1.md", CACHING_EPISODE),
        (CHANGELOG, "gotime/go-time-2.md", TRACING_EPISODE),
        (TALKPYTHON, "youtube_transcripts/301-packaging.vtt", PACKAGING_TALK),
    ):
        item, _ = describe_item(source, path, _write(cache, source, path, body))
        items.append(item)
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(finalize_manifest(items, {}, seed=1, forced_split="dev")))
    return out, cache


def _cfg(tmp_path: Path, manifest: tuple[Path, Path], **kw: Any) -> er.RunConfig:
    path, cache = manifest
    kw.setdefault("run_id", "t")
    kw.setdefault("out_dir", tmp_path / "run")
    return er.RunConfig(manifest=path, cache_dir=cache, offline=True, **kw)


def _settings(tmp_path: Path) -> Any:
    return make_settings(tmp_path, llm_provider="anthropic")


def test_dry_run_plans_without_calls(tmp_path: Path, manifest: tuple[Path, Path]) -> None:
    r = er.run(_cfg(tmp_path, manifest, dry_run=True), base_settings=_settings(tmp_path))
    assert r["dry_run"] and len(r["plan"]) == 3
    p = r["meta"]["projection"]
    assert 0 < p["expected_usd"] < p["worst_case_usd"]
    assert all(e["stages"]["B"]["calls"] >= 1 for e in r["plan"])
    assert (tmp_path / "run" / "plan.json").exists() and not (tmp_path / "run" / "summary.json").exists()


def test_live_run_records_grounding_and_resumes_without_paying_twice(
    tmp_path: Path, manifest: tuple[Path, Path]
) -> None:
    judge = ScriptedProvider(grounded_judge)
    cfg = _cfg(tmp_path, manifest, budget_usd=5.0)
    r = er.run(cfg, base_settings=_settings(tmp_path), raw_provider=judge)
    s = r["summary"]
    assert s["items"] == 3 and s["evaluated_by_llm"] == 3
    assert s["evidence"]["claimed"] > 0 and s["evidence"]["invalid_rate"] == 0.0
    assert s["parse_failures"] == 0 and s["calls"]["error"] == 0
    assert 0 < s["cost_usd"] <= 5.0
    assert r["meta"]["prompt"] == "rubric_prompt_v2" and r["meta"]["model"] == "claude-opus-5-5"
    first_calls = len(judge.calls)
    again = er.run(
        _cfg(tmp_path, manifest, budget_usd=5.0), base_settings=_settings(tmp_path), raw_provider=judge
    )
    assert len(judge.calls) == first_calls  # completed items are skipped
    assert again["meta"]["resumed_items_skipped"] == 3
    assert again["meta"]["spent_before_this_session_usd"] == pytest.approx(s["cost_usd"], abs=1e-6)
    assert (tmp_path / "run" / "report.md").read_text().startswith("# Real evaluation run t")


def test_prompt_variant_is_a_separate_configuration(tmp_path: Path, manifest: tuple[Path, Path]) -> None:
    judge = ScriptedProvider(grounded_judge)
    base = er.run(
        _cfg(tmp_path, manifest, budget_usd=5.0, max_items=1),
        base_settings=_settings(tmp_path),
        raw_provider=judge,
    )
    n = len(judge.calls)
    var = er.run(
        _cfg(tmp_path, manifest, budget_usd=5.0, max_items=1, prompt="rubric_prompt_v2-evidence_first"),
        base_settings=_settings(tmp_path),
        raw_provider=judge,
    )
    assert len(judge.calls) > n  # not served from the base prompt's cache
    assert "select the evidence" in judge.calls[-1].system
    assert var["meta"]["pipeline_config_hashes"] != base["meta"]["pipeline_config_hashes"]


def test_preflight_refusals(tmp_path: Path, manifest: tuple[Path, Path]) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(er.PreflightError, match="ANTHROPIC_API_KEY"):
        er.run(_cfg(tmp_path, manifest, budget_usd=5.0), base_settings=settings)
    judge = ScriptedProvider(grounded_judge)
    with pytest.raises(er.PreflightError, match="exceeds the approved budget"):
        er.run(_cfg(tmp_path, manifest, budget_usd=0.0001), base_settings=settings, raw_provider=judge)
    assert judge.calls == []
    with pytest.raises(er.PreflightError, match="no known price"):
        er.run(
            _cfg(tmp_path, manifest, budget_usd=5.0, model="mystery-model"),
            base_settings=settings,
            raw_provider=judge,
        )
    with pytest.raises(ValueError, match="unknown prompt"):
        er.run(
            _cfg(tmp_path, manifest, budget_usd=5.0, prompt="nope"),
            base_settings=settings,
            raw_provider=judge,
        )


def test_hard_ceiling_holds_when_actual_usage_runs_high(tmp_path: Path, manifest: tuple[Path, Path]) -> None:
    class Expensive(ScriptedProvider):
        def complete_json(self, request: LLMRequest) -> Any:
            resp = super().complete_json(request)
            resp.input_tokens, resp.output_tokens = 60_000, 16_000  # far above the projection
            return resp

    judge = Expensive(grounded_judge)
    dry = er.run(_cfg(tmp_path, manifest, dry_run=True), base_settings=_settings(tmp_path))
    ceiling = round(dry["meta"]["projection"]["expected_usd"] * 1.5, 4)
    r = er.run(
        _cfg(tmp_path, manifest, budget_usd=ceiling, run_id="t2", out_dir=tmp_path / "run2"),
        base_settings=_settings(tmp_path),
        raw_provider=judge,
    )
    s = r["summary"]
    assert s["cost_usd"] <= ceiling
    assert s["budget_refusals"] >= 1 and s["evaluated_by_llm"] < 3  # stopped, never overspent


def test_held_out_split_requires_confirmation(tmp_path: Path, manifest: tuple[Path, Path]) -> None:
    path, cache = manifest
    m = json.loads(path.read_text())
    m["name"] = "real_holdout"
    held = tmp_path / "holdout.json"
    from avd.datasets.pilot import items_digest

    m["items_sha256"] = items_digest(m["items"])
    held.write_text(json.dumps(m))
    with pytest.raises(SplitGuardError, match="held out"):
        er.run(
            er.RunConfig(
                manifest=held, cache_dir=cache, offline=True, run_id="h", out_dir=tmp_path / "h", dry_run=True
            )
        )


def test_concurrent_items(tmp_path: Path, manifest: tuple[Path, Path]) -> None:
    judge = ScriptedProvider(grounded_judge)
    r = er.run(
        _cfg(tmp_path, manifest, budget_usd=5.0, concurrency=3),
        base_settings=_settings(tmp_path),
        raw_provider=judge,
    )
    assert r["summary"]["evaluated_by_llm"] == 3 and r["summary"]["calls"]["error"] == 0
    # each item evaluated exactly once (no double-claimed jobs)
    per_item = [it["calls"]["ok"] for it in r["items"]]
    assert per_item == [per_item[0]] * 3
