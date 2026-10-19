"""Phase 3.1: local inference through Ollama.

Everything except the ``live`` tests runs against ``tests/ollama_mock.py``, a protocol mock of the
Ollama HTTP API (no model, no GPU, no network). Those tests check the client, error semantics,
context sizing, pricing and the evaluation path; they say nothing about model quality.

Live tests run only when ``AVD_TEST_OLLAMA_URL`` and ``AVD_TEST_OLLAMA_MODEL`` are set and the
server answers; otherwise they are *skipped*, never counted as real inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from avd.bench import evaluate_real as er
from avd.bench import local as bl
from avd.config import LOCAL_PROMPT_OVERHEAD_TOKENS, LLMProviderName, Settings
from avd.llm.base import (
    LLMConfigurationError,
    LLMOutputError,
    LLMRequest,
    LLMTransientError,
)
from avd.llm.local import CANDIDATES, GPU, Hardware, detect_hardware, estimate_memory_gb, pull, recommend
from avd.llm.ollama import LLMContextOverflow, OllamaProvider, check_local_url
from avd.llm.pricing import cost_usd, is_local_model, register_local_model
from avd.llm.providers import RetryingProvider, build_llm_provider
from tests.conftest import make_settings
from tests.ollama_mock import FakeOllama
from tests.test_evaluate_real import manifest  # noqa: F401  (fixture)

MODEL = "qwen3.5:4b"


def _provider(fake: FakeOllama, **kw: Any) -> OllamaProvider:
    return OllamaProvider("http://localhost:11434", transport=fake.transport(), **kw)


def _req(user: str = "[0] (00:00) hello world", **kw: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "system": "sys",
        "user": user,
        "json_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        "schema_name": "t",
        "model": MODEL,
        "max_tokens": 512,
    }
    base.update(kw)
    return LLMRequest(**base)


# ------------------------------------------------------------------------------ client


class TestClient:
    def test_connection_model_info_and_fingerprint(self) -> None:
        fake = FakeOllama()
        p = _provider(fake, num_ctx=16384)
        assert p.version() == "0.0.0-mock"
        info = p.model_info(MODEL)
        assert info.quantization == "Q4_K_M" and info.context_length == 262144
        assert info.fingerprint == f"{MODEL}@sha256:ababa:Q4_K_M"
        assert p.effective_context(MODEL) == 16384  # min(requested, native)

    def test_request_shape_native_api(self) -> None:
        fake = FakeOllama(answer=lambda b: {"a": "x"})
        p = _provider(fake, num_ctx=8192, think=False, seed=7)
        r = p.complete_json(_req(sample_index=2, temperature=0.2))
        body = fake.chat_bodies[0]
        assert body["format"]["properties"] == {"a": {"type": "string"}} and body["stream"] is False
        assert body["options"]["num_ctx"] == 8192 and body["options"]["seed"] == 9
        assert body["options"]["temperature"] == 0.2 and body["think"] is False  # model can think
        assert r.data == {"a": "x"} and r.provider == "ollama"
        assert r.served_model.startswith(f"ollama:{MODEL}@") and r.input_tokens > 0 and r.output_tokens > 0
        assert r.runtime and r.runtime["num_ctx"] == 8192 and r.runtime["output_tokens_per_s"] > 0

    def test_think_is_omitted_for_models_without_thinking(self) -> None:
        fake = FakeOllama(answer=lambda b: {"a": "x"})
        fake.models[MODEL]["capabilities"] = ["completion"]
        _provider(fake).complete_json(_req())
        assert "think" not in fake.chat_bodies[0]

    def test_no_proxy_for_local_traffic(self) -> None:
        assert _provider(FakeOllama())._http.trust_env is False

    def test_trailing_v1_is_accepted(self) -> None:
        assert OllamaProvider("http://localhost:11434/v1").base_url == "http://localhost:11434"


class TestFailures:
    def test_unavailable_server_is_transient(self) -> None:
        fake = FakeOllama()
        fake.down = True
        p = _provider(fake)
        with pytest.raises(LLMTransientError):
            p.version()
        with pytest.raises(LLMTransientError):
            p.complete_json(_req())

    def test_missing_model_names_the_pull_command(self) -> None:
        with pytest.raises(LLMConfigurationError, match=r"ollama pull qwen3\.5:9b"):
            _provider(FakeOllama()).complete_json(_req(model="qwen3.5:9b"))

    @pytest.mark.parametrize("content", ["not json", "[1, 2]", "   "])
    def test_invalid_output_is_an_output_error(self, content: str) -> None:
        with pytest.raises(LLMOutputError):
            _provider(FakeOllama(answer=lambda b: content)).complete_json(_req())

    def test_cut_off_output_is_not_a_result(self) -> None:
        fake = FakeOllama(answer=lambda b: {"a": "x"})
        fake.chat_override = {"done_reason": "length"}
        with pytest.raises(LLMOutputError, match="cut off"):
            _provider(fake).complete_json(_req())

    def test_context_overflow_is_refused_before_sending(self) -> None:
        fake = FakeOllama()
        with pytest.raises(LLMContextOverflow):
            _provider(fake, num_ctx=4096).complete_json(_req(user="word " * 20000))
        assert fake.chat_bodies == []  # never silently truncated by the server

    def test_server_reported_context_exhaustion_is_an_error(self) -> None:
        fake = FakeOllama(answer=lambda b: {"a": "x"})
        fake.chat_override = {"prompt_eval_count": 8000, "eval_count": 192}
        with pytest.raises(LLMContextOverflow, match="possibly truncated"):
            _provider(fake, num_ctx=8192).complete_json(_req())

    def test_timeout_is_transient_and_retry_recovers(self) -> None:
        fake = FakeOllama(answer=lambda b: {"a": "x"})
        fake.timeout = True
        p = _provider(fake)
        with pytest.raises(LLMTransientError, match="timed out"):
            p.complete_json(_req())

        class Recovering:
            name = "ollama"
            n = 0

            def complete_json(self, request: LLMRequest) -> Any:
                self.n += 1
                fake.timeout = self.n < 2
                return p.complete_json(request)

        r = RetryingProvider(Recovering(), max_retries=2, base_seconds=0.0).complete_json(_req())
        assert r.data == {"a": "x"}

    @pytest.mark.parametrize("status", [500, 503])
    def test_server_errors_are_transient(self, status: int) -> None:
        fake = FakeOllama()
        fake.chat_status = status
        with pytest.raises(LLMTransientError):
            _provider(fake).complete_json(_req())


class TestLocalOnly:
    @pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "qwen3.5:cloud"])
    def test_cloud_models_are_refused(self, model: str) -> None:
        with pytest.raises(LLMConfigurationError, match="cloud"):
            _provider(FakeOllama()).complete_json(_req(model=model))

    def test_remotely_served_model_is_refused(self) -> None:
        fake = FakeOllama()
        fake.show_extra = {"remote_host": "https://ollama.com:443"}
        with pytest.raises(LLMConfigurationError, match="remotely"):
            _provider(fake).model_info(MODEL)

    @pytest.mark.parametrize(
        "url",
        ["http://localhost:11434", "http://127.0.0.1:11434", "http://10.0.0.5:11434", "http://[::1]:11434"],
    )
    def test_local_and_private_urls_are_allowed(self, url: str) -> None:
        check_local_url(url, allow_remote=False)

    def test_public_url_needs_explicit_opt_in(self) -> None:
        with pytest.raises(LLMConfigurationError, match="not local"):
            OllamaProvider("https://inference.example.com")
        check_local_url("https://inference.example.com", allow_remote=True)


# ------------------------------------------------------------------------------ configuration


class TestSettings:
    def test_local_defaults_follow_ollama_model(self, tmp_path: Path) -> None:
        s = make_settings(tmp_path, llm_provider="ollama", ollama_model="qwen3.5:9b")
        assert s.llm_model_stage_b == s.llm_model_stage_c == "qwen3.5:9b"
        assert s.llm_max_concurrency == 1 and s.llm_max_output_tokens == s.ollama_output_reserve_tokens

    def test_prompts_are_sized_to_the_context_window(self, tmp_path: Path) -> None:
        s = make_settings(tmp_path, llm_provider="ollama", ollama_num_ctx=16384)
        assert s.ollama_output_reserve_tokens == 4096
        room = 16384 - 4096 - LOCAL_PROMPT_OVERHEAD_TOKENS
        assert s.llm_single_pass_max_tokens == room * 3 // 4 < 60_000
        assert s.llm_chunk_target_tokens <= s.llm_single_pass_max_tokens
        big = make_settings(tmp_path, llm_provider="ollama", ollama_num_ctx=131072)
        assert big.llm_single_pass_max_tokens == 60_000  # the configured cap still applies

    def test_explicit_stage_model_is_kept(self, tmp_path: Path) -> None:
        s = make_settings(tmp_path, llm_provider="ollama", llm_model_stage_c="qwen3.5:27b")
        assert s.llm_model_stage_b == "qwen3.5:4b" and s.llm_model_stage_c == "qwen3.5:27b"

    def test_too_small_window_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="raise the context window"):
            make_settings(tmp_path, llm_provider="ollama", ollama_num_ctx=4096)
        assert (
            make_settings(tmp_path, llm_provider="ollama", ollama_num_ctx=8192).ollama_output_reserve_tokens
            == 2048
        )

    def test_hosted_providers_are_unaffected(self, tmp_path: Path) -> None:
        s = make_settings(tmp_path, llm_provider="anthropic")
        assert s.llm_model_stage_b == "claude-opus-5-5" and s.llm_single_pass_max_tokens == 60_000

    def test_factory_builds_the_local_provider(self, tmp_path: Path) -> None:
        s = make_settings(tmp_path, llm_provider="ollama")
        chain = build_llm_provider(s)
        assert chain is not None and chain.name == "ollama"

    def test_local_models_cost_nothing_and_hosted_ones_still_do(self) -> None:
        register_local_model("my-local:1b")
        assert is_local_model("ollama:anything@x:Q4") and cost_usd("my-local:1b", 10**6, 10**6) == 0.0
        assert cost_usd(f"ollama:{MODEL}@abc:Q4_K_M", 10**6, 10**6) == 0.0
        assert cost_usd("claude-opus-5-5", 10**6, 0) == 4.0


# ------------------------------------------------------------------------------ hardware


def _hw(ram: float = 15.0, gpus: list[GPU] | None = None, disk: float = 100.0) -> Hardware:
    return Hardware("Linux", "x86_64", "cpu", 4, [], ram, ram, 0, gpus or [], disk, "/", None)


class TestHardware:
    def test_detects_this_machine(self) -> None:
        hw = detect_hardware()
        assert hw.cpu_logical >= 1 and hw.ram_total_gb > 0 and hw.disk_free_gb > 0
        assert hw.accelerator in ("cpu", "nvidia", "amd", "apple")

    def test_cpu_only_prefers_the_fastest_small_model(self) -> None:
        r = recommend(_hw(15.0))
        assert r["primary"] == "qwen3.5:4b" and "qwen3.5:27b" not in r["benchmark_candidates"]
        assert all(row["estimate"] for row in r["rows"])

    def test_gpu_prefers_the_largest_model_that_fits_vram(self) -> None:
        r = recommend(_hw(64.0, [GPU("nvidia", "RTX", 24.0, 24.0, "570", "CUDA 12.8")]))
        assert r["primary"] == "qwen3.5:27b" and r["judged_against"] == "nvidia VRAM"

    def test_amd_and_apple_are_supported(self) -> None:
        amd = recommend(_hw(32.0, [GPU("amd", "RX 7900", 24.0, 24.0, None, "ROCm 6.4")]))
        apple = recommend(_hw(64.0, [GPU("apple", "M3 Max", 42.7, None, None, "Metal")]))
        assert amd["primary"] and apple["primary"] == "qwen3.5:35b-a3b"

    def test_disk_limits_are_respected(self) -> None:
        r = recommend(_hw(64.0, [GPU("nvidia", "X", 80.0, 80.0)], disk=4.0))
        assert r["primary"] == "qwen3.5:4b"

    def test_nothing_fits(self) -> None:
        assert recommend(_hw(4.0))["primary"] is None

    def test_estimates_scale_with_quantization(self) -> None:
        c = CANDIDATES[0]
        assert estimate_memory_gb(c, "q8_0")["weights_gb"] > estimate_memory_gb(c, "q4_K_M")["weights_gb"]


class TestPull:
    def test_download_needs_size_approval(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeOllama()
        monkeypatch.setattr("avd.llm.local.provider_from_settings", lambda s: _provider(fake))
        s = make_settings(tmp_path)
        with pytest.raises(LLMConfigurationError, match="does not cover"):
            pull(s, "qwen3.5:9b", approve_gb=1, estimate_gb=5.5)
        with pytest.raises(LLMConfigurationError, match="size is unknown"):
            pull(s, "mystery:70b", approve_gb=100, estimate_gb=None)
        with pytest.raises(LLMConfigurationError, match="cloud"):
            pull(s, "x:cloud", approve_gb=100, estimate_gb=1)
        assert fake.pulled == []
        assert pull(s, "qwen3.5:4b", approve_gb=3, estimate_gb=2.4) == {"status": "success"}
        assert fake.pulled == ["qwen3.5:4b"]


# ------------------------------------------------------------------------------ evaluation path


def _local_settings(tmp_path: Path, **kw: Any) -> Settings:
    return make_settings(tmp_path, llm_provider="ollama", **kw)


def _cfg(tmp_path: Path, manifest: tuple[Path, Path], **kw: Any) -> er.RunConfig:  # noqa: F811
    path, cache = manifest
    kw.setdefault("run_id", "local")
    kw.setdefault("out_dir", tmp_path / "run")
    return er.RunConfig(manifest=path, cache_dir=cache, offline=True, **kw)


class TestEvaluateRealLocal:
    def test_end_to_end_through_the_protocol_mock(self, tmp_path: Path, manifest: tuple[Path, Path]) -> None:  # noqa: F811
        fake = FakeOllama()
        r = er.run(_cfg(tmp_path, manifest), _local_settings(tmp_path), _provider(fake))
        s, m = r["summary"], r["meta"]
        assert m["provider"] == "ollama" and m["budget_usd"] is None  # no budget needed for $0 calls
        assert m["local_runtime"]["models"][MODEL]["fingerprint"].startswith(f"{MODEL}@sha256:")
        assert m["local_runtime"]["hardware"]["cpu_logical"] >= 1
        assert s["evaluated_by_llm"] == 3 and s["cost_usd"] == 0.0 and s["parse_failures"] == 0
        assert s["evidence"]["verified"] == s["evidence"]["claimed"] > 0
        assert all(it["models_served"][0].startswith("ollama:") for it in r["items"])
        assert all(it["evaluation"]["scores"] for it in r["items"])
        # Every request fitted the window it was sent with.
        assert all(
            sum(len(x["content"]) for x in b["messages"]) / 3 < b["options"]["num_ctx"]
            for b in fake.chat_bodies
        )

    def test_resume_makes_no_duplicate_calls(self, tmp_path: Path, manifest: tuple[Path, Path]) -> None:  # noqa: F811
        fake = FakeOllama()
        er.run(_cfg(tmp_path, manifest), _local_settings(tmp_path), _provider(fake))
        n = len(fake.chat_bodies)
        r = er.run(_cfg(tmp_path, manifest), _local_settings(tmp_path), _provider(fake))
        assert len(fake.chat_bodies) == n and r["meta"]["resumed_items_skipped"] == 3

    @pytest.mark.parametrize("num_ctx", [8192, 16384, 32768])
    def test_long_transcripts_are_chunked_to_fit_every_window(self, tmp_path: Path, num_ctx: int) -> None:
        from avd.evaluation.pipeline import build_pipeline
        from tests.conftest import LECTURE, make_input, make_transcript

        fake = FakeOllama()
        st = _local_settings(tmp_path, ollama_num_ctx=num_ctx)
        llm = build_llm_provider(st, None, raw=_provider(fake, num_ctx=num_ctx))
        inp = make_input(make_transcript(LECTURE * 40))  # ~1200 segments, ~35k tokens: no window holds it
        res = build_pipeline("B", st, llm).run(inp)
        chunk_calls = [b for b in fake.chat_bodies if "judgments" not in b["format"]["properties"]]
        assert len(chunk_calls) >= 2 and len(fake.chat_bodies) == len(chunk_calls) + 1  # map + one reduce
        assert res.status in ("complete", "partial") and any(
            d.status.value == "scored" for d in res.dimensions
        )
        # Never truncated: every prompt (at 3 chars/token) fitted with the output reserve.
        for b in fake.chat_bodies:
            prompt = sum(len(x["content"]) for x in b["messages"]) / 3
            assert prompt + b["options"]["num_predict"] <= num_ctx + 1

    def test_server_down_refuses_before_running(self, tmp_path: Path, manifest: tuple[Path, Path]) -> None:  # noqa: F811
        fake = FakeOllama()
        fake.down = True
        with pytest.raises(er.PreflightError, match="local model not usable"):
            er.run(_cfg(tmp_path, manifest), _local_settings(tmp_path), _provider(fake))

    def test_missing_model_refuses_before_running(self, tmp_path: Path, manifest: tuple[Path, Path]) -> None:  # noqa: F811
        with pytest.raises(er.PreflightError, match="ollama pull"):
            er.run(
                _cfg(tmp_path, manifest),
                _local_settings(tmp_path, ollama_model="qwen3.5:9b"),
                _provider(FakeOllama()),
            )

    def test_provider_flag_revalidates_settings(self, tmp_path: Path) -> None:
        base = make_settings(tmp_path, llm_provider="anthropic")
        cfg = er.RunConfig(manifest=tmp_path / "m.json", provider="ollama")
        st = er._settings(cfg, base, tmp_path)
        assert st.llm_provider == LLMProviderName.OLLAMA and st.llm_model_stage_b == "qwen3.5:4b"
        assert st.llm_single_pass_max_tokens < 60_000

    def test_popularity_never_reaches_the_model(self, tmp_path: Path, manifest: tuple[Path, Path]) -> None:  # noqa: F811
        fake = FakeOllama()
        er.run(_cfg(tmp_path, manifest), _local_settings(tmp_path), _provider(fake))
        text = json.dumps(fake.chat_bodies).lower()
        assert "view_count" not in text and "like_count" not in text and "subscriber" not in text


class TestOllamaDownWorker:
    def test_stage_b_job_is_deferred_while_ollama_is_down(self, tmp_path: Path) -> None:
        from sqlalchemy import func, select

        from avd.db.models import Evaluation
        from avd.services import seed_demo
        from avd.workers import queue
        from avd.workers.context import build_context
        from avd.workers.runner import execute_job, run_until_idle
        from tests.conftest import _make_db

        db = _make_db(tmp_path, None)
        fake = FakeOllama()
        fake.down = True
        settings = db.settings.model_copy(update={"llm_provider": LLMProviderName.OLLAMA})
        llm = build_llm_provider(settings, db.sf, raw=_provider(fake))
        with db.sf() as s:
            seed_demo(s, db.settings)
            s.commit()
        ctx = build_context(settings=settings, session_factory=db.sf, llm=llm)
        run_until_idle(ctx, job_types=["stage_a"])
        with db.sf() as s:
            jobs = queue.claim(s, limit=1, lease_seconds=60, job_types=["stage_b"])
            s.commit()
        assert execute_job(ctx, jobs[0].id)["outcome"] == "retry"
        with db.sf() as s:
            assert s.scalar(select(func.count()).select_from(Evaluation)) == 0


# ------------------------------------------------------------------------------ bench-local


class TestBenchLocal:
    def test_report_is_labelled_as_injected(self, tmp_path: Path, manifest: tuple[Path, Path]) -> None:  # noqa: F811
        path, cache = manifest
        fake = FakeOllama()
        cfg = bl.LocalBenchConfig(
            manifest=path, models=[MODEL], cache_dir=cache, offline=True, out_dir=tmp_path / "b", repeats=2
        )
        r = bl.bench_local(
            cfg, make_settings(tmp_path), lambda st: _provider(fake, num_ctx=st.ollama_num_ctx)
        )
        assert "NOT real inference" in r["meta"]["transport"]
        w = r["models"][MODEL]["windows"]["16384"]
        assert w["structured_validity"] == 1.0 and w["citation_correctness"] == 1.0 and w["cost_usd"] == 0.0
        assert (
            w["consistency"]["exact_agreement"] == 1.0 and w["consistency"]["pairwise_order_stability"] == 1.0
        )
        p = r["models"][MODEL]["probes"]
        assert p["injection"]["cited_injected_segment"] is False and p["abstention"]["text_dimensions"] > 0
        assert r["gates"]["validity"].startswith("blocked")
        md = (tmp_path / "b" / "report.md").read_text()
        assert "NOT real inference" in md and "agreement with people needs human labels" in md

    def test_refuses_held_out_data(self, tmp_path: Path, manifest: tuple[Path, Path]) -> None:  # noqa: F811
        cfg = bl.LocalBenchConfig(manifest=manifest[0], models=[MODEL], split="test")
        with pytest.raises(er.PreflightError, match="development data only"):
            bl.bench_local(cfg, make_settings(tmp_path))

    def test_missing_model_runs_nothing(self, tmp_path: Path, manifest: tuple[Path, Path]) -> None:  # noqa: F811
        path, cache = manifest
        cfg = bl.LocalBenchConfig(
            manifest=path, models=["qwen3.5:9b"], cache_dir=cache, out_dir=tmp_path / "b"
        )
        with pytest.raises(er.PreflightError, match="No local inference was run"):
            bl.bench_local(cfg, make_settings(tmp_path), lambda st: _provider(FakeOllama()))
        assert not (tmp_path / "b" / "report.json").exists()

    def test_consistency_detects_disagreement(self) -> None:
        def run(a: float, b: float) -> dict[str, Any]:
            return {
                "items": [
                    {"video_id": "x", "evaluation": {"scores": {"clarity": a}, "overall": a}},
                    {"video_id": "y", "evaluation": {"scores": {"clarity": b}, "overall": b}},
                ]
            }

        c = bl._consistency([run(3, 4), run(4, 3)])
        assert (
            c["exact_agreement"] == 0.0 and c["mean_abs_diff"] == 1.0 and c["pairwise_order_stability"] == 0.0
        )


# ------------------------------------------------------------------------------ live (optional)

LIVE_URL = os.environ.get("AVD_TEST_OLLAMA_URL")
LIVE_MODEL = os.environ.get("AVD_TEST_OLLAMA_MODEL")


def _live_ok() -> bool:
    if not (LIVE_URL and LIVE_MODEL):
        return False
    try:
        return httpx.get(f"{LIVE_URL.rstrip('/')}/api/version", timeout=2, trust_env=False).status_code == 200
    except httpx.HTTPError:
        return False


live = pytest.mark.skipif(
    not _live_ok(), reason="no reachable Ollama (AVD_TEST_OLLAMA_URL/MODEL): live test skipped"
)


@pytest.mark.live
@live
def test_live_structured_output_from_a_real_local_model() -> None:
    assert LIVE_URL and LIVE_MODEL
    p = OllamaProvider(LIVE_URL, num_ctx=8192, timeout=600)
    r = p.complete_json(
        _req(
            user="Segment [0]: 'Water boils at 100 C at sea level.' Return the segment index you read.",
            json_schema={
                "type": "object",
                "properties": {"segment_index": {"type": "integer"}},
                "required": ["segment_index"],
            },
            model=LIVE_MODEL,
            max_tokens=64,
        )
    )
    assert r.data.get("segment_index") == 0 and r.served_model.startswith(f"ollama:{LIVE_MODEL}@")
