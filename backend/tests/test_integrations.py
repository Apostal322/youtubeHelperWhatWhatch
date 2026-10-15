"""YouTube Data API client, captions provider, parsing and rate limiting — all offline."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from avd.content.models import SourceType
from avd.content.parsing import parse_timed_text, transcript_from_plain_text
from avd.content.providers import CreatorCaptionsProvider
from avd.db.models import ContentSource, Video
from avd.evaluation.schema import Modality
from avd.ingest import upsert_channel
from avd.llm.ratelimit import InMemoryTokenBucket
from avd.youtube.client import (
    QuotaExceeded,
    YouTubeAPIError,
    YouTubeCaptionsClient,
    YouTubeDataClient,
    parse_iso_duration,
)
from tests.conftest import DB, REDIS_URL

VTT = """WEBVTT

00:00:00.000 --> 00:00:04.200
<c>Welcome to the lecture on entropy.</c>

00:00:04.200 --> 00:00:09.000
Entropy measures the number of microstates.

00:00:09.000 --> 00:00:12.500
Entropy measures the number of microstates.
"""

SRT = """1
00:00:01,000 --> 00:00:03,500
First line.

2
00:01:02,250 --> 00:01:05,000
Second line.
"""


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, op: str, units: int) -> None:
        self.calls.append((op, units))


def transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_parse_vtt_strips_tags_and_dedupes_rolling_captions() -> None:
    t = parse_timed_text(VTT, "en")
    assert [s.text for s in t.segments] == [
        "Welcome to the lecture on entropy.",
        "Entropy measures the number of microstates.",
    ]
    assert t.segments[1].start == 4.2


def test_parse_srt() -> None:
    t = parse_timed_text(SRT)
    assert t.segments[1].start == pytest.approx(62.25)


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_timed_text("not captions at all")


def test_plain_text_gets_estimated_timings() -> None:
    t = transcript_from_plain_text("One sentence here. " * 30)
    assert len(t.segments) > 1 and t.segments[-1].end > t.segments[0].end


def test_iso_duration() -> None:
    assert parse_iso_duration("PT1H2M3S") == 3723
    assert parse_iso_duration("P1DT1S") == 86401
    assert parse_iso_duration(None) is None


def test_search_is_popularity_blind_and_charges_quota() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(dict(req.url.params))
        return httpx.Response(
            200, json={"items": [{"id": {"videoId": "abcdefghijk"}}], "nextPageToken": "T2"}
        )

    guard = Recorder()
    yt = YouTubeDataClient("KEY", guard, transport=transport(handler))
    ids, token = yt.search("entropy lecture", language="en")
    assert ids == ["abcdefghijk"] and token == "T2"
    assert seen["order"] == "date" and seen["relevanceLanguage"] == "en"
    assert guard.calls == [("search.list", 100)]
    with pytest.raises(ValueError):
        yt.search("x", order="viewCount")


def test_videos_batches_50_ids_per_call() -> None:
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        ids = req.url.params["id"].split(",")
        calls.append(len(ids))
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": i,
                        "snippet": {
                            "channelId": "UC1",
                            "title": f"t{i}",
                            "publishedAt": "2025-01-01T00:00:00Z",
                            "defaultAudioLanguage": "en",
                            "categoryId": "27",
                        },
                        "contentDetails": {"duration": "PT10M", "caption": "true"},
                        "statistics": {"viewCount": "12"},
                        "status": {"embeddable": True},
                    }
                    for i in ids
                ]
            },
        )

    guard = Recorder()
    out = YouTubeDataClient("KEY", guard, transport=transport(handler)).videos(
        [f"v{i:010d}" for i in range(120)]
    )
    assert calls == [50, 50, 20] and len(out) == 120
    assert out[0].duration_seconds == 600 and out[0].caption_flag is True
    assert guard.calls == [("videos.list", 1)] * 3


def test_quota_errors_are_mapped() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"errors": [{"reason": "quotaExceeded"}]}})

    with pytest.raises(QuotaExceeded):
        YouTubeDataClient("K", Recorder(), transport=transport(handler)).videos(["a"])

    def handler503(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    with pytest.raises(YouTubeAPIError) as e:
        YouTubeDataClient("K", Recorder(), transport=transport(handler503)).videos(["a"])
    assert e.value.retryable


def test_guard_blocks_call_before_network() -> None:
    def guard(op: str, units: int) -> None:
        raise QuotaExceeded("budget")

    def handler(req: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be reached
        raise AssertionError("network called despite exhausted quota")

    with pytest.raises(QuotaExceeded):
        YouTubeDataClient("K", guard, transport=transport(handler)).search("x")


class Tokens:
    def access_token_for_channel(self, channel_id: str) -> str | None:
        return "creator-token" if channel_id == "UC_creator" else None


def test_creator_captions_provider_only_for_authorized_channels(db: DB) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["Authorization"] == "Bearer creator-token"
        if req.url.path.endswith("/captions"):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": "asr1", "snippet": {"trackKind": "asr", "language": "en"}},
                        {"id": "std1", "snippet": {"trackKind": "standard", "language": "en"}},
                    ]
                },
            )
        assert req.url.path.endswith("/captions/std1")  # creator-uploaded track preferred over ASR
        return httpx.Response(200, text=VTT)

    guard = Recorder()
    provider = CreatorCaptionsProvider(YouTubeCaptionsClient(guard, transport=transport(handler)), Tokens())
    with db.sf() as s:
        upsert_channel(s, "UC_creator", "Creator")
        upsert_channel(s, "UC_other", "Other")
        mine = Video(id="creatorvid1", channel_id="UC_creator", title="Entropy", language="en")
        theirs = Video(id="othervid001", channel_id="UC_other", title="Other")
        s.add_all([mine, theirs])
        s.flush()
        assert provider.check(s, theirs) == set()
        assert provider.check(s, mine) == {Modality.TRANSCRIPT}
        assert provider.acquire(s, mine) == 1
        src = s.query(ContentSource).one()
        assert (
            src.source_type == SourceType.CREATOR_AUTHORIZED_CAPTIONS.value
            and "OAuth" in src.authorization_basis
        )
    assert ("captions.download", 200) in guard.calls


def test_in_memory_token_bucket() -> None:
    b = InMemoryTokenBucket(rate_per_minute=60, burst=2)
    assert b.acquire("k", timeout=0) and b.acquire("k", timeout=0)
    assert not b.acquire("k", timeout=0)


@pytest.mark.redis
def test_redis_token_bucket_shared_state() -> None:
    if not REDIS_URL:
        pytest.skip("AVD_TEST_REDIS_URL not set")
    import redis

    from avd.llm.ratelimit import RedisTokenBucket

    redis.Redis.from_url(REDIS_URL).delete("avd:rl:test-key")
    a = RedisTokenBucket(REDIS_URL, rate_per_minute=60, burst=2)
    b = RedisTokenBucket(REDIS_URL, rate_per_minute=60, burst=2)  # e.g. another worker process
    assert a.acquire("test-key", timeout=0) and b.acquire("test-key", timeout=0)
    assert not a.acquire("test-key", timeout=0)


def test_anthropic_provider_maps_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Anthropic provider sends a JSON-schema output config and maps refusals/truncation."""
    pytest.importorskip("anthropic")  # optional extra: pip install -e ".[anthropic]"
    from types import SimpleNamespace

    from avd.llm.base import LLMOutputError, LLMRefusalError, LLMRequest
    from avd.llm.providers import AnthropicProvider

    sent: dict[str, Any] = {}

    def make(stop: str, text: str) -> Any:
        return SimpleNamespace(
            stop_reason=stop,
            stop_details=SimpleNamespace(category="cyber"),
            model="claude-opus-5-5",
            id="msg_1",
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

    p = AnthropicProvider(api_key="test-key")
    responses = iter([make("end_turn", json.dumps({"ok": 1})), make("refusal", ""), make("max_tokens", "{")])

    def create(**kw: Any) -> Any:
        sent.update(kw)
        return next(responses)

    monkeypatch.setattr(p._client.beta.messages, "create", create)
    req = LLMRequest(
        system="s",
        user="u",
        json_schema={"type": "object"},
        schema_name="x",
        model="claude-opus-5-5",
        effort="medium",
    )
    r = p.complete_json(req)
    assert r.data == {"ok": 1} and r.served_model == "claude-opus-5-5"
    assert (
        sent["output_config"]["format"]["type"] == "json_schema"
        and sent["output_config"]["effort"] == "medium"
    )
    assert sent["fallbacks"] == "default"
    with pytest.raises(LLMRefusalError):
        p.complete_json(req)
    with pytest.raises(LLMOutputError):
        p.complete_json(req)


def test_openai_compatible_provider() -> None:
    from avd.llm.base import LLMRequest
    from avd.llm.providers import OpenAICompatibleProvider

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(
            200,
            json={
                "model": "local-model",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"a": 2}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    p = OpenAICompatibleProvider("http://llm.local/v1", None, transport=transport(handler))
    r = p.complete_json(
        LLMRequest(system="s", user="u", json_schema={}, schema_name="x", model="local-model")
    )
    assert r.data == {"a": 2} and r.input_tokens == 7
