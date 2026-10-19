"""Phase 3: shared API rate limiting — multiple processes/instances, token rotation, Redis failure."""

from __future__ import annotations

import multiprocessing as mp
import re
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from avd.api.ops import RateLimiterPolicy, RedisTokenBucket
from avd.observability import METRICS
from tests.conftest import REDIS_URL, make_settings

needs_redis = pytest.mark.skipif(not REDIS_URL, reason="AVD_TEST_REDIS_URL not set")


def _hammer(url: str, prefix: str, n: int, out: mp.Queue) -> None:  # type: ignore[type-arg]
    bucket = RedisTokenBucket(url, rate_per_minute=1, burst=10, prefix=prefix)
    out.put(sum(1 for _ in range(n) if bucket.take("same-caller") == 0))


@needs_redis
def test_limit_is_shared_across_processes() -> None:
    prefix = uuid.uuid4().hex
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    procs = [ctx.Process(target=_hammer, args=(REDIS_URL, prefix, 8, q)) for _ in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    allowed = sum(q.get(timeout=5) for _ in procs)
    assert allowed == 10  # 24 attempts from 3 processes, one shared burst of 10


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Iterator[TestClient]:
    from avd.config import get_settings
    from avd.db.models import Base
    from avd.db.session import get_engine, get_sessionmaker

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{tmp_path / 'rl.db'}")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for f in (get_settings, get_engine, get_sessionmaker):
        f.cache_clear()
    Base.metadata.create_all(get_engine())
    from avd.api.app import create_app

    with TestClient(create_app()) as c:
        yield c
    get_engine().dispose()
    for f in (get_settings, get_engine, get_sessionmaker):
        f.cache_clear()


def test_rotating_tokens_cannot_multiply_the_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AVD_REDIS_URL", raising=False)
    gen = _app(
        tmp_path,
        monkeypatch,
        AVD_API_RATE_LIMIT_PER_MINUTE="60",
        AVD_API_RATE_LIMIT_BURST="5",
        AVD_API_RATE_LIMIT_PER_IP_PER_MINUTE="60",
        AVD_API_RATE_LIMIT_IP_BURST="8",
    )
    c = next(gen)
    codes = [c.get("/v1/meta", headers={"Authorization": f"Bearer fake-{i}"}).status_code for i in range(12)]
    assert codes[:8] == [200] * 8 and set(codes[8:]) == {429}
    r = c.get("/v1/meta", headers={"Authorization": "Bearer another"})
    assert (
        r.status_code == 429
        and r.json()["error"]["code"] == "rate_limited"
        and int(r.headers["Retry-After"]) >= 1
    )
    gen.close()


@needs_redis
def test_two_api_instances_share_one_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from redis import Redis

    Redis.from_url(REDIS_URL).flushdb()
    env = {
        "AVD_REDIS_URL": REDIS_URL or "",
        "AVD_API_RATE_LIMIT_PER_MINUTE": "1",
        "AVD_API_RATE_LIMIT_BURST": "4",
        "AVD_API_RATE_LIMIT_PER_IP_PER_MINUTE": "600",
        "AVD_API_RATE_LIMIT_IP_BURST": "100",
    }
    a = _app(tmp_path / "a", monkeypatch, **env)
    ca = next(a)
    b = _app(tmp_path / "b", monkeypatch, **env)
    cb = next(b)
    h = {"Authorization": "Bearer same-user"}
    codes = [(ca if i % 2 else cb).get("/v1/meta", headers=h).status_code for i in range(8)]
    assert codes.count(200) == 4 and codes.count(429) == 4  # per-process limits would allow 8
    a.close()
    b.close()


def test_redis_failure_falls_back_to_local_limits_visibly(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        redis_url="redis://127.0.0.1:1/0",  # nothing listens here
        api_rate_limit_per_minute=60,
        api_rate_limit_burst=2,
    )
    policy = RateLimiterPolicy(settings)
    waits = [policy.check("t:x", "ip:1") for _ in range(3)]
    assert waits[:2] == [0.0, 0.0] and waits[2] > 0  # still limiting
    assert re.search(r"^avd_rate_limiter_fallback_total\S* [1-9]", METRICS.render({}), re.M)
