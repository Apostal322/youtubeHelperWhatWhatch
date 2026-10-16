"""Phase 2 P8: request correlation, structured errors, rate limiting, metrics, status, dead
letters and account deletion."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from avd.api.ops import TokenBucket

ADMIN = {"X-Admin-Token": "test-admin"}


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Iterator[TestClient]:
    from avd.config import get_settings
    from avd.db.models import Base
    from avd.db.session import get_engine, get_sessionmaker
    from avd.embeddings.base import get_embedder

    monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{tmp_path / 'ops.db'}")
    monkeypatch.setenv("AVD_ADMIN_TOKEN", "test-admin")
    monkeypatch.setenv("AVD_MEDIA_CACHE_DIR", str(tmp_path / "media"))
    monkeypatch.delenv("AVD_REDIS_URL", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for f in (get_settings, get_engine, get_sessionmaker, get_embedder):
        f.cache_clear()
    Base.metadata.create_all(get_engine())
    from avd.api.app import create_app

    with TestClient(create_app()) as c:
        yield c
    get_engine().dispose()
    for f in (get_settings, get_engine, get_sessionmaker, get_embedder):
        f.cache_clear()


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _client(tmp_path, monkeypatch)


def test_request_ids_are_echoed_and_propagate_to_jobs(api: TestClient) -> None:
    r = api.post("/v1/admin/seed-demo", headers={**ADMIN, "X-Request-ID": "req-test-12345"})
    assert r.headers["X-Request-ID"] == "req-test-12345"
    from avd.db.models import PipelineJob
    from avd.db.session import get_sessionmaker

    with get_sessionmaker()() as s:
        corr = {j.correlation_id for j in s.scalars(select(PipelineJob)).all()}
    assert corr == {"req-test-12345"}
    unsafe = api.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert (
        unsafe.headers["X-Request-ID"] != "bad id\nwith newline" and len(unsafe.headers["X-Request-ID"]) == 16
    )


def test_errors_share_one_shape(api: TestClient) -> None:
    nf = api.get("/v1/videos/does_not_exist", headers={"X-Request-ID": "req-err-0001"})
    assert nf.status_code == 404
    body = nf.json()
    assert body["error"]["code"] == "not_found" and body["error"]["request_id"] == "req-err-0001"
    assert body["detail"]  # backward compatible field
    bad = api.put("/v1/users/me/preferences", json={}, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "unauthenticated"
    tok = api.post("/v1/users").json()["token"]
    v = api.put(
        "/v1/users/me/preferences",
        json={"min_duration_minutes": 50, "max_duration_minutes": 10},
        headers={"Authorization": f"Bearer {tok}"},
    )
    assert v.status_code == 422 and v.json()["error"]["code"] == "validation_error"
    assert isinstance(v.json()["detail"], list)


def test_token_bucket() -> None:
    b = TokenBucket(rate_per_minute=60, burst=2)
    assert b.take("k", now=0.0) == 0 and b.take("k", now=0.0) == 0
    wait = b.take("k", now=0.0)
    assert 0.9 < wait <= 1.0
    assert b.take("k", now=1.01) == 0  # refilled one token per second
    assert b.take("other", now=0.0) == 0  # buckets are per caller


def test_rate_limit_returns_429_with_retry_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _client(tmp_path, monkeypatch, AVD_API_RATE_LIMIT_PER_MINUTE="2", AVD_API_RATE_LIMIT_BURST="2")
    c = next(gen)
    try:
        codes = [c.get("/v1/meta").status_code for _ in range(4)]
        assert codes[:2] == [200, 200] and codes[2] == 429
        r = c.get("/v1/meta")
        assert r.json()["error"]["code"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1
        assert all(c.get("/health").status_code == 200 for _ in range(5))  # exempt
    finally:
        next(gen, None)


def test_metrics_and_status(api: TestClient) -> None:
    assert api.get("/metrics").status_code == 403
    api.get("/v1/meta")
    m = api.get("/metrics", headers=ADMIN)
    assert (
        m.status_code == 200 and "avd_http_requests_total" in m.text and "avd_llm_spend_today_usd" in m.text
    )
    assert 'route="/v1/meta"' in m.text
    st = api.get("/v1/status").json()
    assert (
        st["status"] == "ok" and st["database"] == "ok" and isinstance(st["llm_credentials_configured"], bool)
    )
    assert "test-admin" not in api.get("/v1/status").text


def test_dead_letters_can_be_inspected_and_requeued(api: TestClient) -> None:
    from avd.db.models import PipelineJob
    from avd.db.session import get_sessionmaker

    with get_sessionmaker()() as s:
        s.add(
            PipelineJob(
                job_type="stage_b",
                dedup_key="x",
                status="dead",
                attempts=4,
                last_error="LLMTransientError: 503",
            )
        )
        s.add(PipelineJob(job_type="stage_a", dedup_key="y", status="succeeded"))
        s.commit()
        dead_id = s.scalar(select(PipelineJob.id).where(PipelineJob.dedup_key == "x"))
        ok_id = s.scalar(select(PipelineJob.id).where(PipelineJob.dedup_key == "y"))
    d = api.get("/v1/admin/dead-letters", headers=ADMIN).json()
    assert d["total"] == 1 and d["by_type_and_error"] == {"stage_b | LLMTransientError": 1}
    assert api.post(f"/v1/admin/jobs/{ok_id}/requeue", headers=ADMIN).status_code == 409
    r = api.post(f"/v1/admin/jobs/{dead_id}/requeue", headers=ADMIN)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert api.get("/v1/admin/dead-letters", headers=ADMIN).json()["total"] == 0


def test_account_deletion_removes_user_data(api: TestClient) -> None:
    tok = api.post("/v1/users").json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert api.put("/v1/users/me/preferences", json={"interests": ["biology"]}, headers=h).status_code == 200
    sub = api.post(
        "/v1/videos/user_vid_001/transcripts",
        json={
            "format": "text",
            "content": "A short talk about photosynthesis and the Calvin cycle. " * 20,
            "attestation": "I am the creator of this video and hold the rights to its transcript.",
        },
        headers=h,
    )
    assert sub.status_code == 202
    src_id = sub.json()["content_source_id"]
    out = api.delete("/v1/users/me", headers=h).json()
    assert out["deleted"] and out["transcripts_revoked"] == 1 and out["preferences"] == 1
    assert api.get("/v1/users/me/preferences", headers=h).status_code == 401
    from avd.db.models import ContentSource
    from avd.db.session import get_sessionmaker

    with get_sessionmaker()() as s:
        src = s.get(ContentSource, src_id)
        assert (
            src is not None
            and src.status == "revoked"
            and src.transcript is None
            and src.provided_by_user_id is None
        )
