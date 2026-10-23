from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FIXTURES, PG_URL


def _clear_caches() -> None:
    from avd.config import get_settings
    from avd.db.session import get_engine, get_sessionmaker
    from avd.embeddings.base import get_embedder

    for f in (get_settings, get_engine, get_sessionmaker, get_embedder):
        f.cache_clear()


@pytest.fixture(scope="module", params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def client(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    if request.param == "postgres" and not PG_URL:
        pytest.skip("AVD_TEST_DATABASE_URL not set")
    tmp: Path = tmp_path_factory.mktemp("api")
    url = PG_URL if request.param == "postgres" else f"sqlite:///{tmp / 'api.db'}"
    mp = pytest.MonkeyPatch()
    mp.setenv("AVD_DATABASE_URL", url or "")
    mp.setenv("AVD_ADMIN_TOKEN", "test-admin")
    mp.setenv("AVD_MEDIA_CACHE_DIR", str(tmp / "media"))
    mp.setenv("AVD_FIXTURES_DIR", str(FIXTURES))
    mp.setenv("AVD_LLM_PROVIDER", "heuristic")
    mp.delenv("AVD_REDIS_URL", raising=False)
    _clear_caches()
    from sqlalchemy import text

    from avd.db.models import Base
    from avd.db.session import get_engine

    engine = get_engine()
    if engine.dialect.name == "postgresql":
        with engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    from avd.api.app import create_app

    with TestClient(create_app()) as c:
        admin = {"X-Admin-Token": "test-admin"}
        assert c.post("/v1/admin/seed-demo", headers=admin).json()["seeded"] == 20
        assert c.post("/v1/admin/worker/run", headers=admin).json()["outcomes"] == {"succeeded": 40}
        yield c
    engine.dispose()
    mp.undo()
    _clear_caches()


def _user(client: TestClient) -> dict[str, str]:
    r = client.post("/v1/users")
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_health_and_meta(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    meta = client.get("/v1/meta").json()
    assert meta["demo_mode"] is True and meta["evaluation_mode"] == "heuristic_baseline"
    assert "No AI evaluator is configured" in meta["disclaimer"]
    assert "not been validated against human judgement" in meta["disclaimer"]
    assert len(meta["dimensions"]) == 14 and "science" in meta["genre_profiles"]


def test_evaluation_breakdown_is_labelled_synthetic(client: TestClient) -> None:
    ev = client.get("/v1/videos/syn_edu_001/evaluation").json()
    assert ev["synthetic"] is True and ev["mode"] == "synthetic_demo"
    assert {d["status"] for d in ev["dimensions"]} >= {"scored", "insufficient_evidence", "requires_media"}
    audio = next(d for d in ev["dimensions"] if d["dimension"] == "audio_quality")
    assert audio["status"] == "scored" and audio["score"] <= 2  # noisy, clipped synthetic audio
    assert ev["aggregation_profile"]["genre"] == ev["genre"]
    assert abs(sum(c["contribution"] for c in ev["contributions"]) - ev["overall"]["score"]) < 1e-3


def test_unknown_video_404(client: TestClient) -> None:
    assert client.get("/v1/videos/doesnotexist1/evaluation").status_code == 404
    assert client.get("/v1/videos/doesnotexist1").status_code == 404


def test_video_detail_exposes_provenance_not_popularity(client: TestClient) -> None:
    body = client.get("/v1/videos/syn_sci_001").json()
    assert body["content_sources"][0]["source_type"] == "synthetic_fixture"
    assert body["content_sources"][0]["license"] == "CC0-1.0"
    flat = str(body).lower()
    assert "view_count" not in flat and "subscriber" not in flat


def test_search(client: TestClient) -> None:
    items = client.get("/v1/search", params={"q": "hash table resizing"}).json()["items"]
    assert items and items[0]["video"]["id"] == "syn_tech_001"
    assert {"semantic_similarity", "title_match", "score"} <= set(items[0]["match"])


def test_search_without_a_match_is_empty_not_the_whole_catalog(client: TestClient) -> None:
    # Before the relevance floor, every query (even this one) returned all 20 catalog videos.
    assert client.get("/v1/search", params={"q": "quantum chromodynamics"}).json()["items"] == []
    items = client.get("/v1/search", params={"q": "hash table resizing"}).json()["items"]
    assert len(items) < 5 and items[0]["video"]["id"] == "syn_tech_001"


def test_preferences_validation(client: TestClient) -> None:
    h = _user(client)
    assert (
        client.put("/v1/users/me/preferences", headers=h, json={"genre_weights": {"cooking": 1}}).status_code
        == 422
    )
    assert (
        client.put(
            "/v1/users/me/preferences", headers=h, json={"dimension_weights": {"depth": 9}}
        ).status_code
        == 422
    )
    ok = client.put(
        "/v1/users/me/preferences", headers=h, json={"interests": ["astronomy", "astronomy", " "]}
    )
    assert ok.json()["interests"] == ["astronomy"]


def test_auth_required(client: TestClient) -> None:
    assert client.get("/v1/recommendations").status_code == 401
    assert client.get("/v1/recommendations", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/admin/seed-demo").status_code == 403


def test_recommendations_keep_three_scores_separate(client: TestClient) -> None:
    h = _user(client)
    client.put(
        "/v1/users/me/preferences",
        headers=h,
        json={
            "interests": ["physics of everyday things", "astronomy and space weather"],
            "languages": ["en"],
            "depth_preference": "intermediate",
            "dimension_weights": {"depth": 2.0},
        },
    )
    body = client.get("/v1/recommendations", headers=h, params={"limit": 8}).json()
    items = body["items"]
    assert items and body["ranker_version"] == "linear_v1"
    for it in items:
        assert {"intrinsic_quality", "relevance", "ranking"} <= set(it)
        assert set(it["ranking"]["contributions"]) == set(it["ranking"]["features"])
        assert abs(sum(it["ranking"]["contributions"].values()) - it["ranking"]["score"]) < 1e-4
        assert it["video"]["language"] == "en"
        assert "view" not in str(it["ranking"]["features"])
    channels = [it["video"]["channel_title"] for it in items]
    assert max(channels.count(c) for c in channels) <= 2  # per-channel diversity cap
    assert any(it["video"]["id"] in ("syn_sci_001", "syn_sci_002", "syn_doc_001") for it in items[:4])


def test_feedback_flow(client: TestClient) -> None:
    h = _user(client)
    client.put("/v1/users/me/preferences", headers=h, json={"interests": ["programming data structures"]})
    rec = client.get("/v1/recommendations", headers=h).json()["items"][0]
    vid = rec["video"]["id"]
    ok = client.post(
        "/v1/feedback",
        headers=h,
        json={"video_id": vid, "kind": "recommendation_useful", "impression_id": rec["impression_id"]},
    )
    assert ok.status_code == 201
    ev_id = rec["intrinsic_quality"]["evaluation_id"]
    bad = client.post("/v1/feedback", headers=h, json={"video_id": vid, "kind": "evaluation_incorrect"})
    assert bad.status_code == 400  # must reference the evaluation being disputed
    rep = client.post(
        "/v1/feedback",
        headers=h,
        json={
            "video_id": vid,
            "kind": "evaluation_incorrect",
            "evaluation_id": ev_id,
            "dimension": "depth",
            "suggested_score": 5,
            "comment": "Covers amortised analysis in depth.",
        },
    )
    assert rep.status_code == 201
    other = _user(client)
    stolen = client.post(
        "/v1/feedback",
        headers=other,
        json={"video_id": vid, "kind": "recommendation_useful", "impression_id": rec["impression_id"]},
    )
    assert stolen.status_code == 400
    # "not interested" removes the video from future recommendations
    client.post("/v1/feedback", headers=h, json={"video_id": vid, "kind": "not_interested"})
    again = [i["video"]["id"] for i in client.get("/v1/recommendations", headers=h).json()["items"]]
    assert vid not in again


def test_transcript_submission_requires_attestation(client: TestClient) -> None:
    h = _user(client)
    body = {"format": "text", "content": "word " * 80, "attestation": "short"}
    assert client.post("/v1/videos/abcdefghijk/transcripts", headers=h, json=body).status_code == 422
    body["attestation"] = "I am the channel owner and authorise this use."
    r = client.post("/v1/videos/abcdefghijk/transcripts", headers=h, json=body)
    assert r.status_code == 202 and r.json()["created"] is True
    again = client.post("/v1/videos/abcdefghijk/transcripts", headers=h, json=body)
    assert again.json()["created"] is False


def test_admin_observability(client: TestClient) -> None:
    admin = {"X-Admin-Token": "test-admin"}
    cov = client.get("/v1/admin/coverage", headers=admin).json()
    assert sum(cov["channel_size"]["evaluated"].values()) >= 20
    costs = client.get("/v1/admin/costs", headers=admin).json()
    assert costs["today"]["llm_usd"] == 0.0
    jobs = client.get("/v1/admin/jobs", headers=admin).json()
    assert jobs["stats"]["stage_b"]["succeeded"] == 20


def test_off_topic_items_only_fill_a_labelled_exploration_quota(client: TestClient) -> None:
    h = _user(client)
    client.put("/v1/users/me/preferences", headers=h, json={"interests": ["medieval falconry techniques"]})
    items = client.get("/v1/recommendations", headers=h, params={"limit": 10}).json()["items"]
    explore = [it for it in items if it["ranking"]["exploration"]]
    topical = [it for it in items if not it["ranking"]["exploration"]]
    assert len(explore) <= 2  # exploration quota = limit // 5; the feed is not padded with off-topic items
    assert len(topical) < 5  # a nonsense interest matches (almost) nothing
    for it in explore:
        assert it["explanations"][0].startswith("Exploration pick")
        # an exploration pick is outside the interests: it must not also claim to match one
        assert not any(e.startswith("Matches your interest") for e in it["explanations"])
    for it in topical:
        assert it["ranking"]["features"]["relevance"] > 0


def test_status_reports_catalog_size(client: TestClient) -> None:
    catalog = client.get("/v1/status").json()["catalog"]
    assert catalog["videos"] >= 20 and catalog["synthetic"] >= 20 and catalog["evaluated"] >= 20


def test_non_ascii_admin_token_is_refused_not_a_server_error(client: TestClient) -> None:
    # Before: hmac.compare_digest raised TypeError on non-ASCII text -> HTTP 500 with a traceback.
    r = client.get("/v1/admin/jobs", headers={"X-Admin-Token": "caf\xe9".encode("latin-1")})
    assert r.status_code == 403
    assert client.get("/metrics", headers={"X-Admin-Token": "caf\xe9".encode("latin-1")}).status_code in (200, 403)
