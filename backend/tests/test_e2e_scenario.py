"""End-to-end scenario across the Phase 2 surfaces, through the real CLI and HTTP API:

licensed import (checksums, rights) → queued evaluation → public evaluation with its assessment
state → manual genre correction re-runs the evaluation → blind human annotation and export →
personalised recommendations with factual explanations → feedback → takedown → account deletion.

The texts below are written for this test in the sources' *formats*; they are not copied from the
licensed transcripts, and the annotation submitted here is a test input, not a label.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from avd.datasets.pilot import describe_item, finalize_manifest
from avd.datasets.sources import CHANGELOG, TALKPYTHON, LicensedSource
from tests.conftest import PG_URL

ADMIN = {"X-Admin-Token": "e2e-admin"}

CACHING_EPISODE = """**Host:** Welcome back. Today we are talking about caching layers in web services, and why \
they fail in production more often than people expect.

**Guest:** Thanks for having me. I have spent the last five years debugging cache problems.

**Host:** \\[00:01:30.00\\] Let's start with the basics. What does a cache actually buy you?

**Guest:** A cache stores the result of an expensive computation so that the next request can reuse \
it. For example, if rendering a product page takes two hundred milliseconds, a cache can serve it in \
two. The trade-off is that the cached copy can become stale, which means users see old data.

**Host:** \\[00:04:10.00\\] So invalidation is the hard part.

**Guest:** Exactly. There are three common strategies. First, time-based expiry: every entry lives for a \
fixed duration. Second, explicit invalidation: when the data changes, you delete the entry. Third, \
versioned keys: the key includes a version number, so a change simply creates a new key. However, \
each has costs. Expiry serves stale data until the timer runs out, and explicit invalidation breaks \
when a code path forgets to delete.

**Host:** \\[00:09:45.00\\] What about caching errors?

**Guest:** That is the second big mistake. If the database times out and you cache the error page, \
every user gets the error until the entry expires. In summary, cache successes, measure your hit \
rate, and decide on an invalidation strategy before you add the cache, not after the first incident.

**Host:** \\[00:14:20.00\\] Great advice. Thanks for coming on.
"""

TRACING_EPISODE = CACHING_EPISODE.replace("caching layers in web services", "distributed tracing").replace(
    "cache", "trace span"
)


def _vtt(lines: list[str], seconds: float = 6.0) -> str:
    def ts(t: float) -> str:
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"

    cues = [f"{ts(i * seconds)} --> {ts((i + 1) * seconds)}\n{text}" for i, text in enumerate(lines)]
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


PACKAGING_TALK = _vtt(
    [
        "Welcome to the show. Today we explain how Python packaging works from source to wheel.",
        "First, a package is described by a pyproject file, which declares the build backend.",
        "The build backend turns your source tree into a wheel, which is a zip file with metadata.",
        "For example, hatchling and setuptools are build backends; pip is the installer, not the builder.",
        "This distinction matters because the installer only needs the wheel, never your build tools.",
        "However, packages with compiled extensions need a wheel per platform and Python version.",
        "That is why projects publish many wheels, and why a missing wheel triggers a slow source build.",
        "Next, lock files record the exact versions you resolved, so installs are reproducible.",
        "In summary: declare in pyproject, build wheels with a backend, install with an installer, lock.",
    ]
    * 3
)


def _write(cache: Path, source: LicensedSource, path: str, text: str) -> bytes:
    p = cache / source.key / source.commit / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode())
    return text.encode()


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def env(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, Path, Path]]:
    if request.param == "postgres" and not PG_URL:
        pytest.skip("AVD_TEST_DATABASE_URL not set")
    from avd.config import get_settings
    from avd.db.session import get_engine, get_sessionmaker
    from avd.embeddings.base import get_embedder

    caches = (get_settings, get_engine, get_sessionmaker, get_embedder)
    monkeypatch.setenv(
        "AVD_DATABASE_URL", PG_URL if request.param == "postgres" else f"sqlite:///{tmp_path / 'e2e.db'}"
    )
    monkeypatch.setenv("AVD_ADMIN_TOKEN", ADMIN["X-Admin-Token"])
    monkeypatch.setenv("AVD_MEDIA_CACHE_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("AVD_LLM_PROVIDER", "heuristic")
    monkeypatch.delenv("AVD_REDIS_URL", raising=False)
    for f in caches:
        f.cache_clear()
    from sqlalchemy import text

    from avd.db.models import Base

    engine = get_engine()
    if engine.dialect.name == "postgresql":
        with engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    cache = tmp_path / "dataset_cache"
    items = []
    for source, path, body in (
        (CHANGELOG, "podcast/the-changelog-1.md", CACHING_EPISODE),
        (CHANGELOG, "gotime/go-time-2.md", TRACING_EPISODE),
        (TALKPYTHON, "youtube_transcripts/301-packaging.vtt", PACKAGING_TALK),
    ):
        # The pilot's selection filter (3k-20k words) is ignored here: these texts are short on purpose.
        item, _ = describe_item(source, path, _write(cache, source, path, body))
        items.append(item)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(finalize_manifest(items, {}, seed=1)))

    from avd.api.app import create_app

    with TestClient(create_app()) as client:
        yield client, manifest, cache
    engine.dispose()
    for f in caches:
        f.cache_clear()


def _cli(capsys: pytest.CaptureFixture[str], *argv: str) -> dict[str, Any]:
    from avd.cli import main

    main(list(argv))
    out: dict[str, Any] = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return out


def _run_worker(client: TestClient) -> dict[str, int]:
    r = client.post("/v1/admin/worker/run", headers=ADMIN)
    assert r.status_code == 200
    outcomes: dict[str, int] = r.json()["outcomes"]
    assert set(outcomes) <= {"succeeded"}, outcomes
    return outcomes


def test_licensed_item_to_account_deletion(
    env: tuple[TestClient, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    client, manifest, cache = env

    # 1. Import: checksums and rights verified, evaluation queued, nothing fetched from the network.
    imported = _cli(
        capsys, "dataset", "import", str(manifest), "--cache-dir", str(cache), "--offline", "--evaluate"
    )
    assert imported["imported"] == 3
    assert _run_worker(client)["succeeded"] >= 6  # Stage A + Stage B per item

    # 2. The public evaluation says what kind of assessment it is; nothing pretends to be AI.
    meta = client.get("/v1/meta").json()
    assert meta["evaluation_mode"] == "heuristic_baseline"
    assert "No AI evaluator is configured" in meta["disclaimer"]
    detail = client.get("/v1/videos/chl_podcast_1").json()
    src = detail["content_sources"][0]
    assert src["license"] == "CC-BY-SA-4.0" and CHANGELOG.commit in src["provenance_url"]
    ev = client.get("/v1/videos/chl_podcast_1/evaluation").json()
    assert ev["synthetic"] is False and ev["mode"] == "heuristic_baseline"
    assert ev["assessment_state"] == "heuristic_baseline"
    by_dim = {d["dimension"]: d for d in ev["dimensions"]}
    # Unavailable dimensions are not zero: they carry a status and no score.
    for dim in ("audio_quality", "visual_quality"):
        assert by_dim[dim]["status"] == "requires_media" and by_dim[dim]["score"] is None
    assert by_dim["factual_accuracy"]["score"] is None  # text statistics cannot verify facts
    assert ev["overall"]["score"] is not None
    for d in ev["dimensions"]:
        for e in d["evidence"]:
            assert e["verified"] is True
    assert client.get("/v1/videos/chl_podcast_1/claims").json()["items"] == []

    # 3. A manual genre correction re-runs the evaluation with the corrected profile.
    r = client.put("/v1/admin/videos/tpy_301/genre", json={"genre": "lecture", "by": "e2e"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["requeued"]
    _run_worker(client)
    corrected = client.get("/v1/videos/tpy_301/evaluation").json()
    assert corrected["genre"] == "lecture" and corrected["aggregation_profile"]["genre"] == "lecture"
    assert corrected["evaluation_id"] != ev["evaluation_id"]
    assert client.get("/v1/videos/tpy_301").json()["video"]["genre_status"] == "overridden"

    # 4. Blind annotation: the rater sees transcript and rubric, never model scores or popularity.
    rater = client.post("/v1/annotation/raters", json={"pseudonym": "e2e-rater"}, headers=ADMIN).json()
    tok = {"Authorization": f"Bearer {rater['token']}"}
    item = client.get("/v1/annotation/next", params={"dataset": "real_pilot@v1"}, headers=tok).json()
    assert item["done"] is False and item["license"]
    assert "overall" not in json.dumps(item) and "view_count" not in json.dumps(item)
    seg = item["segments"][2]
    scores = [
        {"dimension": d["dimension"], "status": "insufficient_evidence"}
        for d in item["rubric"]
        if d["dimension"] != "clarity"
    ]
    scores.append(
        {
            "dimension": "clarity",
            "status": "scored",
            "score": 4,
            "evidence": [{"segment_index": seg["index"], "quote": " ".join(seg["text"].split()[:6])}],
        }
    )
    sub = client.post(
        f"/v1/annotation/annotations/{item['annotation_id']}/submit", json={"scores": scores}, headers=tok
    )
    assert sub.status_code == 200, sub.text
    exported = client.get("/v1/annotation/export", params={"dataset": "real_pilot@v1"}, headers=ADMIN).json()
    rows = [x for x in exported["labels"] if x["dimension"] == "clarity"]
    assert not item["is_calibration"]  # three single-item groups: no calibration split (test/dev/test)
    assert rows and rows[0]["label_source"] == "human_annotation" and rows[0]["score"] == 4
    assert rows[0]["evidence"][0]["segment_index"] == seg["index"]
    assert any("CC BY-SA 4.0" in a or "CC-BY-SA-4.0" in a for a in exported["attributions"])

    # 5. Recommendations keep quality, relevance and ranking separate and explain themselves.
    user = client.post("/v1/users").json()
    h = {"Authorization": f"Bearer {user['token']}"}
    prefs = {"interests": ["cache invalidation", "python packaging wheels"], "intent": "learn"}
    assert client.put("/v1/users/me/preferences", headers=h, json=prefs).status_code == 200
    body = client.get("/v1/recommendations", headers=h, params={"limit": 5}).json()
    assert body["eligibility"]["candidates"] >= 3
    recs = {it["video"]["id"]: it for it in body["items"]}
    assert {"chl_podcast_1", "tpy_301"} <= set(recs)
    for it in recs.values():
        assert {"intrinsic_quality", "relevance", "ranking"} <= set(it)
        assert "youtube.com" not in (it["video"]["url"] or "")  # licensed items are not YouTube videos
        assert not any(k.startswith(("view", "like", "subscriber")) for k in it["ranking"]["features"])
        assert any("heuristic" in x.lower() for x in it["explanations"])

    # 6. Feedback is tied to the impression it answers.
    it = recs["tpy_301"]
    fb = {"video_id": "tpy_301", "kind": "recommendation_useful", "impression_id": it["impression_id"]}
    assert client.post("/v1/feedback", headers=h, json=fb).status_code == 201

    # 7. Takedown: revoked content is withdrawn everywhere.
    _cli(capsys, "dataset", "revoke", "chl_podcast_1", "--reason", "rights holder request")
    assert client.get("/v1/videos/chl_podcast_1/evaluation").status_code == 404
    again = client.get("/v1/recommendations", headers=h, params={"limit": 5}).json()["items"]
    assert "chl_podcast_1" not in {x["video"]["id"] for x in again}

    # 8. Account deletion removes the user's data and invalidates the token.
    gone = client.delete("/v1/users/me", headers=h)
    assert gone.status_code == 200 and gone.json()["deleted"] is True and gone.json()["feedback"] >= 1
    assert client.get("/v1/users/me/preferences", headers=h).status_code == 401
