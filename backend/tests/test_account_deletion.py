"""Account deletion erases what was derived from the user's content (audit 2026-10-23).

Before the fix, verbatim transcript text survived in evidence quotes (and justifications, claims,
cached model responses), uploaded media stayed on disk, and the video stayed "evaluated".
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FIXTURES

MARKER = "zebrafinch"
TEXT = " ".join(
    f"The {MARKER} method, step {i}, explains how photosynthesis fixes carbon in the Calvin cycle."
    for i in range(40)
)
ATT = "I am the creator of this video and hold the rights to its transcript."
ADMIN = {"X-Admin-Token": "test-admin"}


def _clear() -> None:
    from avd.config import get_settings
    from avd.db.session import get_engine, get_sessionmaker
    from avd.embeddings.base import get_embedder

    for f in (get_settings, get_engine, get_sessionmaker, get_embedder):
        f.cache_clear()


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Path, Path]]:
    db = tmp_path / "avd.db"
    monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("AVD_ADMIN_TOKEN", "test-admin")
    monkeypatch.setenv("AVD_MEDIA_CACHE_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("AVD_FIXTURES_DIR", str(FIXTURES))
    monkeypatch.setenv("AVD_LLM_PROVIDER", "heuristic")
    _clear()
    from avd.api.app import create_app
    from avd.db.models import Base
    from avd.db.session import get_engine

    Base.metadata.create_all(get_engine())
    with TestClient(create_app()) as c:
        yield c, tmp_path, db
    _clear()


def _tables_containing(db: Path, needle: str) -> dict[str, int]:
    con = sqlite3.connect(db)
    found: dict[str, int] = {}
    for (table,) in con.execute("select name from sqlite_master where type='table'"):
        n = sum(1 for row in con.execute(f'select * from "{table}"') if any(needle in str(v) for v in row))
        if n:
            found[table] = n
    con.close()
    return found


def test_no_text_from_a_deleted_users_transcript_remains(api: tuple[TestClient, Path, Path]) -> None:
    c, _, db = api
    h = {"Authorization": "Bearer " + c.post("/v1/users").json()["token"]}
    assert (
        c.post(
            "/v1/videos/userVid0001/transcripts",
            headers=h,
            json={"format": "text", "content": TEXT, "attestation": ATT},
        ).status_code
        == 202
    )
    c.post("/v1/admin/worker/run", headers=ADMIN)
    assert "evidence" in _tables_containing(db, MARKER)  # the evaluation quoted the transcript
    out = c.delete("/v1/users/me", headers=h).json()
    assert out["deleted"] and out["quotes_erased"] > 0
    assert _tables_containing(db, MARKER) == {}
    assert c.get("/v1/videos/userVid0001/evaluation").status_code == 404
    assert c.get("/v1/videos/userVid0001").json()["video"]["status"] != "evaluated"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to make and validate a test WAV")
def test_uploaded_media_files_are_deleted(api: tuple[TestClient, Path, Path]) -> None:
    c, tmp, _ = api
    h = {"Authorization": "Bearer " + c.post("/v1/users").json()["token"]}
    wav = tmp / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(wav)],
        check=True,
    )
    r = c.put(
        "/v1/videos/userVid0002/media",
        content=wav.read_bytes(),
        headers={**h, "Content-Type": "audio/wav", "X-Content-Attestation": ATT},
    )
    assert r.status_code == 202, r.text
    c.post("/v1/admin/worker/run", headers=ADMIN)
    assert any((tmp / "media").rglob("*.wav"))
    out = c.delete("/v1/users/me", headers=h).json()
    assert out["media_files"] >= 1
    assert not any(p.is_file() for p in (tmp / "media").rglob("*"))


def test_resubmitting_text_whose_first_submitter_was_deleted_works(
    api: tuple[TestClient, Path, Path],
) -> None:
    c, _, _ = api
    a = {"Authorization": "Bearer " + c.post("/v1/users").json()["token"]}
    b = {"Authorization": "Bearer " + c.post("/v1/users").json()["token"]}
    body = {"format": "text", "content": TEXT, "attestation": ATT}
    assert c.post("/v1/videos/abcdefghijk/transcripts", json=body, headers=a).json()["created"]
    assert c.delete("/v1/users/me", headers=a).status_code == 200
    r = c.post("/v1/videos/abcdefghijk/transcripts", json=body, headers=b).json()
    assert r["created"] is True  # before: the revoked row came back, "already on file", and stayed revoked


def _evaluated_source(tmp_path: Path) -> tuple[object, str, str]:
    from avd.db.models import Channel, ContentSource, Evaluation, LLMCache, Video
    from tests.conftest import _make_db

    db = _make_db(tmp_path, None)
    with db.sf() as s:
        s.add(Channel(id="ch1", title="c"))
        s.add(Video(id="vid_one_123", title="t", channel_id="ch1", status="evaluated"))
        src = ContentSource(
            video_id="vid_one_123",
            source_type="user_provided_transcript",
            authorization_basis="attested",
            content_hash="h1",
            modalities=["transcript"],
            status="revoked",
        )
        s.add(src)
        s.flush()
        s.add(
            Evaluation(
                video_id="vid_one_123",
                stage="B",
                framework_version="x",
                mode="llm",
                genre="science",
                genre_confidence=1.0,
                input_fingerprint="f",
                content_source_ids=[src.id],
                status="complete",
                coverage_report={
                    "evaluators": {
                        "llm_rubric": {
                            "chunks": 3,
                            "claims_extracted": [
                                {"segment_index": 2, "quote": f"the {MARKER} quote", "claim": "c"}
                            ],
                            "map_detected_issues": [f"issue about {MARKER}"],
                        }
                    }
                },
                validation={
                    "passed": True,
                    "evidence_claimed": 4,
                    "warnings": [f"model-detected issue: {MARKER}"],
                },
            )
        )
        # A part summary paraphrases the transcript: it shares no quote with the evidence.
        s.add(
            LLMCache(
                key="k1",
                provider="ollama",
                model="m",
                response={"data": {"summary": f"{MARKER} paraphrase"}, "video_id": "vid_one_123"},
            )
        )
        s.add(
            LLMCache(
                key="k2",
                provider="ollama",
                model="m",
                response={"data": {"summary": "other"}, "video_id": "other_video"},
            )
        )
        s.commit()
        return db, src.id, "vid_one_123"


def test_paraphrases_and_copies_of_deleted_content_are_erased(tmp_path: Path) -> None:
    import json

    from sqlalchemy import select

    from avd.datactl import erase_derived_content
    from avd.db.models import Evaluation, LLMCache

    db, source_id, _ = _evaluated_source(tmp_path)
    with db.sf() as s:  # type: ignore[attr-defined]
        erase_derived_content(s, {source_id}, tmp_path / "media")
        s.commit()
        ev = s.scalars(select(Evaluation)).one()
        dumped = json.dumps([ev.coverage_report, ev.validation])
        assert MARKER not in dumped  # before: claims, model issues and warnings kept the text
        assert ev.coverage_report["evaluators"]["llm_rubric"]["chunks"] == 3  # counts stay for audit
        assert [r.key for r in s.scalars(select(LLMCache))] == ["k2"]  # before: the paraphrase stayed


def test_audio_still_used_by_another_source_is_kept(tmp_path: Path) -> None:
    from avd.datactl import erase_derived_content
    from avd.db.models import ContentSource

    db, source_id, vid = _evaluated_source(tmp_path)
    media = tmp_path / "media"
    media.mkdir()
    wav = media / "abc.audio.wav"
    wav.write_bytes(b"RIFF")
    with db.sf() as s:  # type: ignore[attr-defined]
        revoked = s.get(ContentSource, source_id)
        revoked.preprocessing = {"audio_wav": str(wav)}
        s.add(
            ContentSource(
                video_id=vid,
                source_type="user_provided_media",
                authorization_basis="attested",
                content_hash="h2",
                modalities=["audio"],
                status="active",
                preprocessing={"audio_wav": str(wav)},  # the same clip, uploaded again
            )
        )
        s.commit()
        erase_derived_content(s, {source_id}, media)
    assert wav.exists()  # before: deleted, though an active source still needed it
