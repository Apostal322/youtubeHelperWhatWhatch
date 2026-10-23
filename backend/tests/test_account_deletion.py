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
