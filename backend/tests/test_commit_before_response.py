"""A write is committed before its response reaches the client.

Regression test for the intermittent Phase 3.3 E2E failure ("19 cleanup verified"): FastAPI runs a
``yield`` dependency's exit code *after* the response is sent unless the dependency is declared
with ``scope="function"``. The request session committed in that exit code, so a client could
receive "deleted" and immediately read the not-yet-deleted account, and a failed commit was
reported as success. ``TestClient`` waits for the whole ASGI call, which hides the race, so this
test runs a real server.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from tests.conftest import FIXTURES


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    import uvicorn

    from avd.config import get_settings
    from avd.db.models import Base
    from avd.db.session import get_engine, get_sessionmaker
    from avd.embeddings.base import get_embedder

    caches = (get_settings, get_engine, get_sessionmaker, get_embedder)
    monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{tmp_path / 'live.db'}")
    monkeypatch.setenv("AVD_FIXTURES_DIR", str(FIXTURES))
    monkeypatch.setenv("AVD_LLM_PROVIDER", "heuristic")
    monkeypatch.setenv("AVD_ALLOWED_HOSTS", json.dumps(["127.0.0.1", "localhost"]))
    monkeypatch.delenv("AVD_REDIS_URL", raising=False)
    for f in caches:
        f.cache_clear()
    Base.metadata.create_all(get_engine())
    from avd.api.app import create_app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(10)
    get_engine().dispose()
    for f in caches:
        f.cache_clear()


def _req(
    base: str, path: str, method: str = "GET", token: str | None = None, body: Any = None
) -> tuple[int, Any]:
    r = urllib.request.Request(
        base + path, method=method, data=None if body is None else json.dumps(body).encode()
    )
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None


def test_writes_commit_before_the_response(live: str, monkeypatch: pytest.MonkeyPatch) -> None:
    st, user = _req(live, "/v1/users", "POST")
    assert st == 201
    token = user["token"]
    assert _req(live, "/v1/users/me/preferences", token=token)[0] == 200

    real_commit = Session.commit
    commits: list[float] = []

    def slow_commit(self: Session) -> None:
        time.sleep(0.4)  # a commit waiting on a busy database
        real_commit(self)
        commits.append(time.monotonic())

    monkeypatch.setattr(Session, "commit", slow_commit)
    st, body = _req(live, "/v1/users/me", "DELETE", token=token)
    answered = time.monotonic()
    assert st == 200 and body["deleted"] is True
    assert commits and commits[0] <= answered, "response was sent before the deletion was committed"
    # The very next request must already see the deletion.
    assert _req(live, "/v1/users/me/preferences", token=token)[0] == 401


def test_a_failed_commit_is_not_reported_as_success(live: str, monkeypatch: pytest.MonkeyPatch) -> None:
    st, user = _req(live, "/v1/users", "POST")
    token = user["token"]

    def failing_commit(self: Session) -> None:
        raise RuntimeError("simulated commit failure")

    monkeypatch.setattr(Session, "commit", failing_commit)
    st, _ = _req(live, "/v1/users/me", "DELETE", token=token)
    monkeypatch.undo()
    assert st == 500
    assert _req(live, "/v1/users/me/preferences", token=token)[0] == 200  # nothing was deleted


def test_every_session_dependency_is_function_scoped() -> None:
    import re

    import avd

    root = Path(avd.__file__).resolve().parent
    offenders = [
        f"{p.relative_to(root)}:{n}"
        for p in root.rglob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"Depends\(db\s*\)", line)
    ]
    assert offenders == [], 'use Depends(db, scope="function"): ' + ", ".join(offenders)
