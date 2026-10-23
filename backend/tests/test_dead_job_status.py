"""A video whose evaluation job gave up says so (audit 2026-10-23). Model failures are scripted."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from avd.llm.base import LLMTransientError
from avd.services import seed_demo
from avd.workers.context import build_context
from avd.workers.runner import run_until_idle
from tests.conftest import ScriptedProvider, _make_db
from tests.test_api import _clear_caches


def test_evaluation_404_reports_a_job_that_gave_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_db(tmp_path, None)
    db.settings.job_retry_base_seconds = 0.0
    db.settings.llm_max_retries = 0
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    down = ScriptedProvider(lambda r, i: LLMTransientError("Ollama unreachable at http://10.0.0.5:11434"))
    ctx = build_context(settings=db.settings, session_factory=db.sf, llm=down)  # type: ignore[arg-type]
    for _ in range(6):
        run_until_idle(ctx)

    monkeypatch.setenv("AVD_DATABASE_URL", db.settings.database_url)
    monkeypatch.setenv("AVD_LLM_PROVIDER", "heuristic")
    monkeypatch.delenv("AVD_REDIS_URL", raising=False)
    from avd.api.app import create_app

    _clear_caches()
    try:
        with TestClient(create_app()) as c:
            r = c.get("/v1/videos/syn_tech_001/evaluation")
    finally:
        _clear_caches()
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "gave up" in detail  # before: only "status: content_available", as if still on its way
    assert "10.0.0.5" not in detail
