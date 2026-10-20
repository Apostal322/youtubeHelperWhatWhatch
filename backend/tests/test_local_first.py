"""Phase 3.2: local-first operation: deployment guard, registration, private servers, bounded
queue, cancellation, embedded worker and the `avd start` / `avd serve` safety checks."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from avd.api.guard import host_without_port, is_loopback, trusted_origin
from tests.conftest import make_settings

ADMIN = "a" * 32


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from avd.config import get_settings
    from avd.db.models import Base
    from avd.db.session import get_engine, get_sessionmaker

    clients: list[TestClient] = []

    def make(client: tuple[str, int] = ("testclient", 5000), **env: str) -> TestClient:
        monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{tmp_path / 'lf.db'}")
        monkeypatch.delenv("AVD_REDIS_URL", raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        for f in (get_settings, get_engine, get_sessionmaker):
            f.cache_clear()
        Base.metadata.create_all(get_engine())
        from avd.api.app import create_app

        c = TestClient(create_app(), client=client)
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)
    get_engine().dispose()
    for f in (get_settings, get_engine, get_sessionmaker):
        f.cache_clear()


class TestGuardHelpers:
    @pytest.mark.parametrize(
        "raw,host", [("localhost:8000", "localhost"), ("[::1]:8000", "[::1]"), ("Example.ORG", "example.org")]
    )
    def test_host_parsing(self, raw: str, host: str) -> None:
        assert host_without_port(raw) == host

    def test_loopback(self) -> None:
        assert is_loopback("127.0.0.1") and is_loopback("::1") and not is_loopback("192.168.1.5")
        assert not is_loopback("evil.example")

    def test_trusted_origins(self) -> None:
        s = make_settings(Path("/tmp"))
        assert trusted_origin("chrome-extension://abcdef", s) and trusted_origin("http://localhost:5173", s)
        assert not trusted_origin("https://evil.example", s) and not trusted_origin("null", s)


class TestLocalMode:
    def test_dns_rebinding_host_is_rejected(self, app_factory: Any) -> None:
        c = app_factory()
        r = c.get("/v1/meta", headers={"Host": "attacker.example:8000"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_host"
        assert c.get("/health").status_code == 200

    def test_other_machines_are_refused(self, app_factory: Any) -> None:
        c = app_factory(client=("192.168.1.50", 5000))
        r = c.get("/v1/meta")
        assert r.status_code == 403 and r.json()["error"]["code"] == "local_only"

    def test_docker_bridge_can_be_trusted_explicitly(self, app_factory: Any) -> None:
        c = app_factory(client=("172.18.0.1", 5000), AVD_LOCAL_PEER_NETWORKS='["172.16.0.0/12"]')
        assert c.get("/v1/meta").status_code == 200

    @pytest.mark.parametrize("net", ["0.0.0.0/0", "8.8.8.0/24"])
    def test_public_peer_networks_are_rejected(self, tmp_path: Path, net: str) -> None:
        with pytest.raises(ValueError, match="private networks"):
            make_settings(tmp_path, local_peer_networks=[net])

    def test_loopback_peer_is_served(self, app_factory: Any) -> None:
        assert app_factory(client=("127.0.0.1", 5000)).get("/v1/meta").status_code == 200

    def test_cross_site_writes_are_refused(self, app_factory: Any) -> None:
        c = app_factory()
        r = c.post("/v1/users", headers={"Origin": "https://evil.example"})
        assert r.status_code == 403 and r.json()["error"]["code"] == "cross_site"
        assert c.post("/v1/users", headers={"Origin": "chrome-extension://abc"}).status_code == 201
        assert c.post("/v1/users").status_code == 201  # no Origin: CLI, curl
        assert (
            c.get("/v1/meta", headers={"Origin": "https://evil.example"}).status_code == 200
        )  # reads: CORS applies

    def test_status_reports_mode_and_worker(self, app_factory: Any) -> None:
        st = app_factory().get("/v1/status").json()
        assert st["deployment_mode"] == "local" and st["registration"] == "open"
        assert st["worker"] == {"embedded": False} and st["inference"]["provider"] == "heuristic"

    def test_no_cors_wildcard_for_extensions(self, app_factory: Any) -> None:
        r = app_factory().get("/v1/meta", headers={"Origin": "chrome-extension://someone-else"})
        assert "access-control-allow-origin" not in r.headers


def test_empty_admin_token_disables_admin(app_factory: Any) -> None:
    c = app_factory(AVD_ADMIN_TOKEN="")
    r = c.post("/v1/admin/jobs/cancel", json={}, headers={"X-Admin-Token": ""})
    assert r.status_code == 403 and "disabled" in r.json()["error"]["message"]


def test_health_works_for_any_host_in_server_mode(app_factory: Any) -> None:
    c = app_factory(client=("172.18.0.1", 1), **TestServerMode.SAFE)
    assert c.get("/health", headers={"Host": "127.0.0.1:8000"}).status_code == 200
    assert c.get("/v1/meta", headers={"Host": "127.0.0.1:8000"}).status_code == 400


class TestServerMode:
    SAFE = {
        "AVD_DEPLOYMENT_MODE": "server",
        "AVD_ADMIN_TOKEN": ADMIN,
        "AVD_ALLOWED_HOSTS": '["avd.example.org", "testserver"]',
        "AVD_INVITE_CODES": "join-us-1, join-us-2",
    }

    @pytest.mark.parametrize(
        "env,problem",
        [
            ({"AVD_ADMIN_TOKEN": ""}, "AVD_ADMIN_TOKEN"),
            ({"AVD_ADMIN_TOKEN": "change-me-local-admin-token-xx"}, "placeholder"),
            ({"AVD_ALLOWED_HOSTS": '["localhost"]'}, "AVD_ALLOWED_HOSTS"),
            ({"AVD_INVITE_CODES": ""}, "AVD_INVITE_CODES"),
            ({"AVD_METRICS_PUBLIC": "true"}, "METRICS_PUBLIC"),
        ],
    )
    def test_unsafe_configuration_refuses_to_start(
        self, app_factory: Any, env: dict[str, str], problem: str
    ) -> None:
        merged = {**self.SAFE, **env}
        merged = {k: v for k, v in merged.items() if v != ""}
        with pytest.raises(RuntimeError, match=problem):
            app_factory(**merged)

    def test_invite_registration(self, app_factory: Any) -> None:
        c = app_factory(client=("203.0.113.9", 5000), **self.SAFE)
        assert c.post("/v1/users").status_code == 403
        assert c.post("/v1/users", headers={"X-Invite-Code": "wrong"}).status_code == 403
        assert c.post("/v1/users", headers={"X-Invite-Code": "join-us-2"}).status_code == 201
        assert c.post("/v1/users", headers={"X-Admin-Token": ADMIN}).status_code == 201
        assert c.get("/v1/meta").status_code == 200  # public read by default

    def test_closed_registration_and_private_reads(self, app_factory: Any) -> None:
        c = app_factory(
            client=("203.0.113.9", 5000),
            **{**self.SAFE, "AVD_USER_REGISTRATION": "closed", "AVD_REQUIRE_AUTH_FOR_READ": "true"},
        )
        assert c.post("/v1/users", headers={"X-Invite-Code": "join-us-1"}).status_code == 403
        assert c.get("/v1/videos").status_code == 401
        token = c.post("/v1/users", headers={"X-Admin-Token": ADMIN}).json()["token"]
        assert c.get("/v1/videos", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        assert c.get("/v1/status").json()["auth_required_for_read"] is True

    def test_server_hosts_are_enforced(self, app_factory: Any) -> None:
        c = app_factory(client=("203.0.113.9", 5000), **self.SAFE)
        assert c.get("/health", headers={"Host": "avd.example.org"}).status_code == 200
        assert c.get("/v1/meta", headers={"Host": "other.example"}).status_code == 400


class TestQueueControls:
    def test_backlog_limit_refuses_new_submissions(self, app_factory: Any) -> None:
        from avd.db.session import get_sessionmaker
        from avd.workers import queue

        c = app_factory(AVD_MAX_QUEUED_JOBS="2")
        with get_sessionmaker()() as s:
            for i in range(2):
                queue.enqueue(s, "stage_a", f"k{i}")
            s.commit()
        token = c.post("/v1/users").json()["token"]
        r = c.post(
            "/v1/videos/abcdefghijk/transcripts",
            json={
                "content": "hello there",
                "format": "text",
                "attestation": "I am the creator of this video.",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 503 and "queue is full" in r.json()["error"]["message"]

    def test_cancel_queued_jobs(self, tmp_path: Path) -> None:
        from avd.workers import queue
        from tests.conftest import _make_db

        db = _make_db(tmp_path, None)
        with db.sf() as s:
            a, _ = queue.enqueue(s, "stage_a", "a")
            queue.enqueue(s, "stage_b", "b")
            s.commit()
            assert queue.cancel(s, job_types=["stage_b"]) == 1
            assert queue.cancel(s, job_ids=[a.id]) == 1
            s.commit()
            assert queue.stats(s) == {"stage_a": {"cancelled": 1}, "stage_b": {"cancelled": 1}}
            _, created = queue.enqueue(s, "stage_a", "a", requeue_terminal=True)
            assert created  # cancelled is terminal: it can be requeued

    def test_admin_cancel_and_worker_control(self, app_factory: Any) -> None:
        c = app_factory(AVD_ADMIN_TOKEN=ADMIN)
        h = {"X-Admin-Token": ADMIN}
        assert c.post("/v1/admin/jobs/cancel", json={}, headers=h).json() == {"cancelled": 0}
        assert c.post("/v1/admin/worker/pause", headers=h).status_code == 409  # no embedded worker


class TestEmbeddedWorker:
    def test_processes_the_queue_and_survives_restart(self, app_factory: Any) -> None:
        from avd.db.models import PipelineJob
        from avd.db.session import get_sessionmaker
        from avd.services import seed_demo

        c = app_factory(
            AVD_EMBEDDED_WORKER="true", AVD_EMBEDDED_WORKER_POLL_SECONDS="0.05", AVD_ADMIN_TOKEN=ADMIN
        )
        from avd.config import get_settings

        with get_sessionmaker()() as s:
            seed_demo(s, get_settings())
            s.commit()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if c.get("/v1/videos?limit=50").json()["items"]:
                st = c.get("/v1/status").json()
                if not st["queue"].get("queued") and not st["queue"].get("running"):
                    break
            time.sleep(0.2)
        st = c.get("/v1/status").json()
        assert st["worker"]["running"] and st["worker"]["processed"].get("succeeded", 0) > 0
        assert c.post("/v1/admin/worker/pause", headers={"X-Admin-Token": ADMIN}).json()["paused"] is True
        assert c.post("/v1/admin/worker/resume", headers={"X-Admin-Token": ADMIN}).json()["paused"] is False
        items = c.get("/v1/videos?limit=50").json()["items"]
        assert items and all(
            v["video"]["is_synthetic"] and v["intrinsic_quality"]["mode"] == "synthetic_demo" for v in items
        )
        with get_sessionmaker()() as s:
            assert s.query(PipelineJob).filter(PipelineJob.status == "dead").count() == 0

    def test_interrupted_jobs_are_requeued_on_start(self, tmp_path: Path) -> None:
        from avd.workers import queue
        from avd.workers.embedded import release_stale_leases
        from tests.conftest import _make_db

        db = _make_db(tmp_path, None)
        with db.sf() as s:
            queue.enqueue(s, "stage_a", "x")
            s.commit()
            queue.claim(s, limit=1, lease_seconds=600)
            s.commit()
        assert release_stale_leases(db.sf) == 1
        with db.sf() as s:
            assert queue.stats(s) == {"stage_a": {"queued": 1}}


class TestCli:
    def test_start_prepares_an_isolated_local_instance(self, tmp_path: Path) -> None:
        import os

        from avd.cli import prepare_local
        from avd.config import get_settings
        from avd.db.session import get_engine, get_sessionmaker

        saved = dict(os.environ)
        try:
            for k in (
                "AVD_DATABASE_URL",
                "AVD_DEPLOYMENT_MODE",
                "AVD_EMBEDDED_WORKER",
                "AVD_MEDIA_CACHE_DIR",
            ):
                os.environ.pop(k, None)
            prepare_local(tmp_path / "data", "ollama", worker=True)
            s = get_settings()
            assert s.database_url == f"sqlite:///{tmp_path / 'data' / 'avd.db'}" and s.embedded_worker
            assert (
                s.deployment_mode.value == "local"
                and s.llm_provider.value == "ollama"
                and s.redis_url is None
            )
        finally:
            os.environ.clear()
            os.environ.update(saved)  # prepare_local writes os.environ directly; restore it exactly
            for f in (get_settings, get_engine, get_sessionmaker):
                f.cache_clear()

    def test_local_instance_refuses_a_public_bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from avd.cli import _check_bind
        from avd.config import get_settings

        monkeypatch.setenv("AVD_DEPLOYMENT_MODE", "local")
        get_settings.cache_clear()
        with pytest.raises(SystemExit, match="loopback only"):
            _check_bind("0.0.0.0")
        _check_bind("127.0.0.1")
        monkeypatch.setenv("AVD_LOCAL_PEER_NETWORKS", '["172.16.0.0/12"]')
        get_settings.cache_clear()
        _check_bind("0.0.0.0")  # container: the bridge is declared, the guard filters peers
        monkeypatch.delenv("AVD_LOCAL_PEER_NETWORKS")
        monkeypatch.setenv("AVD_DEPLOYMENT_MODE", "server")
        monkeypatch.delenv("AVD_ADMIN_TOKEN", raising=False)
        get_settings.cache_clear()
        with pytest.raises(SystemExit, match="unsafe server configuration"):
            _check_bind("0.0.0.0")
        get_settings.cache_clear()

    def test_data_dir_is_per_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from avd.cli import default_data_dir

        monkeypatch.delenv("AVD_DATA_DIR", raising=False)
        assert "ai-video-discovery" in str(default_data_dir())
        monkeypatch.setenv("AVD_DATA_DIR", "/srv/avd")
        assert str(default_data_dir()) == "/srv/avd"


class TestExperimentalEvaluators:
    EXAMPLE = "word_count_evaluator:WordRateEvaluator"

    @pytest.fixture(autouse=True)
    def _examples_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from avd.config import REPO_ROOT

        monkeypatch.syspath_prepend(str(REPO_ROOT / "docs" / "examples"))

    def test_plugin_cannot_override_official_scores_and_is_recorded(self, tmp_path: Path) -> None:
        from avd.evaluation.pipeline import build_pipeline
        from avd.evaluation.schema import Dimension
        from tests.conftest import make_input

        official = build_pipeline("B", make_settings(tmp_path), None)
        with_plugin = build_pipeline(
            "B", make_settings(tmp_path, experimental_evaluators=[self.EXAMPLE]), None
        )
        a, b = official.run(make_input()), with_plugin.run(make_input())
        da = {d.dimension: d for d in a.dimensions}
        db = {d.dimension: d for d in b.dimensions}
        assert da[Dimension.INFORMATION_DENSITY].status.value == "scored"
        assert (
            db[Dimension.INFORMATION_DENSITY].method == da[Dimension.INFORMATION_DENSITY].method
        )  # official kept
        assert (
            b.versions["experimental_evaluators"] == "word_rate@0.1"
            and "experimental_evaluators" not in a.versions
        )
        assert official.config_hash() != with_plugin.config_hash()

    def test_plugin_fills_only_official_abstentions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from avd.evaluation.evaluators.heuristic import HeuristicTextEvaluator
        from avd.evaluation.pipeline import build_pipeline
        from avd.evaluation.schema import Dimension, DimensionScore, DimensionStatus
        from tests.conftest import make_input

        original = HeuristicTextEvaluator.evaluate

        def abstain_on_density(self: Any, inp: Any, profile: Any) -> Any:
            out = original(self, inp, profile)
            out.dimensions = [
                DimensionScore.abstain(
                    d.dimension, DimensionStatus.INSUFFICIENT_EVIDENCE, "simulated", "test"
                )
                if d.dimension == Dimension.INFORMATION_DENSITY
                else d
                for d in out.dimensions
            ]
            return out

        monkeypatch.setattr(HeuristicTextEvaluator, "evaluate", abstain_on_density)
        inp = make_input().model_copy(update={"duration_seconds": 240})
        res = build_pipeline("B", make_settings(tmp_path, experimental_evaluators=[self.EXAMPLE]), None).run(
            inp
        )
        d = next(x for x in res.dimensions if x.dimension == Dimension.INFORMATION_DENSITY)
        assert d.status.value == "scored" and d.method == "word_rate@0.1"

    @pytest.mark.parametrize("spec", ["nomodule", "json:loads", "word_count_evaluator:Missing"])
    def test_bad_plugin_specs_fail_loudly(self, spec: str) -> None:
        from avd.plugins import PluginError, load_evaluators

        with pytest.raises(PluginError):
            load_evaluators([spec])


def test_embedded_worker_survives_database_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing claim or job (e.g. "database or disk is full") must not kill the worker thread;
    it records the error, backs off, and processes work once the database recovers."""
    from sqlalchemy.exc import OperationalError

    from avd.services import seed_demo
    from avd.workers import runner
    from avd.workers.context import build_context
    from avd.workers.embedded import EmbeddedWorker
    from tests.conftest import _make_db

    db = _make_db(tmp_path, None)
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    real_claim, real_exec = runner.claim_batch, runner.execute_job
    calls = {"claim": 0, "exec": 0}

    def flaky_claim(*a: Any, **k: Any) -> Any:
        calls["claim"] += 1
        if calls["claim"] == 1:
            raise OperationalError("INSERT", {}, Exception("database or disk is full"))
        return real_claim(*a, **k)

    def flaky_exec(*a: Any, **k: Any) -> Any:
        calls["exec"] += 1
        if calls["exec"] == 1:
            raise OperationalError("UPDATE", {}, Exception("database or disk is full"))
        return real_exec(*a, **k)

    monkeypatch.setattr(runner, "claim_batch", flaky_claim)
    monkeypatch.setattr(runner, "execute_job", flaky_exec)
    w = EmbeddedWorker(build_context(settings=db.settings, session_factory=db.sf), poll_seconds=0.01)
    w.start()
    from avd.workers import queue

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with db.sf() as s:
            st = queue.stats(s)
        if all(set(v) <= {"succeeded", "skipped"} for v in st.values()) and st:
            break
        time.sleep(0.05)
    w.stop()
    assert "disk is full" in (w.last_error or "") and calls["claim"] >= 2 and calls["exec"] >= 2
    # The crashed job was released at once and completed: nothing is left locked or dead.
    assert all(set(v) <= {"succeeded", "skipped"} for v in st.values()), st
