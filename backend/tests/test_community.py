"""Phase 3.3: contributor benchmark results (pack, verify, import).

Reports here come from ``bench-local`` runs against the Ollama *protocol mock*. They are refused
as "not real inference" in their original form. The acceptance path is tested by editing that
report into a structurally valid "real" report: a test fixture, never committed or presented
as a result.
"""

# ruff: noqa: F811  (the `manifest` fixture is imported, then requested by name)

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from avd.bench import community as cm
from avd.bench import local as bl
from tests.conftest import make_settings
from tests.ollama_mock import FakeOllama
from tests.test_evaluate_real import manifest  # noqa: F401
from tests.test_local_ollama import MODEL, _provider


@pytest.fixture
def bench_dir(tmp_path: Path, manifest: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    path, cache = manifest
    monkeypatch.setattr(cm, "REPO_ROOT", tmp_path)  # verify() compares against this manifest
    (tmp_path / "benchmarks" / "datasets" / "real_pilot_v1").mkdir(parents=True)
    (tmp_path / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json").write_text(path.read_text())
    out = tmp_path / "bench"
    fake = FakeOllama()
    cfg = bl.LocalBenchConfig(
        manifest=path, models=[MODEL], cache_dir=cache, offline=True, out_dir=out, repeats=1
    )
    bl.bench_local(cfg, make_settings(tmp_path), lambda st: _provider(fake, num_ctx=st.ollama_num_ctx))
    return out


def _as_real(bench_dir: Path) -> Path:
    """TEST FIXTURE ONLY: relabel a mock run as real to exercise the acceptance path."""
    r = json.loads((bench_dir / "report.json").read_text())
    r["meta"].update(transport="ollama http (real local inference)", transport_kind="real", git_dirty=False)
    p = bench_dir / "report.json"
    p.write_text(json.dumps(r))
    return p


class TestPack:
    def test_pack_excludes_databases_and_media(self, bench_dir: Path, tmp_path: Path) -> None:
        assert list(bench_dir.rglob("run.db")), "the run directory has databases with transcript text"
        out = cm.pack(bench_dir, tmp_path / "pilot.zip")
        names = zipfile.ZipFile(tmp_path / "pilot.zip").namelist()
        assert "report.json" in names and "PACK.json" in names
        assert not any(n.endswith(".db") or "/media/" in n for n in names)
        assert out["excluded"] > 0

    def test_report_records_dataset_fingerprint(self, bench_dir: Path) -> None:
        meta = json.loads((bench_dir / "report.json").read_text())["meta"]
        assert meta["transport_kind"] == "injected" and len(meta["manifest_items_sha256"]) == 64


class TestVerify:
    def test_mock_runs_are_refused(self, bench_dir: Path) -> None:
        v = cm.verify(bench_dir / "report.json")
        assert v["accepted"] is False
        assert next(c for c in v["checks"] if c["check"].startswith("real inference"))["ok"] is False

    def test_structurally_valid_real_report_is_accepted_as_contributor_reported(
        self, bench_dir: Path
    ) -> None:
        v = cm.verify(_as_real(bench_dir))
        assert v["accepted"] and v["status"] == "contributor_reported", [
            c for c in v["checks"] if not c["ok"]
        ]
        assert "self-reported" in v["cannot_verify"][0]

    @pytest.mark.parametrize(
        "mutate,failing",
        [
            (lambda r: r["meta"].update(split="test"), "development split only"),
            (lambda r: r["meta"].update(manifest_items_sha256="0" * 64), "dataset matches this repository"),
            (lambda r: r["meta"].update(git_dirty=True), "git commit recorded and clean"),
            (
                lambda r: next(iter(r["models"].values()))["runtime"]["models"][MODEL].update(
                    fingerprint="qwen"
                ),
                "weights digest recorded",
            ),
            (
                lambda r: next(iter(next(iter(r["models"].values()))["windows"].values())).update(
                    evidence_verified=999
                ),
                "numbers in range and consistent",
            ),
        ],
    )
    def test_each_provenance_check(self, bench_dir: Path, mutate: Any, failing: str) -> None:
        p = _as_real(bench_dir)
        r = json.loads(p.read_text())
        mutate(r)
        p.write_text(json.dumps(r))
        v = cm.verify(p)
        assert not v["accepted"] and any(failing in c["check"] and not c["ok"] for c in v["checks"])

    def test_tampered_pack_and_unsafe_paths_are_refused(self, bench_dir: Path, tmp_path: Path) -> None:
        _as_real(bench_dir)
        cm.pack(bench_dir, tmp_path / "p.zip")
        data = zipfile.ZipFile(tmp_path / "p.zip")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for n in data.namelist():
                content = data.read(n)
                z.writestr(n, content.replace(b'"calls"', b'"calls" ') if n == "report.json" else content)
        (tmp_path / "t.zip").write_bytes(buf.getvalue())
        assert "does not match the pack manifest" in cm.verify(tmp_path / "t.zip")["checks"][0]["detail"]
        for bad in ("../evil.json", "runs/x/run.db"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("report.json", "{}")
                z.writestr(bad, "x")
            (tmp_path / "bad.zip").write_bytes(buf.getvalue())
            assert cm.verify(tmp_path / "bad.zip")["accepted"] is False


class TestImport:
    def test_import_records_verdict_and_refuses_duplicates(self, bench_dir: Path, tmp_path: Path) -> None:
        _as_real(bench_dir)
        cm.pack(bench_dir, tmp_path / "p.zip")
        out = cm.import_result(tmp_path / "p.zip", "volunteer-7", dest_root=tmp_path / "community")
        rec = json.loads((Path(out["imported_to"]) / "VERIFICATION.json").read_text())
        assert rec["status"] == "contributor_reported" and rec["contributor"] == "volunteer-7"
        with pytest.raises(cm.CommunityError, match="already exists"):
            cm.import_result(tmp_path / "p.zip", "volunteer-7", dest_root=tmp_path / "community")

    def test_import_refuses_mock_runs(self, bench_dir: Path, tmp_path: Path) -> None:
        with pytest.raises(cm.CommunityError, match="real inference"):
            cm.import_result(bench_dir / "report.json", "x", dest_root=tmp_path / "community")
