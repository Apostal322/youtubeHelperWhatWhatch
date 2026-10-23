"""Contributor benchmark results: inspect, pack, verify and import.

Reports here come from ``bench-local`` runs against the Ollama *protocol mock*. They are refused
as "not real inference" in their original form. The acceptance path is tested by editing that
report into a structurally valid "real" report: a test fixture, never committed or presented
as a result.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from avd.bench import community as cm
from avd.bench import local as bl
from avd.datasets.pilot import describe_item, finalize_manifest
from avd.datasets.sources import CHANGELOG, TALKPYTHON
from tests.conftest import make_settings
from tests.ollama_mock import FakeOllama
from tests.test_e2e_scenario import CACHING_EPISODE, PACKAGING_TALK, TRACING_EPISODE, _write
from tests.test_local_ollama import MODEL, _provider


@dataclass
class Bench:
    dir: Path
    cache: Path
    manifest: Path


def _bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_talkpython: bool) -> Bench:
    cache = tmp_path / "cache"
    episodes = [
        (CHANGELOG, "podcast/the-changelog-1.md", CACHING_EPISODE),
        (CHANGELOG, "gotime/go-time-2.md", TRACING_EPISODE),
    ]
    if with_talkpython:
        episodes.append((TALKPYTHON, "youtube_transcripts/301-packaging.vtt", PACKAGING_TALK))
    items = [describe_item(src, path, _write(cache, src, path, body))[0] for src, path, body in episodes]
    manifest = tmp_path / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(finalize_manifest(items, {}, seed=1, forced_split="dev")))
    monkeypatch.setattr(cm, "REPO_ROOT", tmp_path)  # verify() compares against this manifest
    monkeypatch.setattr(cm, "default_cache_dir", lambda: cache)
    monkeypatch.setattr(cm, "_commit_known", lambda sha: True)
    out = tmp_path / "bench"
    fake = FakeOllama()
    cfg = bl.LocalBenchConfig(
        manifest=manifest, models=[MODEL], cache_dir=cache, offline=True, out_dir=out, repeats=1
    )
    bl.bench_local(cfg, make_settings(tmp_path), lambda st: _provider(fake, num_ctx=st.ollama_num_ctx))
    return Bench(out, cache, manifest)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    return _bench(tmp_path, monkeypatch, with_talkpython=False)


def _as_real(b: Bench) -> Path:
    """TEST FIXTURE ONLY: relabel a mock run as real to exercise the acceptance path."""
    p = b.dir / "report.json"
    r = json.loads(p.read_text())
    r["meta"].update(transport="ollama http (real local inference)", transport_kind="real", git_dirty=False)
    entry = r["models"][MODEL]
    entry["runtime"]["ollama_version"] = "0.12.3"
    for w in entry["windows"].values():
        w["latency_ms_p50"] = w["latency_ms_p95"] = 1234
    p.write_text(json.dumps(r))
    return p


def _edit(b: Bench, name: str, fn: Any) -> None:
    p = b.dir / name
    if name.endswith(".json"):
        doc = json.loads(p.read_text())
        fn(doc)
        p.write_text(json.dumps(doc))
    else:
        p.write_text(fn(p.read_text()))


def _run_dir(b: Bench) -> str:
    return next(p.relative_to(b.dir).as_posix() for p in (b.dir / "runs").iterdir())


class TestInspectAndPack:
    def test_inspect_is_a_dry_run_listing_exclusions(self, bench: Bench, tmp_path: Path) -> None:
        before = sorted(p for p in tmp_path.rglob("*"))
        out = cm.inspect(bench.dir)
        assert sorted(p for p in tmp_path.rglob("*")) == before  # nothing written
        assert out["ok"], out["blocking_findings"]
        assert out["transcripts_checked"] == 2
        excluded = {e["file"]: e["reason"] for e in out["would_exclude"]}
        assert any(f.endswith("run.db") and "transcript" in r for f, r in excluded.items())
        assert any("/items/" in f and "quotes" in r for f, r in excluded.items())
        included = {i["file"] for i in out["would_include"]}
        assert "report.json" in included and not any(f.endswith(".db") or "/items/" in f for f in included)

    def test_pack_contains_only_listed_redacted_files(self, bench: Bench, tmp_path: Path) -> None:
        _edit(bench, "report.json", lambda d: d["meta"].update(note=f"ran in {tmp_path}/secret-dir"))
        _edit(
            bench,
            f"{_run_dir(bench)}/plan.json",
            lambda d: d["meta"]["local_runtime"].update(base_url="http://192.168.1.20:11434"),
        )
        out = cm.pack(bench.dir, tmp_path / "pilot.zip")
        z = zipfile.ZipFile(tmp_path / "pilot.zip")
        names = set(z.namelist())
        assert "report.json" in names and "PACK.json" in names
        assert not any(n.endswith(".db") or "/items/" in n or "/media/" in n for n in names)
        blob = b"".join(z.read(n) for n in names)
        assert str(tmp_path).encode() not in blob and b"192.168.1.20" not in blob
        assert b"/root" not in blob and b"/home/" not in blob
        assert out["redactions"] >= 2

    @pytest.mark.parametrize(
        "where,mutate,expect",
        [
            (
                "report.md",
                lambda t: t + "\nNote: Today we are talking about caching layers in web services, and why\n",
                "transcript text",
            ),
            (
                "summary",
                lambda d: d["items"][0].update(quote="any verbatim excerpt from the episode"),
                "transcript/quote field",
            ),
            (
                "report.json",
                lambda d: d["meta"].update(key="sk-ant-api03-abcdefghijklmnopqrstuvwxyz"),
                "API key",
            ),
            (
                "report.json",
                lambda d: d["meta"].update(admin_token_value="AVD_ADMIN_TOKEN=hunter2hunter2"),
                "credential",
            ),
            ("report.json", lambda d: d["meta"].update(contact="alice@example.org"), "e-mail address"),
            (
                "report.json",
                lambda d: d["meta"].update(server="see http://10.0.0.7:11434/api"),
                "not loopback",
            ),
            (
                "report.json",
                lambda d: d["meta"].update(human_labels=[{"rater": "r1", "score": 4}]),
                "human labels",
            ),
            ("report.json", lambda d: d["meta"].update(status="verified"), "trust status"),
        ],
    )
    def test_forbidden_content_blocks_the_pack(
        self, bench: Bench, tmp_path: Path, where: str, mutate: Any, expect: str
    ) -> None:
        name = f"{_run_dir(bench)}/summary.json" if where == "summary" else where
        _edit(bench, name, mutate)
        assert not cm.inspect(bench.dir)["ok"]
        with pytest.raises(cm.CommunityError, match=expect):
            cm.pack(bench.dir, tmp_path / "p.zip")
        assert not (tmp_path / "p.zip").exists()

    def test_symlinks_are_refused(self, bench: Bench, tmp_path: Path) -> None:
        secret = tmp_path / "id_rsa"
        secret.write_text("-----BEGIN OPENSSH PRIVATE KEY-----")
        target = bench.dir / _run_dir(bench) / "report.md"
        target.unlink()
        os.symlink(secret, target)
        with pytest.raises(cm.CommunityError, match="symbolic link"):
            cm.pack(bench.dir, tmp_path / "p.zip")

    def test_pack_needs_the_transcript_cache(
        self, bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cm, "default_cache_dir", lambda: tmp_path / "nowhere")
        with pytest.raises(cm.CommunityError, match="cannot check for transcript text"):
            cm.pack(bench.dir, tmp_path / "p.zip")

    def test_items_without_confirmed_rights_are_not_shared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        b = _bench(tmp_path, monkeypatch, with_talkpython=True)
        out = cm.inspect(b.dir)
        assert any("talkpython" in f and "unconfirmed" in f for f in out["blocking_findings"])
        with pytest.raises(cm.CommunityError, match="unconfirmed uses"):
            cm.pack(b.dir, tmp_path / "p.zip")


class TestVerify:
    def test_mock_runs_are_refused(self, bench: Bench) -> None:
        v = cm.verify(bench.dir / "report.json")
        failed = {c["check"] for c in v["checks"] if not c["ok"]}
        assert v["accepted"] is False
        assert "real inference (not a mock)" in failed
        assert any("runtime version" in c for c in failed)  # the mock's version says so

    def test_http_mock_fails_the_real_inference_check_itself(self, bench: Bench) -> None:
        # A mock served over HTTP is a "real" transport; its runtime version is what gives it away.
        _as_real(bench)
        _edit(
            bench, "report.json", lambda r: r["models"][MODEL]["runtime"].update(ollama_version="0.0.0-mock")
        )
        v = cm.verify(bench.dir / "report.json")
        real = next(c for c in v["checks"] if c["check"] == "real inference (not a mock)")
        assert v["accepted"] is False and real["ok"] is False and "mock" in real["detail"]

    def test_structurally_valid_real_report_is_accepted_as_contributor_reported(
        self, bench: Bench, tmp_path: Path
    ) -> None:
        _as_real(bench)
        cm.pack(bench.dir, tmp_path / "p.zip")
        v = cm.verify(tmp_path / "p.zip")
        assert v["accepted"] and v["status"] == "contributor_reported", [
            c for c in v["checks"] if not c["ok"]
        ]
        assert "self-reported" in v["cannot_verify"][0] and "authorship" in v["cannot_verify"][1]
        names = " | ".join(c["check"] for c in v["checks"])
        for needed in (
            "weights digest",
            "quantization",
            "runtime version",
            "hardware",
            "prompt and rubric versions",
            "model calls made",
            "structured-output validity",
            "duration recorded",
            "git commit exists",
            "no transcript text",
            "shareable development item",
        ):
            assert needed in names, needed

    @pytest.mark.parametrize(
        "mutate,failing",
        [
            (lambda r: r["meta"].update(split="test"), "development split only"),
            (lambda r: r["meta"].update(manifest_items_sha256="0" * 64), "dataset matches this repository"),
            (lambda r: r["meta"].update(git_dirty=True), "git commit recorded and clean"),
            (lambda r: r["meta"].update(prompt="made_up_prompt"), "prompt and rubric versions"),
            (
                lambda r: r["models"][MODEL]["runtime"]["models"][MODEL].update(fingerprint="qwen"),
                "weights digest recorded",
            ),
            (
                lambda r: r["models"][MODEL]["runtime"]["models"][MODEL].update(quantization=None),
                "quantization",
            ),
            (
                lambda r: next(iter(r["models"][MODEL]["windows"].values())).update(evidence_verified=999),
                "numbers in range and consistent",
            ),
            (
                lambda r: next(iter(r["models"][MODEL]["windows"].values())).update(calls=0),
                "model calls made",
            ),
            (
                lambda r: r["models"][MODEL]["heuristic_side_by_side"]["rows"].append(
                    {"video_id": "chl_holdout_9"}
                ),
                "shareable development item",
            ),
        ],
    )
    def test_each_provenance_check(self, bench: Bench, mutate: Any, failing: str) -> None:
        p = _as_real(bench)
        r = json.loads(p.read_text())
        mutate(r)
        p.write_text(json.dumps(r))
        v = cm.verify(p)
        assert not v["accepted"] and any(failing in c["check"] and not c["ok"] for c in v["checks"])

    def test_unpacked_reports_with_local_paths_are_refused(self, bench: Bench) -> None:
        v = cm.verify(_as_real(bench))  # a raw report.json was never redacted by `pack`
        assert not v["accepted"] and any("local path" in c["detail"] for c in v["checks"] if not c["ok"])

    def test_unknown_commit_is_refused(self, bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cm, "_commit_known", lambda sha: False)
        v = cm.verify(_as_real(bench))
        assert not v["accepted"] and any("commit exists" in c["check"] and not c["ok"] for c in v["checks"])

    def test_tampered_or_malformed_packs_are_refused(self, bench: Bench, tmp_path: Path) -> None:
        _as_real(bench)
        cm.pack(bench.dir, tmp_path / "p.zip")
        data = zipfile.ZipFile(tmp_path / "p.zip")

        def rebuild(edit: Any) -> Path:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                edit(z, data)
            out = tmp_path / "t.zip"
            out.write_bytes(buf.getvalue())
            return out

        def tamper(z: zipfile.ZipFile, src: zipfile.ZipFile) -> None:
            for n in src.namelist():
                c = src.read(n)
                z.writestr(n, c.replace(b'"calls"', b'"calls" ') if n == "report.json" else c)

        def unlisted(z: zipfile.ZipFile, src: zipfile.ZipFile) -> None:
            for n in src.namelist():
                z.writestr(n, src.read(n))
            z.writestr("runs/extra/summary.json", "{}")

        def symlink(z: zipfile.ZipFile, src: zipfile.ZipFile) -> None:
            for n in src.namelist():
                z.writestr(n, src.read(n))
            info = zipfile.ZipInfo("runs/x/report.md")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")

        assert "does not match" in cm.verify(rebuild(tamper))["checks"][0]["detail"]
        assert "not in the pack manifest" in cm.verify(rebuild(unlisted))["checks"][0]["detail"]
        assert "symbolic link" in cm.verify(rebuild(symlink))["checks"][0]["detail"]
        for bad in ("../evil.json", "runs/x/run.db", "C:evil.json", "runs/x/items/a.json", "run.sh"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("report.json", "{}")
                z.writestr(bad, "x")
            (tmp_path / "bad.zip").write_bytes(buf.getvalue())
            assert cm.verify(tmp_path / "bad.zip")["accepted"] is False, bad

    def test_oversized_expansion_is_refused(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("report.json", "{}")
            for i in range(3):
                z.writestr(f"runs/r{i}/report.md", b"0" * (19 * 1024 * 1024))
        (tmp_path / "bomb.zip").write_bytes(buf.getvalue())
        v = cm.verify(tmp_path / "bomb.zip")
        assert not v["accepted"] and "50 MB" in v["checks"][0]["detail"]


class TestImport:
    def test_import_records_verdict_and_never_overwrites(self, bench: Bench, tmp_path: Path) -> None:
        _as_real(bench)
        cm.pack(bench.dir, tmp_path / "p.zip")
        out = cm.import_result(tmp_path / "p.zip", "volunteer-7", dest_root=tmp_path / "community")
        dest = Path(out["imported_to"])
        rec = json.loads((dest / "VERIFICATION.json").read_text())
        assert rec["status"] == "contributor_reported" and rec["contributor"] == "volunteer-7"
        assert rec["independently_reproduced"] is False and rec["human_labels"] == 0
        before = {p: p.read_bytes() for p in dest.rglob("*") if p.is_file()}
        with pytest.raises(cm.CommunityError, match="already exists"):
            cm.import_result(tmp_path / "p.zip", "volunteer-7", dest_root=tmp_path / "community")
        assert {p: p.read_bytes() for p in dest.rglob("*") if p.is_file()} == before

    def test_contributor_name_cannot_escape_the_directory(self, bench: Bench, tmp_path: Path) -> None:
        _as_real(bench)
        cm.pack(bench.dir, tmp_path / "p.zip")
        out = cm.import_result(tmp_path / "p.zip", "../../etc", dest_root=tmp_path / "community")
        assert Path(out["imported_to"]).parent == (tmp_path / "community").resolve()

    def test_import_refuses_mock_runs(self, bench: Bench, tmp_path: Path) -> None:
        with pytest.raises(cm.CommunityError, match="real inference"):
            cm.import_result(bench.dir / "report.json", "x", dest_root=tmp_path / "community")


def test_report_records_dataset_fingerprint_and_relative_paths(bench: Bench) -> None:
    r = json.loads((bench.dir / "report.json").read_text())
    meta = r["meta"]
    assert meta["transport_kind"] == "injected" and len(meta["manifest_items_sha256"]) == 64
    assert meta["prompt"] and meta["rubric_version"] and meta["manifest"] == "manifest.json"
    for w in r["models"][MODEL]["windows"].values():
        assert all(not d.startswith("/") for d in w["run_dirs"])


def test_smoke_runs_one_item_and_reports_the_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = _bench(tmp_path, monkeypatch, with_talkpython=True)
    fake = FakeOllama()
    out = cm.smoke(
        MODEL,
        tmp_path / "smoke",
        cache_dir=b.cache,
        manifest_path=b.manifest,
        base=make_settings(tmp_path),
        provider_factory=lambda st: _provider(fake, num_ctx=st.ollama_num_ctx),
    )
    assert out["transport"] == "injected"  # a mock run is labelled as such, never as real inference
    assert out["item"]["id"].startswith("chl_")  # never an item with unconfirmed rights
    assert out["model_calls"] >= 1 and out["structured_validity"] is not None
    assert out["model_digest"].startswith("sha256:") and out["quantization"] == "Q4_K_M"
    assert out["hardware"]["cpu_model"] and out["hardware"]["ram_total_gb"]
    assert out["evidence_verified"] <= out["evidence_claimed"] and "latency_ms_p50" in out
    assert out["prompt"] and out["rubric_version"]
    assert any("mock" in w for w in out["warnings"])  # the protocol mock says it is one
    assert (
        out["quality_control"]["items"] == 1 and out["quality_control"]["evidence_existence"]["claimed"] > 0
    )
    assert out["pack_dry_run_ok"], out["pack_blocking_findings"]
    plan = json.loads(next((tmp_path / "smoke" / "runs").iterdir()).joinpath("plan.json").read_text())
    assert plan["meta"]["items"] == [out["item"]["id"]]


def test_windows_and_macos_user_paths_are_redacted(bench: Bench, tmp_path: Path) -> None:
    win = r"C:\Users\alice\AppData\Local\Programs\Ollama\ollama.exe"
    mac = "/Users/alice/.ollama/models"

    def edit(d: dict[str, Any]) -> None:
        d["models"][MODEL]["runtime"]["hardware"]["ollama_binary"] = win
        d["meta"]["note"] = f"models in {mac}"

    _edit(bench, "report.json", edit)
    cm.pack(bench.dir, tmp_path / "p.zip")
    report = zipfile.ZipFile(tmp_path / "p.zip").read("report.json").decode()
    assert "alice" not in report and cm.REDACTED_PATH in report


class TestReview:
    """review.json: what the model said, shareable only within the source's quotation rights."""

    def test_run_writes_a_review_and_pack_shares_it(self, bench: Bench, tmp_path: Path) -> None:
        from avd.bench.run_review import quality_control, render_markdown

        review_path = bench.dir / _run_dir(bench) / "review.json"
        doc = json.loads(review_path.read_text())
        assert doc["format"] == "avd-run-review" and len(doc["items"]) == 2
        item = doc["items"][0]
        dims = item["evaluation"]["dimensions"]
        assert item["evaluation"]["evaluator_source"] == "local_ai"
        assert (
            item["evaluation"]["rights"]["quotes_included"] is True
        )  # Changelog: CC BY-SA, quotation confirmed
        assert any(d["status"] == "scored" and d["evidence"] for d in dims)
        assert any(item["justifications"].values())
        qc = quality_control(doc)
        assert qc["evidence_existence"]["claimed"] >= qc["evidence_existence"]["verified"] > 0
        assert "accurate" in " ".join(qc["not_established_here"])
        assert "| dimension | status |" in render_markdown(doc)
        cm.pack(bench.dir, tmp_path / "p.zip")
        assert any(n.endswith("/review.json") for n in zipfile.ZipFile(tmp_path / "p.zip").namelist())
        assert len(cm.load_reviews(tmp_path / "p.zip")) == 1

    def test_quotes_for_items_without_quotation_rights_are_refused(
        self, bench: Bench, tmp_path: Path
    ) -> None:
        def strip_rights(d: dict[str, Any]) -> None:
            d["items"][0]["evaluation"]["rights"]["quotes_included"] = False

        _edit(bench, f"{_run_dir(bench)}/review.json", strip_rights)
        with pytest.raises(cm.CommunityError, match="does not permit quotation"):
            cm.pack(bench.dir, tmp_path / "p.zip")

    def test_wholesale_transcript_text_is_refused(self, bench: Bench, tmp_path: Path) -> None:
        def paste(d: dict[str, Any]) -> None:
            first = next(iter(d["items"][0]["justifications"]))
            d["items"][0]["justifications"][first] = (CACHING_EPISODE + " ") * 40

        _edit(bench, f"{_run_dir(bench)}/review.json", paste)
        with pytest.raises(cm.CommunityError, match="quotes too much transcript text"):
            cm.pack(bench.dir, tmp_path / "p.zip")


def _repack(src: Path, out: Path, edit: Any = None, pack_extra: dict[str, Any] | None = None) -> Path:
    """Rebuild an accepted pack with edited members and a consistent PACK.json, so only the
    content checks (not the integrity check) can refuse it."""
    with zipfile.ZipFile(src) as z:
        members = {n: z.read(n) for n in z.namelist() if n != "PACK.json"}
        pack = json.loads(z.read("PACK.json"))
    if edit:
        members = edit(members)
    pack["files"] = {n: hashlib.sha256(d).hexdigest() for n, d in members.items()}
    pack.update(pack_extra or {})
    with zipfile.ZipFile(out, "w") as z:
        for n, d in members.items():
            z.writestr(n, d)
        z.writestr("PACK.json", json.dumps(pack))
    return out


class TestArchiveHardening:
    """Audit 2026-10-23: each case was accepted (or crashed verify) before."""

    @pytest.fixture
    def packed(self, bench: Bench, tmp_path: Path) -> Path:
        _as_real(bench)
        cm.pack(bench.dir, tmp_path / "p.zip")
        assert cm.verify(tmp_path / "p.zip")["accepted"]
        return tmp_path / "p.zip"

    def test_pack_manifest_cannot_carry_extra_content(self, packed: Path, tmp_path: Path) -> None:
        out = _repack(
            packed, tmp_path / "x.zip", pack_extra={"status": "verified", "contact": "alice@example.com"}
        )
        assert not cm.verify(out)["accepted"]

    @pytest.mark.parametrize(
        "value",
        [
            "alice\\u0040example.com",  # JSON escape hiding an e-mail address
            "http:\\/\\/192.168.1.20:11434\\/",  # escaped slashes hiding a LAN address
            "HTTP://192.168.1.5:11434/api",  # upper-case scheme
            "http://user:hunter2pass@10.0.0.2/x",  # credentials in the URL
            "ws://192.168.1.9:1/",  # another scheme
            "ollama at 10.0.0.5:11434",  # a bare private address
            "title\\u001b]0;pwned\\u0007",  # terminal escape sequence for `community review`
        ],
    )
    def test_hidden_or_unusual_private_content_is_found(
        self, packed: Path, tmp_path: Path, value: str
    ) -> None:
        def edit(m: dict[str, bytes]) -> dict[str, bytes]:
            raw = m["report.json"].decode()
            m["report.json"] = raw.replace('"meta": {', f'"meta": {{"note": "{value}", ', 1).encode()
            return m

        assert not cm.verify(_repack(packed, tmp_path / "x.zip", edit))["accepted"], value

    def test_a_review_item_without_a_video_id_is_refused(self, packed: Path, tmp_path: Path) -> None:
        name = next(n for n in zipfile.ZipFile(packed).namelist() if n.endswith("review.json"))

        def edit(m: dict[str, bytes]) -> dict[str, bytes]:
            doc = json.loads(m[name])
            doc["items"].append({"evaluation": {"rights": {"quotes_included": True}, "dimensions": []}})
            m[name] = json.dumps(doc).encode()
            return m

        v = cm.verify(_repack(packed, tmp_path / "x.zip", edit))
        assert not v["accepted"] and "without a video_id" in json.dumps(v["checks"])

    def test_without_the_transcript_cache_nothing_is_accepted(
        self, packed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cm, "default_cache_dir", lambda: tmp_path / "no-cache")
        v = cm.verify(packed)
        assert not v["accepted"] and any(
            c["check"] == "transcript-text check ran" and not c["ok"] for c in v["checks"]
        )

    def test_a_model_without_measurements_is_refused(self, packed: Path, tmp_path: Path) -> None:
        def edit(m: dict[str, bytes]) -> dict[str, bytes]:
            doc = json.loads(m["report.json"])
            for entry in doc["models"].values():
                entry["windows"] = {}
            m["report.json"] = json.dumps(doc).encode()
            return m

        assert not cm.verify(_repack(packed, tmp_path / "x.zip", edit))["accepted"]

    @pytest.mark.parametrize(
        "report",
        [
            "[]",
            '{"format": "avd-local-bench", "format_version": "1.0", "meta": {}, "models": {"m": null}}',
            '{"format": "avd-local-bench", "format_version": "1.0", "meta": {}, "models": {"m": {"windows": {"8": {"structured_validity": "0.5"}}}}}',
        ],
    )
    def test_malformed_reports_get_a_verdict_not_a_crash(self, tmp_path: Path, report: str) -> None:
        (tmp_path / "r.json").write_text(report)
        assert cm.verify(tmp_path / "r.json")["accepted"] is False
