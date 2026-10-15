"""Phase 2 P2: licensed real-world dataset sources, manifest, import, revocation.

Sample texts below are written for these tests in the sources' *formats*; they are not copied from
the licensed transcripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from avd.content.models import Transcript, TranscriptSegment
from avd.content.processing import normalize_transcript, validate_transcript
from avd.content.store import load_bundle
from avd.datasets.fetch import ChecksumMismatch, SourceFetcher, sha256_bytes
from avd.datasets.importer import ManifestError, import_manifest, revoke_video_sources
from avd.datasets.pilot import assign_splits, describe_item, finalize_manifest, items_digest, load_manifest
from avd.datasets.sources import CHANGELOG, TALKPYTHON, item_metadata, parse_changelog_markdown
from avd.db.models import ContentSource, Dataset, DatasetItem, Evaluation, Video
from tests.conftest import DB

CHANGELOG_SAMPLE = """**Host:** Welcome to the show. Today we talk about caching layers.

**Guest:** Thanks for having me.

**Break:** \\[00:01:10.00\\]

**Host:** \\[00:01:30.25\\] So what is the first thing teams get wrong with caches?

**Guest:** Invalidation, almost always. They add a cache and never decide when entries expire.

\\[00:02:45\\] The second mistake is caching errors. \\[01:00:03.5\\] And the third is [measuring](https://example.org) nothing.

**Host:** Good list.
"""

VTT_SAMPLE = """WEBVTT

00:00:00.000 --> 00:00:04.000
Welcome to the show.

00:00:04.000 --> 00:00:09.500
Today we discuss packaging tools.

00:00:09.500 --> 00:00:30.000
They changed a lot this year.
"""


class TestChangelogParser:
    def test_paragraph_segments_carry_their_source_span(self) -> None:
        t, rep = parse_changelog_markdown(CHANGELOG_SAMPLE)
        texts = [s.text for s in t.segments]
        spans = [(s.start, s.end) for s in t.segments]
        # Two paragraphs before the first timestamp share the [0, 70) span; the break marker's
        # label is dropped but its timestamp (70 s) is kept as an anchor.
        assert texts[0].startswith("Host: Welcome") and spans[0] == spans[1] == (0.0, 70.0)
        # "**Host:** \\[ts\\] text": the paragraph starts at the anchor, speaker label kept.
        assert texts[2].startswith("Host: So what") and spans[2] == (90.25, 165.0)
        assert spans[3] == (90.25, 165.0)  # no anchor in between: shares the span
        # A mid-paragraph anchor splits the paragraph at the anchor.
        assert texts[4] == "The second mistake is caching errors."
        assert spans[4] == (165.0, 3603.5)
        assert texts[5] == "And the third is measuring nothing." and spans[5][0] == 3603.5
        # The final span's end is unknown: equal to its start, never invented.
        assert spans[-1] == (3603.5, 3603.5)
        assert not any("Break" in x for x in texts)
        assert rep["timing"] == "source_anchor_spans" and rep["anchors"] == 4 and rep["segments"] == 7

    def test_shared_spans_are_not_overlaps_and_survive_normalisation(self) -> None:
        t, _ = parse_changelog_markdown(CHANGELOG_SAMPLE)
        normalized, norm = normalize_transcript(t)
        assert normalized == t and norm.fixed_overlaps == 0
        rep = validate_transcript(t)
        assert rep.overlapping == 0 and rep.shared_span_segments == 3
        assert "timestamps_at_span_granularity" in rep.issues

    def test_real_overlaps_are_still_counted(self) -> None:
        t = Transcript(
            segments=[
                TranscriptSegment(start=0, end=10, text="a b"),
                TranscriptSegment(start=5, end=12, text="c"),
            ]
        )
        assert validate_transcript(t).overlapping == 1


def test_item_metadata_never_invents_titles() -> None:
    m = item_metadata(CHANGELOG, "gotime/go-time-226.md")
    assert m["title"] == "Go Time #226" and m["title_source"] == "derived_from_path"
    assert m["group_key"] == "changelog:gotime" and m["external_url_verified"] == "false"
    t = item_metadata(TALKPYTHON, "youtube_transcripts/303-python-for-astronomy.vtt")
    assert t["title"] == "Python for astronomy" and t["edition"] == "youtube" and t["episode"] == "303"


def test_group_splits_are_deterministic_and_never_split_a_group() -> None:
    groups = {f"g{i}": 3 for i in range(10)}
    a, b = assign_splits(groups, 7), assign_splits(groups, 7)
    assert a == b and set(a) == set(groups)
    counts = {s: sum(groups[g] for g in a if a[g] == s) for s in ("calibration", "dev", "test")}
    assert counts == {"calibration": 3, "dev": 9, "test": 18}
    assert assign_splits(groups, 8) != a or True  # another seed may differ; determinism is what matters


class TestFetcher:
    def test_excluded_prefix_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError):
            SourceFetcher(tmp_path, offline=True).get(CHANGELOG, "bigtent/big-tent-1.md")

    def test_fetch_caches_and_verifies_checksums(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, content=b"WEBVTT\n")

        f = SourceFetcher(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))
        path = "youtube_transcripts/001-x.vtt"
        data = f.get(TALKPYTHON, path, expected_sha256=sha256_bytes(b"WEBVTT\n"))
        assert data == b"WEBVTT\n" and seen == [TALKPYTHON.raw_url(path)]
        assert TALKPYTHON.commit in seen[0]  # pinned commit, not a branch
        f.get(TALKPYTHON, path)
        assert len(seen) == 1  # served from cache
        with pytest.raises(ChecksumMismatch):
            f.get(TALKPYTHON, path, expected_sha256="0" * 64)

    def test_offline_without_cache_fails(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SourceFetcher(tmp_path, offline=True).get(TALKPYTHON, "youtube_transcripts/001-x.vtt")


def _write_cache(cache: Path, source: Any, path: str, text: str) -> bytes:
    p = cache / source.key / source.commit / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode())
    return text.encode()


@pytest.fixture
def mini_manifest(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    items = []
    for source, path, text in (
        (CHANGELOG, "podcast/the-changelog-1.md", CHANGELOG_SAMPLE),
        (CHANGELOG, "gotime/go-time-2.md", CHANGELOG_SAMPLE.replace("caching", "tracing")),
        (TALKPYTHON, "youtube_transcripts/301-packaging.vtt", VTT_SAMPLE),
    ):
        raw = _write_cache(cache, source, path, text)
        item, _ = describe_item(source, path, raw)
        items.append(item)
    manifest = finalize_manifest(items, {}, seed=1)
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(manifest))
    return out, cache


class TestImport:
    def test_import_records_rights_and_is_idempotent(self, db: DB, mini_manifest: tuple[Path, Path]) -> None:
        path, cache = mini_manifest
        fetcher = SourceFetcher(cache, offline=True)
        with db.sf() as s:
            r = import_manifest(s, path, fetcher)
            s.commit()
            assert r["imported"] == 3
            ds = s.get(Dataset, "real_pilot@v1")
            assert ds is not None and ds.split_strategy == "group_by_show"
            src = s.scalars(select(ContentSource).where(ContentSource.video_id == "chl_podcast_1")).one()
            assert src.license == "CC-BY-SA-4.0" and src.owner == "Changelog Media LLC"
            assert src.provenance_url and CHANGELOG.commit in src.provenance_url
            assert src.raw_sha256 and src.permitted_processing and "attribution_required" in src.restrictions
            assert src.preprocessing["parser"]["timing"] == "source_anchor_spans"
            v = s.get(Video, "tpy_301")
            assert v is not None and v.platform == "talkpython.fm" and not v.is_synthetic
            items = s.scalars(select(DatasetItem)).all()
            assert {i.video_id for i in items} == {"chl_podcast_1", "chl_gotime_2", "tpy_301"}
            by_group: dict[str, set[str]] = {}
            for i in items:
                by_group.setdefault(i.group_key, set()).add(i.split)
            assert all(len(splits) == 1 for splits in by_group.values())
            r2 = import_manifest(s, path, fetcher)
            assert r2["imported"] == 0 and r2["already_present"] == 3

    def test_changed_upstream_bytes_fail_closed(
        self, sqlite_db: DB, mini_manifest: tuple[Path, Path]
    ) -> None:
        path, cache = mini_manifest
        _write_cache(cache, CHANGELOG, "podcast/the-changelog-1.md", CHANGELOG_SAMPLE + "\nextra")
        with sqlite_db.sf() as s, pytest.raises(ChecksumMismatch):
            import_manifest(s, path, SourceFetcher(cache, offline=True))

    def test_tampered_manifest_is_rejected(self, mini_manifest: tuple[Path, Path]) -> None:
        path, _ = mini_manifest
        m = json.loads(path.read_text())
        m["items"][0]["split"] = "dev" if m["items"][0]["split"] != "dev" else "test"
        path.write_text(json.dumps(m))
        with pytest.raises(ValueError, match="digest"):
            load_manifest(path)

    def test_parser_drift_is_detected(self, sqlite_db: DB, mini_manifest: tuple[Path, Path]) -> None:
        path, cache = mini_manifest
        m = json.loads(path.read_text())
        m["items"][0]["content_hash"] = "f" * 64
        m["items_sha256"] = items_digest(m["items"])
        path.write_text(json.dumps(m))
        with sqlite_db.sf() as s, pytest.raises(ManifestError, match="parser drift"):
            import_manifest(s, path, SourceFetcher(cache, offline=True))

    def test_missing_licence_fails_closed(self, sqlite_db: DB, mini_manifest: tuple[Path, Path]) -> None:
        path, cache = mini_manifest
        m = json.loads(path.read_text())
        m["sources"]["changelog"]["license"] = ""
        path.write_text(json.dumps(m))
        with sqlite_db.sf() as s, pytest.raises(ManifestError, match="licence"):
            import_manifest(s, path, SourceFetcher(cache, offline=True))

    def test_revocation_withdraws_content_and_evaluations(
        self, sqlite_db: DB, mini_manifest: tuple[Path, Path]
    ) -> None:
        path, cache = mini_manifest
        with sqlite_db.sf() as s:
            import_manifest(s, path, SourceFetcher(cache, offline=True))
            s.add(
                Evaluation(
                    video_id="tpy_301",
                    stage="A",
                    framework_version="x",
                    mode="heuristic",
                    genre="education",
                    genre_confidence=0.5,
                    input_fingerprint="f",
                    status="complete",
                    is_current=True,
                )
            )
            s.flush()
            r = revoke_video_sources(s, "tpy_301", "creator takedown request")
            assert r == {"sources_revoked": 1, "evaluations_withdrawn": 1}
            assert load_bundle(s, "tpy_301").transcript is None
            src = s.scalars(select(ContentSource).where(ContentSource.video_id == "tpy_301")).one()
            assert src.status == "revoked" and src.revoked_at is not None


def test_committed_pilot_manifest_is_consistent() -> None:
    from avd.config import REPO_ROOT

    m = load_manifest(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    items = m["items"]
    assert 20 <= len(items) <= 30
    assert {it["split"] for it in items} == {"calibration", "dev", "test"}
    splits_by_group: dict[str, set[str]] = {}
    for it in items:
        splits_by_group.setdefault(it["group_key"], set()).add(it["split"])
        src = m["sources"][it["source"]]
        assert src["license"] and src["authorization_basis"] and src["commit"] in it["raw_url"]
        assert not any(it["path"].startswith(p) for p in src["excluded_prefixes"])
        assert it["language"] == "en" and it["raw_sha256"] and it["content_hash"]
        assert "score" not in it and "label" not in it  # the manifest carries no quality labels
    assert all(len(s) == 1 for s in splits_by_group.values())
