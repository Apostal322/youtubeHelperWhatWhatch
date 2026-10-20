"""Phase 3: use-time rights review — pending uses fail closed until a written confirmation exists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avd.annotation import service as svc
from avd.datasets import rights
from avd.datasets.fetch import SourceFetcher
from avd.datasets.importer import import_manifest
from avd.evaluation.service import serialize_evaluation
from tests.conftest import DB
from tests.test_datasets import mini_manifest  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _fresh_confirmations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    f = tmp_path / "rights_confirmations.json"
    f.write_text(json.dumps({"confirmations": []}))
    monkeypatch.setattr(rights, "CONFIRMATIONS_FILE", f)
    rights._confirmations.cache_clear()
    yield f
    rights._confirmations.cache_clear()


def test_reviews_separate_facts_from_assumptions() -> None:
    rep = rights.report()
    assert rep["changelog"]["pending"] == [] and rep["changelog"]["status"] == "explicit_content_licence"
    tp = rep["talkpython"]
    assert tp["status"] == "licence_scope_unconfirmed"
    assert set(tp["pending"]) == {"external_annotation", "public_quotes", "redistribution"}
    assert tp["facts"] and tp["assumptions"] and "rights_confirmations.json" in tp["remediation"]


def test_confirmation_requires_a_complete_record(_fresh_confirmations: Path) -> None:
    _fresh_confirmations.write_text(
        json.dumps(
            {"confirmations": [{"source": "talkpython", "uses": ["public_quotes"], "date": "2026-10-19"}]}
        )
    )
    rights._confirmations.cache_clear()
    assert "public_quotes" in rights.pending_uses("talkpython")  # no who/evidence → ignored
    _fresh_confirmations.write_text(
        json.dumps(
            {
                "confirmations": [
                    {
                        "source": "talkpython",
                        "uses": ["public_quotes", "external_annotation"],
                        "confirmed_by": "publisher",
                        "evidence": "email thread ref #1",
                        "date": "2026-10-19",
                    }
                ]
            }
        )
    )
    rights._confirmations.cache_clear()
    assert rights.pending_uses("talkpython") == {"redistribution"}


def test_unconfirmed_items_are_not_annotated_and_quotes_are_withheld(
    sqlite_db: DB,
    mini_manifest: tuple[Path, Path],  # noqa: F811
) -> None:
    path, cache = mini_manifest
    with sqlite_db.sf() as s:
        import_manifest(s, path, SourceFetcher(cache, offline=True))
        s.flush()
        assert rights.video_pending_uses(s, "tpy_301") >= {"external_annotation", "public_quotes"}
        assert rights.video_pending_uses(s, "chl_podcast_1") == set()
        assert svc._available(s, "chl_podcast_1") and not svc._available(s, "tpy_301")
        r, _ = svc.create_rater(s, "r1", [], "rater")
        r.status = "active"
        seen = set()
        while (a := svc.next_assignment(s, r, "real_pilot@v1")) is not None:
            seen.add(a.video_id)
            a.status = "submitted"
            s.flush()
        assert "tpy_301" not in seen and seen  # Changelog items are still assigned


def test_withhold_quotes_keeps_positions() -> None:
    payload = {
        "dimensions": [
            {"evidence": [{"quote": "verbatim text", "segment_index": 3, "verified": True}, {"quote": None}]}
        ]
    }
    out = rights.withhold_quotes(payload)
    ev = out["dimensions"][0]["evidence"]
    assert ev[0]["quote"] == rights.WITHHELD and ev[0]["segment_index"] == 3 and ev[0]["verified"]
    assert ev[1]["quote"] is None and out["quotes_withheld"] is True
    assert callable(serialize_evaluation)
