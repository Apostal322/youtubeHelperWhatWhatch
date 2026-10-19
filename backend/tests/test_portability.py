"""Phase 3.2: portable evaluation exports (format v1), import as untrusted, data ownership."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from avd import portability as pt
from avd.db.models import ContentSource, Evaluation, ImportedEvaluation
from tests.conftest import DB, _make_db


@pytest.fixture
def demo_db(tmp_path: Path) -> DB:
    from avd.services import seed_demo
    from avd.workers.context import build_context
    from avd.workers.runner import run_until_idle

    db = _make_db(tmp_path, None)
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    run_until_idle(build_context(settings=db.settings, session_factory=db.sf))
    return db


def _export(db: DB, **kw: Any) -> dict[str, Any]:
    with db.sf() as s:
        return pt.export_evaluations(s, **kw)


class TestExport:
    def test_document_shape_and_provenance(self, demo_db: DB) -> None:
        doc = _export(demo_db)
        assert doc["format"] == "avd-evaluation-export" and doc["format_version"] == "1.0"
        assert len(doc["evaluations"]) == 20
        ev = doc["evaluations"][0]
        assert ev["evaluator_source"] == "synthetic" and ev["video"]["is_synthetic"] is True
        assert {"framework", "rubric", "prompt"} <= set(ev["versions"])
        assert ev["config_hash"] and ev["source_fingerprint"] and ev["dimensions"]
        assert all("uncertainty" in d for d in ev["dimensions"])
        assert doc["integrity"]["evaluations_sha256"] == pt.sha256(pt.canonical(doc["evaluations"]))
        assert "not validated" in doc["notice"]

    def test_no_transcript_and_no_quotes_by_default(self, demo_db: DB) -> None:
        doc = _export(demo_db)
        text = json.dumps(doc)
        with demo_db.sf() as s:
            src = s.scalars(select(ContentSource).where(ContentSource.transcript.is_not(None))).first()
            assert src is not None
            seg = src.transcript["segments"][3]["text"]  # type: ignore[index]
        assert seg not in text
        refs = [e for ev in doc["evaluations"] for d in ev["dimensions"] for e in d["evidence"]]
        hashed = [e for e in refs if e["quote_sha256"]]
        assert (
            hashed
            and all(e["quote"] is None for e in refs)
            and all(len(e["quote_sha256"]) == 64 for e in hashed)
        )
        assert all(ev["rights"]["transcript_included"] is False for ev in doc["evaluations"])

    def test_quotes_only_where_every_source_permits(self, demo_db: DB) -> None:
        doc = _export(demo_db, include_quotes=True)
        assert all(ev["rights"]["quotes_included"] for ev in doc["evaluations"])  # synthetic fixtures (CC0)
        with demo_db.sf() as s:
            vid = doc["evaluations"][0]["video"]["video_id"]
            s.scalars(select(ContentSource).where(ContentSource.video_id == vid)).first().source_type = (  # type: ignore[union-attr]
                "user_provided_transcript"
            )
            s.commit()
        doc = _export(demo_db, include_quotes=True)
        ev = next(e for e in doc["evaluations"] if e["video"]["video_id"] == vid)
        assert (
            ev["rights"]["quotes_included"] is False
            and "user_provided_transcript" in ev["rights"]["quotes_reason"]
        )
        assert all(e["quote"] is None for d in ev["dimensions"] for e in d["evidence"])

    def test_blinded_and_synthetic_filters(self, demo_db: DB, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _export(demo_db, include_synthetic=False)["evaluations"] == []
        first = _export(demo_db)["evaluations"][0]["video"]["video_id"]
        monkeypatch.setattr("avd.datasets.blinding.blinded_video_ids", lambda s: {first})
        assert first not in {e["video"]["video_id"] for e in _export(demo_db)["evaluations"]}


class TestImport:
    def _raw(self, db: DB, **kw: Any) -> bytes:
        return json.dumps(_export(db, **kw)).encode()

    def test_round_trip_is_stored_apart_and_untrusted(self, demo_db: DB, tmp_path: Path) -> None:
        raw = self._raw(demo_db, include_quotes=True)
        (tmp_path / "other").mkdir()
        other = _make_db(tmp_path / "other", None)
        with other.sf() as s:
            out = pt.import_evaluations(s, raw, origin="friend's instance")
            s.commit()
            assert out["imported"] == 20 and out["trusted"] is False
            assert s.scalar(select(func.count()).select_from(Evaluation)) == 0  # local evaluations untouched
            again = pt.import_evaluations(s, raw, origin="friend's instance")
            assert again == {**again, "imported": 0, "already_present": 20}
            vid = json.loads(raw)["evaluations"][0]["video"]["video_id"]
            rows = pt.imported_for_video(s, vid)
            assert rows[0]["status"] == "imported_unverified" and rows[0]["origin"] == "friend's instance"
            assert s.scalar(select(func.count()).select_from(ImportedEvaluation)) == 20

    @pytest.mark.parametrize(
        "mutate,message",
        [
            (lambda d: d.update(format="other"), "not an avd-evaluation-export"),
            (lambda d: d.update(format_version="2.0"), "unsupported format version"),
            (lambda d: d["evaluations"][0]["dimensions"][0].update(score=9), "invalid export"),
            (lambda d: d["evaluations"][0].update(unexpected=1), "invalid export"),
            (lambda d: d["evaluations"][0]["overall"].update(score=1.0), "integrity check failed"),
        ],
    )
    def test_tampered_or_invalid_files_are_refused(self, demo_db: DB, mutate: Any, message: str) -> None:
        doc = _export(demo_db)
        mutate(doc)
        with pytest.raises(pt.ImportRefused, match=message):
            pt.validate_export(json.dumps(doc).encode())

    def test_quote_not_matching_its_hash_is_refused(self, demo_db: DB) -> None:
        doc = _export(demo_db, include_quotes=True)
        e = next(
            e for ev in doc["evaluations"] for d in ev["dimensions"] for e in d["evidence"] if e["quote"]
        )
        e["quote"] = "words the speaker never said"
        doc["integrity"]["evaluations_sha256"] = pt.sha256(
            pt.canonical(doc["evaluations"])
        )  # re-hashed by a forger
        with pytest.raises(pt.ImportRefused, match="does not match its hash"):
            pt.validate_export(json.dumps(doc).encode())

    def test_garbage_and_oversize(self) -> None:
        with pytest.raises(pt.ImportRefused, match="not JSON"):
            pt.validate_export(b"{nope")
        with pytest.raises(pt.ImportRefused, match="larger than"):
            pt.validate_export(b" " * (pt.MAX_IMPORT_BYTES + 1))


class TestDataOwnership:
    def test_sqlite_backup_restore_and_clear(self, demo_db: DB, tmp_path: Path) -> None:
        from avd import datactl

        backup = tmp_path / "backup.db"
        info = datactl.backup_sqlite(demo_db.settings.database_url, backup)
        assert backup.exists() and info["bytes"] > 0
        with demo_db.sf() as s:
            n = datactl.clear(s, evaluations=True)
            s.commit()
            assert n["evaluations"] == 20 and s.scalar(select(func.count()).select_from(Evaluation)) == 0
        restored = tmp_path / "restored.db"
        datactl.restore_sqlite(backup, f"sqlite:///{restored}")
        from sqlalchemy.orm import Session

        from avd.db.session import make_engine

        eng = make_engine(f"sqlite:///{restored}")
        with Session(eng) as s:
            assert s.scalar(select(func.count()).select_from(Evaluation)) == 20
        eng.dispose()

    def test_restore_refuses_to_overwrite(self, tmp_path: Path) -> None:
        from avd import datactl

        target = tmp_path / "x.db"
        target.write_bytes(b"existing")
        (tmp_path / "b.db").write_bytes(b"SQLite format 3\x00")
        with pytest.raises(datactl.DataError, match="already exists"):
            datactl.restore_sqlite(tmp_path / "b.db", f"sqlite:///{target}")

    def test_user_data_export(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        from avd.config import get_settings
        from avd.db.models import Base
        from avd.db.session import get_engine, get_sessionmaker

        monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{tmp_path / 'u.db'}")
        for f in (get_settings, get_engine, get_sessionmaker):
            f.cache_clear()
        Base.metadata.create_all(get_engine())
        from avd.api.app import create_app

        with TestClient(create_app()) as c:
            token = c.post("/v1/users").json()["token"]
            h = {"Authorization": f"Bearer {token}"}
            c.put("/v1/users/me/preferences", json={"interests": ["sourdough"]}, headers=h)
            out = c.get("/v1/users/me/export", headers=h).json()
            assert out["format"] == "avd-user-data-export" and out["preferences"]["interests"] == [
                "sourdough"
            ]
            assert token not in json.dumps(out) and "token_hash" not in json.dumps(out)
        get_engine().dispose()
        for f in (get_settings, get_engine, get_sessionmaker):
            f.cache_clear()
