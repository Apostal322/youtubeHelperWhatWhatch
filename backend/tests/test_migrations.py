"""Migrations apply, match the ORM models, and roll back — on SQLite and (when configured) PostgreSQL."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

import avd
from avd.db.models import Base
from tests.conftest import PG_URL

MIGRATIONS = Path(avd.__file__).resolve().parent / "migrations"


def _cfg() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    return cfg


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def db_url(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "postgres":
        if not PG_URL:
            pytest.skip("AVD_TEST_DATABASE_URL not set")
        url = PG_URL
        eng = create_engine(url)
        with eng.begin() as c:
            Base.metadata.drop_all(c)
            c.execute(text("DROP TABLE IF EXISTS alembic_version"))
        eng.dispose()
    else:
        url = f"sqlite:///{tmp_path / 'mig.db'}"
    monkeypatch.setenv("AVD_DATABASE_URL", url)
    from avd.config import get_settings

    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


def test_upgrade_matches_models_and_downgrade_is_clean(db_url: str) -> None:
    command.upgrade(_cfg(), "head")
    eng = create_engine(db_url)
    with eng.connect() as conn:
        diff = [
            d
            for d in compare_metadata(MigrationContext.configure(conn), Base.metadata)
            if not (d[0] == "remove_index" and d[1].name == "ix_videos_embedding_hnsw")
        ]
    assert diff == []
    command.downgrade(_cfg(), "base")
    assert set(inspect(eng).get_table_names()) <= {"alembic_version"}
    eng.dispose()
