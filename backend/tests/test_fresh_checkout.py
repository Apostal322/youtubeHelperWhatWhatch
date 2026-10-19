"""A volunteer's first commands on a fresh checkout work without a manual `avd migrate`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_evaluate_real import manifest  # noqa: F401


def test_dataset_import_creates_the_schema_on_a_new_database(
    tmp_path: Path,
    manifest: tuple[Path, Path],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from avd.cli import main
    from avd.config import get_settings
    from avd.db.session import get_engine, get_sessionmaker

    path, cache = manifest
    monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{tmp_path / 'fresh.db'}")  # no tables yet
    for f in (get_settings, get_engine, get_sessionmaker):
        f.cache_clear()
    try:
        main(["dataset", "import", str(path), "--cache-dir", str(cache), "--offline"])
        out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert out["imported"] == 3
    finally:
        get_engine().dispose()
        for f in (get_settings, get_engine, get_sessionmaker):
            f.cache_clear()
