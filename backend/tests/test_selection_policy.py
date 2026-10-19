"""Selection-policy versioning: every impression says which selection configuration produced it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select, text

import avd
from avd.embeddings.base import HashingEmbedder
from avd.recsys import policy
from avd.recsys.ranker import DEFAULT_WEIGHTS, LinearRanker
from avd.recsys.recommender import DEFAULT_RANKER
from tests.conftest import DB, _make_db

# Pinned on purpose. If this fails, the selection behaviour changed: bump ALGORITHM_REVISION when
# the code changed, describe the change in avd/recsys/policy.py, then update this value.
EXPECTED_HASHING_DEFAULT = "two_stage_mmr_r2:1628b579786d"


def _version(**over: Any) -> str:
    return policy.policy_version(
        policy.policy_config(over.get("ranker", DEFAULT_RANKER), over.get("emb", HashingEmbedder()))
    )


def test_default_policy_version_is_pinned() -> None:
    assert _version() == EXPECTED_HASHING_DEFAULT


def test_version_changes_with_every_selection_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _version()
    # ranker weights, even under the same version name
    tweaked = LinearRanker(weights={**DEFAULT_WEIGHTS, "quality": 0.31}, version=DEFAULT_RANKER.version)
    assert _version(ranker=tweaked) != base
    # embedder (and with it the relevance threshold)
    floor = HashingEmbedder()
    floor.relevance_floor = 0.07  # type: ignore[misc]
    assert _version(emb=floor) != base
    for name, value in (
        ("MAX_PER_CHANNEL", 3),
        ("CHANNEL_CAP", "hard"),
        ("EXPLORATION_SHARE", 4),
        ("RETRIEVAL_PER_INTEREST", 80),
        ("QUALITY_CANDIDATES", 50),
        ("ALGORITHM_REVISION", 3),
    ):
        with monkeypatch.context() as m:
            m.setattr(policy, name, value)
            assert _version() != base, name
    assert _version() == base


@pytest.fixture
def evaluated(tmp_path: Path) -> DB:
    from avd.services import seed_demo
    from avd.workers.context import build_context
    from avd.workers.runner import run_until_idle

    db = _make_db(tmp_path, None)
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    run_until_idle(build_context(settings=db.settings, session_factory=db.sf, embedder=HashingEmbedder()))
    return db


def _user(db: DB, **prefs: Any) -> str:
    from avd.db.models import User, UserPreferences

    with db.sf() as s:
        u = User(token_hash=f"t{len(prefs)}{id(prefs)}")
        u.preferences = UserPreferences(**prefs)
        s.add(u)
        s.commit()
        return u.id


def test_impressions_record_policy_and_context(evaluated: DB) -> None:
    from avd.db.models import RecommendationImpression, RecommendationPolicy
    from avd.recsys.recommender import recommend

    uid = _user(evaluated, interests=["physics of everyday things"], diversity=0.4)
    emb = HashingEmbedder()
    with evaluated.sf() as s:
        first = recommend(s, uid, emb, limit=6)
        second = recommend(s, uid, emb, limit=6)
        s.commit()
    assert first["selection_policy_version"] == second["selection_policy_version"] == EXPECTED_HASHING_DEFAULT
    assert all(it["ranking"]["selection_policy_version"] == EXPECTED_HASHING_DEFAULT for it in first["items"])
    with evaluated.sf() as s:
        policies = s.scalars(select(RecommendationPolicy)).all()
        imps = s.scalars(select(RecommendationImpression)).all()
    assert [p.version for p in policies] == [EXPECTED_HASHING_DEFAULT]  # stored once
    assert policies[0].config["channel_cap"] == {"mode": "soft", "max_per_channel": 2}
    assert policies[0].config["ranker"]["version"] == "linear_v1"
    assert len(imps) == len(first["items"]) + len(second["items"]) > 0
    for imp in imps:
        assert imp.selection_policy_version == EXPECTED_HASHING_DEFAULT
        ctx = imp.selection_context or {}
        assert ctx["diversity"] == 0.4 and ctx["limit"] == 6 and ctx["has_interests"] is True
        assert {"exploration", "channel_cap_relaxed", "mmr_penalty"} <= set(ctx)


def test_new_configuration_adds_a_row_and_keeps_the_old_one(evaluated: DB) -> None:
    from avd.db.models import RecommendationPolicy
    from avd.recsys.recommender import recommend

    uid = _user(evaluated, interests=["astronomy"])
    emb = HashingEmbedder()
    other = LinearRanker(weights={**DEFAULT_WEIGHTS, "relevance": 0.5}, version="linear_v1_test")
    with evaluated.sf() as s:
        a = recommend(s, uid, emb, limit=4)["selection_policy_version"]
        b = recommend(s, uid, emb, limit=4, ranker=other)["selection_policy_version"]
        s.commit()
        rows = {p.version: p.config for p in s.scalars(select(RecommendationPolicy)).all()}
    assert a != b and set(rows) == {a, b}
    assert rows[a]["ranker"]["version"] == "linear_v1" and rows[b]["ranker"]["version"] == "linear_v1_test"


def test_no_policy_row_without_impressions(evaluated: DB) -> None:
    from avd.db.models import RecommendationPolicy
    from avd.recsys.recommender import recommend

    uid = _user(evaluated, interests=["astronomy"])
    with evaluated.sf() as s:
        out = recommend(s, uid, HashingEmbedder(), limit=4, log_impressions=False)
        s.commit()
        assert out["selection_policy_version"] == EXPECTED_HASHING_DEFAULT
        assert s.scalars(select(RecommendationPolicy)).all() == []


def test_migration_keeps_historical_impressions_unversioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Impressions logged before 0008 stay NULL: their policy is unknown and is not guessed."""
    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("AVD_DATABASE_URL", url)
    from avd.config import get_settings

    get_settings.cache_clear()
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(avd.__file__).resolve().parent / "migrations"))
    command.upgrade(cfg, "0007")
    eng = create_engine(url)
    with eng.begin() as c:  # plain engine: SQLite does not enforce the foreign keys here
        c.execute(
            text(
                "INSERT INTO recommendation_impressions (id, request_id, user_id, video_id, position, "
                "ranker_version, relevance, rank_score, features, created_at) "
                "VALUES ('i1', 'r1', 'u1', 'v1', 0, 'linear_v1', 0.5, 0.5, '{}', '2026-01-01')"
            )
        )
    command.upgrade(cfg, "head")
    with eng.connect() as c:
        row = c.execute(
            text(
                "SELECT ranker_version, selection_policy_version, selection_context FROM recommendation_impressions"
            )
        ).one()
        assert tuple(row) == ("linear_v1", None, None)
        assert c.execute(text("SELECT count(*) FROM recommendation_policies")).scalar() == 0
    eng.dispose()
    get_settings.cache_clear()


def test_relaxed_cap_is_recorded_per_item() -> None:
    from avd.recsys.recommender import _mmr

    def cand(vid: str, ch: str, rank: float) -> dict[str, Any]:
        return {"video": SimpleNamespace(id=vid, channel_id=ch, embedding=None), "rank": rank}

    picked = _mmr([cand(f"v{i}", "only", 1.0 - i / 10) for i in range(4)], 4, 0.0)
    assert [bool(c.get("channel_cap_relaxed")) for c in picked] == [False, False, True, True]
