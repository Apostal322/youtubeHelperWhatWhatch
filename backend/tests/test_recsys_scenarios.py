"""Soft per-channel cap and candidate selection under different catalog shapes and user types.

Synthetic candidates and the synthetic demo catalog only: these test the selection policy's
behaviour, not recommendation quality.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from avd.embeddings.base import HashingEmbedder
from avd.recsys.policy import EXPLORATION_SHARE, MAX_PER_CHANNEL
from avd.recsys.recommender import _mmr
from tests.conftest import DB, _make_db


def cand(vid: str, ch: str, rank: float) -> dict[str, Any]:
    return {"video": SimpleNamespace(id=vid, channel_id=ch, embedding=None), "rank": rank}


def per_channel(sel: list[dict[str, Any]]) -> Counter[str]:
    return Counter(c["video"].channel_id for c in sel)


def relaxed(sel: list[dict[str, Any]]) -> int:
    return sum(1 for c in sel if c.get("channel_cap_relaxed"))


def test_small_catalog_returns_what_exists_without_padding() -> None:
    sel = _mmr([cand(f"v{i}", "a", 1 - i / 10) for i in range(3)], 5, 0.0)
    assert len(sel) == 3  # never more than the catalog holds
    assert relaxed(sel) == 1  # only the third item needed the cap relaxed


def test_large_catalog_never_relaxes_the_cap() -> None:
    rng = random.Random(1)
    pool = [cand(f"v{i}", f"c{i % 50}", rng.random()) for i in range(200)]
    sel = _mmr(pool, 20, 0.3)
    assert len(sel) == 20 and relaxed(sel) == 0
    assert max(per_channel(sel).values()) <= MAX_PER_CHANNEL


def test_one_dominant_channel_spreads_first_then_relaxes_only_for_leftover_slots() -> None:
    pool = [cand(f"a{i}", "A", 0.99 - i / 100) for i in range(20)]
    pool += [cand(f"b{i}", "B", 0.5 - i / 100) for i in range(3)]
    pool += [cand(f"c{i}", "C", 0.4 - i / 100) for i in range(2)]
    sel = _mmr(pool, 8, 0.0)
    first_six = per_channel(sel[:6])
    assert first_six == Counter({"A": 2, "B": 2, "C": 2})  # every creator gets a turn first
    assert relaxed(sel) == 2 and all(c.get("channel_cap_relaxed") for c in sel[6:])
    assert relaxed(sel[:6]) == 0


def test_many_small_channels_need_no_relaxation() -> None:
    sel = _mmr([cand(f"v{i}", f"c{i}", 1 - i / 100) for i in range(30)], 10, 0.3)
    assert len(sel) == 10 and relaxed(sel) == 0 and set(per_channel(sel).values()) == {1}


# ------------------------------------------------------------------- full recommender, demo catalog


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> DB:
    from avd.services import seed_demo
    from avd.workers.context import build_context
    from avd.workers.runner import run_until_idle

    db = _make_db(Path(tmp_path_factory.mktemp("scen")), None)
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    run_until_idle(build_context(settings=db.settings, session_factory=db.sf, embedder=HashingEmbedder()))
    return db


def _recommend(db: DB, limit: int = 10, **prefs: Any) -> tuple[dict[str, Any], dict[str, str]]:
    from avd.db.models import User, UserPreferences
    from avd.recsys.recommender import recommend

    with db.sf() as s:
        u = User(token_hash=f"scen-{prefs}-{limit}")
        if prefs:
            u.preferences = UserPreferences(**prefs)
        s.add(u)
        s.flush()
        trace: dict[str, str] = {}
        out = recommend(s, u.id, HashingEmbedder(), limit=limit, log_impressions=False, trace=trace)
        s.rollback()
    return out, trace


@pytest.mark.parametrize(
    "interests",
    [
        ["photosynthesis and the enzyme RuBisCO"],  # highly specific
        ["videos"],  # ambiguous
        ["zxqv blorf quantum basket weaving"],  # matches nothing
        ["physics of everyday things", "programming data structures"],  # several needs
    ],
)
def test_feed_is_never_padded_with_off_topic_items(demo: DB, interests: list[str]) -> None:
    limit = 10
    out, trace = _recommend(demo, limit, interests=interests)
    explore = [it for it in out["items"] if it["ranking"]["exploration"]]
    topical = [it for it in out["items"] if not it["ranking"]["exploration"]]
    assert len(explore) <= max(1, limit // EXPLORATION_SHARE)  # the only slots below the floor
    assert all(it["relevance"]["normalized"] > 0 for it in topical)
    assert all("Exploration pick" in " ".join(it["explanations"]) for it in explore)
    # Every retrieved candidate has a recorded fate, and the selected ones are exactly the feed.
    selected = {v for v, f in trace.items() if f.startswith("selected")}
    assert selected == {it["video"]["id"] for it in out["items"]}


def test_cold_start_user_gets_quality_candidates_without_interest_claims(demo: DB) -> None:
    out, _ = _recommend(demo, 8)  # no preferences at all
    assert out["items"] and not any(it["ranking"]["exploration"] for it in out["items"])
    for it in out["items"]:
        assert it["relevance"]["matched_interest"] is None
        assert not any(e.startswith("Matches your interest") for e in it["explanations"])
    assert out["selection_policy_version"].startswith("two_stage_mmr_r")


def test_explanations_name_a_relaxed_cap(demo: DB) -> None:
    out, _ = _recommend(demo, 20, interests=["science"])
    for it in out["items"]:
        if it["ranking"]["channel_cap_relaxed"]:
            assert any("channel already shown" in e for e in it["explanations"])


def test_a_short_feed_says_why_it_is_short(demo: DB) -> None:
    out, _ = _recommend(demo, 10, interests=["zxqv blorf quantum basket weaving"])
    assert any("not filled up with unrelated videos" in n for n in out["notes"])


def test_items_hidden_for_blind_annotation_are_explained(tmp_path: Path) -> None:
    """Imported dataset items are withheld while blind annotation is open; the feed must say so
    (it used to claim "no analyzed videos match your filters")."""
    from avd.db.models import Dataset, DatasetItem, User, UserPreferences
    from avd.recsys.recommender import recommend
    from avd.services import seed_demo

    db = _make_db(tmp_path, None)
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.add(
            Dataset(
                id="pilot@v1",
                name="pilot",
                version="v1",
                manifest_sha256="0" * 64,
                split_strategy="x",
                split_seed=1,
            )
        )
        s.flush()
        s.add(DatasetItem(dataset_id="pilot@v1", video_id="syn_sci_001", split="dev", group_key="g"))
        u = User(token_hash="blind")
        u.preferences = UserPreferences(interests=["science"])
        s.add(u)
        s.flush()
        out = recommend(s, u.id, HashingEmbedder(), limit=5, log_impressions=False)
    assert any(
        "hidden while blind human annotation of pilot@v1" in n and "avd annotate unblind" in n
        for n in out["notes"]
    )
    assert "syn_sci_001" not in {it["video"]["id"] for it in out["items"]}


def test_empty_catalog_and_no_match_are_told_apart(tmp_path: Path, demo: DB) -> None:
    from avd.db.models import User, UserPreferences
    from avd.recsys.recommender import recommend

    db = _make_db(tmp_path, None)  # nothing imported
    with db.sf() as s:
        u = User(token_hash="empty")
        u.preferences = UserPreferences(interests=["science"])
        s.add(u)
        s.flush()
        out = recommend(s, u.id, HashingEmbedder(), limit=5, log_impressions=False)
    assert out["empty_reason"] == "empty_catalog" and "catalog is empty" in out["notes"][0]
    filtered, _ = _recommend(demo, 5, interests=["science"], languages=["xx"])
    assert filtered["items"] == [] and filtered["empty_reason"] == "no_match"


def test_a_list_emptied_by_filters_says_so(demo: DB) -> None:
    out, _ = _recommend(demo, 5, min_quality=4.99)
    assert out["items"] == [] and out["empty_reason"] == "no_match"  # before: None and no note
    assert "filtered out" in out["notes"][0] and "below min quality" in out["notes"][0]


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [r["video"]["id"] for r in rows]


@pytest.mark.parametrize("query", ["science", "how computers work", "history and culture"])
def test_search_results_do_not_depend_on_the_page_size(demo: DB, query: str) -> None:
    from avd.recsys.search import search

    with demo.sf() as s:
        for mq in (3.0, 3.5):
            full = _ids(search(s, HashingEmbedder(), query, limit=100, min_quality=mq))
            for k in (1, 2, 3):
                # Before: filtered-out videos used up the limit*4 window, so small pages came back empty.
                assert set(_ids(search(s, HashingEmbedder(), query, limit=k, min_quality=mq))) <= set(full)
                assert len(search(s, HashingEmbedder(), query, limit=k, min_quality=mq)) == min(k, len(full))


def test_search_lists_a_video_once_and_ignores_function_words(demo: DB) -> None:
    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy import select

    from avd.db.models import Evaluation
    from avd.recsys.search import search

    with demo.sf() as s:
        ev = s.scalars(select(Evaluation).where(Evaluation.is_current.is_(True))).first()
        assert ev is not None
        cols = {
            a.key: getattr(ev, a.key) for a in sa_inspect(Evaluation).mapper.column_attrs if a.key != "id"
        }
        s.add(Evaluation(**cols))  # two "current" evaluations for one video
        s.flush()
        ids = _ids(search(s, HashingEmbedder(), "science history technology", limit=20))
        assert len(ids) == len(set(ids))  # before: the video was listed twice
        rows = search(s, HashingEmbedder(), "what and the", limit=20)
        assert all(r["match"]["title_match"] == 0 for r in rows)  # before: matched most titles
        s.rollback()
