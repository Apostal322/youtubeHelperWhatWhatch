"""Metrics, rankers, benchmark harness and Celery transport."""

from __future__ import annotations

import math

import numpy as np
import pytest

from avd.bench import metrics as M
from avd.config import REPO_ROOT
from avd.recsys.ranker import (
    DEFAULT_WEIGHTS,
    FEATURES,
    LinearRanker,
    pairwise_accuracy,
    train_pairwise_ranker,
)
from tests.conftest import DB, FIXTURES, REDIS_URL


class TestMetrics:
    def test_rank_correlations(self) -> None:
        assert M.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert M.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
        assert M.kendall_tau_b([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
        assert math.isnan(M.spearman([1, 1, 1], [1, 2, 3]))

    def test_pairwise_accuracy_handles_ties_and_abstentions(self) -> None:
        r = M.pairwise_accuracy(
            {"a": 4, "b": 2, "c": None, "d": 2},
            [("a", "b", "a"), ("a", "c", "a"), ("b", "d", "b"), ("a", "b", "tie")],
        )
        assert r["decided"] == 2 and r["abstained"] == 1 and r["accuracy"] == pytest.approx(0.75)

    def test_krippendorff_alpha(self) -> None:
        perfect = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=float)
        assert M.krippendorff_alpha_interval(perfect) == pytest.approx(1.0)
        noisy = np.array([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]], dtype=float)
        assert M.krippendorff_alpha_interval(noisy) < 0.2
        with_missing = np.array([[1, 2, np.nan], [1, 2, 3]], dtype=float)
        assert M.krippendorff_alpha_interval(with_missing) == pytest.approx(1.0)

    def test_ece(self) -> None:
        assert M.expected_calibration_error([0.9] * 10, [True] * 9 + [False])["ece"] == pytest.approx(
            0.0, abs=1e-9
        )
        assert M.expected_calibration_error([0.9] * 10, [False] * 10)["ece"] == pytest.approx(0.9)

    def test_ranking_metrics(self) -> None:
        assert M.ndcg_at_k(["a", "b", "c"], {"a", "b"}, 2) == pytest.approx(1.0)
        assert M.ndcg_at_k(["c", "a", "b"], {"a"}, 3) == pytest.approx(1 / math.log2(3))
        assert M.precision_at_k(["a", "x"], {"a"}, 2) == 0.5
        assert M.intra_list_diversity([[1, 0], [0, 1]]) == pytest.approx(1.0)


class TestRankers:
    def test_linear_contributions_sum(self) -> None:
        r = LinearRanker(dict(DEFAULT_WEIGHTS))
        f = {k: 0.5 for k in FEATURES}
        score, contrib = r.score(f)
        assert score == pytest.approx(sum(contrib.values()))
        assert r.score(f, {"small_creator": 0.3})[1]["small_creator"] == pytest.approx(0.15)

    def test_pairwise_ltr_learns_preference(self) -> None:
        rng = np.random.default_rng(1)
        pairs = []
        for _ in range(300):
            a = {k: float(rng.random()) for k in FEATURES}
            b = {k: float(rng.random()) for k in FEATURES}
            # hidden truth: users prefer quality, ignore relevance noise
            pairs.append((a, b) if a["quality"] > b["quality"] else (b, a))
        model = train_pairwise_ranker(pairs[:240])
        assert model.weights["quality"] == max(model.weights.values())
        assert pairwise_accuracy(model, pairs[240:]) > 0.9
        assert model.version.startswith("ltr_pairwise_v1:")


def test_mmr_caps_items_per_channel() -> None:
    from types import SimpleNamespace

    from avd.recsys.recommender import _mmr

    cands = [
        {"video": SimpleNamespace(channel_id="same", embedding=[1.0, 0.0]), "rank": 1.0 - i * 0.01}
        for i in range(5)
    ]
    cands.append({"video": SimpleNamespace(channel_id="other", embedding=[0.0, 1.0]), "rank": 0.5})
    picked = _mmr(cands, 5, diversity=0.3)
    assert sum(1 for c in picked if c["video"].channel_id == "same") == 2


def test_benchmark_harness_end_to_end() -> None:
    from avd.bench.harness import render_markdown, run_benchmark

    r = run_benchmark(REPO_ROOT / "benchmarks" / "datasets" / "synthetic_v1", FIXTURES, evaluator="heuristic")
    assert r["popularity_invariance"]["passed"] is True
    assert r["popularity_invariance"]["videos_checked"] == 20
    assert r["meta"]["warnings"] and "SYNTHETIC" in r["meta"]["warnings"][0]
    assert r["evaluation"]["inter_rater"]["krippendorff_alpha_interval"] is None  # single rater
    assert r["cost"]["total_usd"] == 0.0
    md = render_markdown(r)
    assert "PASSED" in md and "Reproducibility" in md
    # Regression guards for the deterministic baseline (update deliberately when the heuristic changes).
    assert r["evaluation"]["pairwise"]["_all"]["accuracy"] >= 0.7
    assert r["recommendation"]["mean"]["ndcg@10"] >= 0.6


@pytest.mark.redis
def test_celery_dispatch_executes_db_jobs(db: DB, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Celery transport claims jobs from the DB queue and executes them (eager mode)."""
    if not REDIS_URL:
        pytest.skip("AVD_TEST_REDIS_URL not set")
    from avd.services import seed_demo
    from avd.workers import celery_app
    from avd.workers.context import build_context

    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    monkeypatch.setattr(celery_app, "_ctx", build_context(settings=db.settings, session_factory=db.sf))
    celery_app.app.conf.task_always_eager = True
    try:
        assert celery_app.dispatch_task.delay(limit=50).get() == 20  # stage A for all fixtures
        assert celery_app.dispatch_task.delay(limit=50).get() == 20  # stage B, routed to the deep queue
    finally:
        celery_app.app.conf.task_always_eager = False
    from sqlalchemy import func, select

    from avd.db.models import Evaluation

    with db.sf() as s:
        assert s.scalar(select(func.count()).select_from(Evaluation)) == 20
