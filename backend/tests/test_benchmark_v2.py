"""Phase 2 benchmark v2: statistics validated against published values, the real-data harness,
release-gate locking and the test-split guard.

Ratings constructed in this file are test inputs for the statistics code. They are not labels
for any real item and are never written to ``benchmarks/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avd.bench import stats as S
from avd.bench.real import (
    SplitGuardError,
    _human_metrics,
    evaluate_gates,
    load_gates,
    lock_gates,
    render_markdown,
    run_real_benchmark,
)
from avd.config import REPO_ROOT
from avd.datasets.pilot import describe_item, finalize_manifest
from avd.datasets.sources import CHANGELOG, TALKPYTHON
from tests.conftest import make_settings
from tests.test_datasets import CHANGELOG_SAMPLE, VTT_SAMPLE, _write_cache

nan = np.nan


class TestReferenceValues:
    def test_icc_matches_shrout_fleiss_1979(self) -> None:
        r = np.array(
            [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float
        )
        assert round(S.icc(r, "1,1"), 2) == 0.17
        assert round(S.icc(r, "2,1"), 2) == 0.29
        assert round(S.icc(r, "3,1"), 2) == 0.71

    def test_krippendorff_alpha_matches_krippendorff_2011(self) -> None:
        k = np.array(
            [
                [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
                [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
                [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
                [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
            ]
        )
        assert round(S.krippendorff_alpha(k, "nominal"), 3) == 0.743
        assert round(S.krippendorff_alpha(k, "ordinal"), 3) == 0.815
        assert round(S.krippendorff_alpha(k, "interval"), 3) == 0.849

    def test_weighted_kappa_bounds(self) -> None:
        assert S.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
        assert S.weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
        near = S.weighted_kappa([1, 2, 3, 4, 5, 3], [2, 2, 3, 4, 4, 3])
        far = S.weighted_kappa([1, 2, 3, 4, 5, 3], [5, 2, 3, 4, 1, 3])
        assert near > far  # quadratic weights penalise distant disagreements more

    def test_icc_handles_unbalanced_designs(self) -> None:
        r = np.array([[4, 5, nan], [2, 2, 3], [5, nan, nan], [1, 2, nan]], float)
        assert 0.8 < S.icc(r, "1,1") <= 1.0  # the single-rated row is ignored


class TestGroupedBootstrap:
    def test_resamples_whole_groups_and_is_deterministic(self) -> None:
        units = [{"g": g, "x": float(i)} for g in range(6) for i in range(3)]
        seen_groups: list[int] = []

        def stat(us: Any) -> float:
            seen_groups.append(len({u["g"] for u in us}))
            # every group in a resample appears with all 3 of its units
            counts: dict[int, int] = {}
            for u in us:
                counts[u["g"]] = counts.get(u["g"], 0) + 1
            assert all(c % 3 == 0 for c in counts.values())
            return float(np.mean([u["x"] for u in us]))

        a = S.grouped_bootstrap(units, lambda u: u["g"], stat, n_boot=200, seed=3)
        b = S.grouped_bootstrap(units, lambda u: u["g"], stat, n_boot=200, seed=3)
        assert a == b and a["n_groups"] == 6 and a["ci_low"] <= a["value"] <= a["ci_high"]

    def test_single_group_has_no_ci(self) -> None:
        r = S.grouped_bootstrap([1.0, 2.0, 3.0], lambda u: "one", lambda us: float(np.mean(us)))
        assert r["value"] == 2.0 and r["ci_low"] is None


def test_risk_coverage_rewards_informative_confidence() -> None:
    loss = [0, 0, 0, 1, 2, 3]
    good = S.risk_coverage([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], loss)
    bad = S.risk_coverage([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], loss)
    assert good["aurc"] < good["risk_at_full_coverage"] < bad["aurc"]
    assert good["aurc"] == good["oracle_aurc"]


def test_pairwise_tie_aware() -> None:
    scores = {"a": 4.0, "b": 2.0, "c": 4.1, "d": None}
    r = S.pairwise_tie_aware(
        scores,
        [("a", "b", "a"), ("b", "a", "a"), ("a", "c", "equal"), ("a", "d", "a"), ("a", "b", "insufficient")],
    )
    assert r["strict_accuracy"] == 0.5 and r["tie_aware_accuracy"] == pytest.approx(2 / 3)
    assert r["model_abstained"] == 1 and r["label_insufficient"] == 1 and r["label_ties"] == 1


def test_macro_f1_and_confusion() -> None:
    r = S.macro_f1(["x", "x", "y", "y"], ["x", "y", "y", "y"])
    assert r["confusion"]["x"]["y"] == 1 and r["accuracy"] == 0.75
    assert r["per_class"]["x"]["recall"] == 0.5 and r["per_class"]["y"]["precision"] == pytest.approx(0.6667)


# ------------------------------------------------------------------------------- human metrics


def _labels(rows: list[tuple[str, str, str, int | None]], split: str = "dev") -> list[dict[str, Any]]:
    return [
        {
            "video_id": v,
            "dimension": d,
            "status": "scored" if s is not None else "insufficient_evidence",
            "score": s,
            "rater_id": r,
            "label_source": "human_annotation",
            "split": split,
        }
        for v, d, r, s in rows
    ]


def test_human_metrics_compare_model_with_leave_one_out_humans() -> None:
    rng = np.random.default_rng(0)
    vids = [f"v{i}" for i in range(12)]
    truth = {v: int(rng.integers(1, 6)) for v in vids}
    rows = []
    for v in vids:
        for r in ("r1", "r2", "r3"):
            # independent noise of at most one point, so no cell is disputed (range >= 2)
            rows.append((v, "depth", r, int(np.clip(truth[v] + rng.integers(0, 2), 1, 5))))
        rows += [(v, "originality", r, None) for r in ("r1", "r2")]
    model = {(v, "depth"): {"status": "scored", "score": float(truth[v]), "confidence": 0.8} for v in vids}
    model |= {(v, "originality"): {"status": "scored", "score": 3.0, "confidence": 0.5} for v in vids}
    groups = {v: f"g{i % 4}" for i, v in enumerate(vids)}

    scores = {v: {"overall": None, "content": None, "presentation": None} for v in vids}
    out = _human_metrics(model, scores, _labels(rows), [], groups, seed=1, n_boot=200)
    assert out["available"]
    mv, hh = out["model_vs_human"], out["human_vs_human"]
    assert mv["per_dimension"]["depth"]["n"] == 12 and mv["per_dimension"]["depth"]["spearman"]["value"] > 0.7
    assert mv["per_dimension"]["depth"]["spearman"]["n_groups"] == 4
    assert hh["leave_one_rater_out"]["n"] == 36 and hh["raters"] == 3
    # Humans said originality was not assessable; the model scored it anyway.
    assert mv["abstention_agreement"]["model_scored_when_humans_did_not_rate"] == 1.0
    assert out["model_minus_human_mae_same_reference"]["value"] is not None


def test_non_human_label_sources_are_refused(tmp_path: Path) -> None:
    from avd.bench.real import _load_labels

    (tmp_path / "labels.jsonl").write_text(
        json.dumps(
            {
                "video_id": "v",
                "dimension": "depth",
                "score": 3,
                "split": "dev",
                "label_source": "synthetic_design_intent",
            }
        )
    )
    with pytest.raises(ValueError, match="non-human"):
        _load_labels(tmp_path, "dev")


# ----------------------------------------------------------------------------------------- gates


@pytest.fixture
def gates(tmp_path: Path) -> tuple[Path, Path]:
    g, lk = tmp_path / "gates.json", tmp_path / "gates.lock"
    shutil.copy(REPO_ROOT / "benchmarks" / "release_gates.json", g)
    lock_gates("test lock", g, lk)
    return g, lk


def test_committed_gates_are_locked() -> None:
    g = load_gates()
    assert g["lock_valid"], "benchmarks/release_gates.json changed after it was locked"


def test_relocking_changed_gates_requires_a_new_test_set(gates: tuple[Path, Path]) -> None:
    g, lk = gates
    data = json.loads(g.read_text())
    data["gates"][0]["threshold"] = 0.1
    g.write_text(json.dumps(data))
    assert not load_gates(g, lk)["lock_valid"]
    with pytest.raises(SplitGuardError):
        lock_gates("weaken", g, lk)
    lock = lock_gates("new test set", g, lk, new_test_set=True)
    assert lock["history"] and load_gates(g, lk)["lock_valid"]


def test_gates_without_labels_are_not_evaluable_and_never_release_ready(gates: tuple[Path, Path]) -> None:
    g = load_gates(*gates)
    report = {
        "human_grounded": {"available": False},
        "label_free": {"popularity_invariance": {"passed": True}, "injection": {"items_flagged_rate": 0.0}},
    }
    res = evaluate_gates(g, report, "test")
    by = {r["id"]: r for r in res["results"]}
    assert by["content_spearman_lower_ci"] == {
        **by["content_spearman_lower_ci"],
        "status": "blocked_missing_labels",
        "evaluable": False,
        "detail": "no human labels",
    }
    assert by["popularity_invariance"]["status"] == "pass"
    assert res["binding"] and not res["release_ready"]


def test_a_heuristic_run_never_passes_candidate_gates(gates: tuple[Path, Path]) -> None:
    g = load_gates(*gates)
    report = {
        "human_grounded": {"available": False},
        "label_free": {
            "popularity_invariance": {"passed": True},
            "injection": {"items_flagged_rate": 0.0},
            "cost": {"per_video_mean": 0.0},
            "latency_ms": {"p95": 80.0},
        },
    }
    res = evaluate_gates(g, report, "dev", evaluator="heuristic")
    by = {r["id"]: r for r in res["results"]}
    assert by["popularity_invariance"]["status"] == "blocked_no_inference"
    assert by["popularity_invariance"]["baseline_value"] is True
    assert by["injection_benign_flag_rate"]["status"] == "pass"  # evaluator-independent
    assert "pass" not in {r["status"] for r in res["results"] if r["id"] != "injection_benign_flag_rate"}
    exposed = evaluate_gates(
        g, report, "test", exposures={"injection_benign_flag_rate": ["injection_v2 (db9126f)"]}
    )
    assert {r["id"]: r for r in exposed["results"]}["injection_benign_flag_rate"][
        "status"
    ] == "not_applicable_exposed"
    assert not exposed["release_ready"]


# ------------------------------------------------------------------------------ real harness


@pytest.fixture
def real_manifest(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    items = []
    texts = [
        (CHANGELOG, f"{show}/{show}-{i}.md", CHANGELOG_SAMPLE.replace("caching", topic))
        for i, (show, topic) in enumerate(
            [
                ("podcast", "caching"),
                ("gotime", "tracing"),
                ("jsparty", "bundling"),
                ("practicalai", "embeddings"),
                ("shipit", "deploys"),
            ]
        )
    ] + [(TALKPYTHON, "youtube_transcripts/301-packaging.vtt", VTT_SAMPLE)]
    for source, path, text in texts:
        raw = _write_cache(cache, source, path, text)
        items.append(describe_item(source, path, raw)[0])
    m = finalize_manifest(items, {}, seed=5)
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(m))
    return out, cache


def test_real_benchmark_label_free_run(
    tmp_path: Path, real_manifest: tuple[Path, Path], gates: tuple[Path, Path]
) -> None:
    manifest, cache = real_manifest
    r = run_real_benchmark(
        manifest,
        None,
        split="dev",
        base_settings=make_settings(tmp_path),
        cache_dir=cache,
        offline=True,
        n_boot=50,
        gates_path=gates[0],
        lock_path=gates[1],
    )
    lf = r["label_free"]
    assert lf["items_evaluated"] == lf["items_in_split"] > 0
    assert set(lf["assessment_state"]) == {"heuristic_baseline"}  # real content, heuristic evaluator
    assert lf["popularity_invariance"]["passed"]
    assert r["human_grounded"] == {"available": False, "reason": "no human labels for this split"}
    assert not r["gates"]["binding"] and not r["gates"]["release_ready"]
    assert any("HEURISTIC" in w for w in r["meta"]["warnings"])
    for key in ("git_commit", "manifest_items_sha256", "pipeline_config_hash", "seed", "runtime_seconds"):
        assert r["meta"][key] is not None
    assert "Not available" in render_markdown(r)


def test_real_benchmark_with_human_labels(
    tmp_path: Path, real_manifest: tuple[Path, Path], gates: tuple[Path, Path]
) -> None:
    manifest, cache = real_manifest
    m = json.loads(manifest.read_text())
    dev = [it["item_id"] for it in m["items"] if it["split"] == "dev"]
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    rows = _labels(
        [
            (v, d, r, 3 + (i % 2))
            for i, v in enumerate(dev)
            for d in ("depth", "clarity")
            for r in ("r1", "r2")
        ]
    )
    for row in rows:
        row["group_key"] = next(it["group_key"] for it in m["items"] if it["item_id"] == row["video_id"])
    (labels_dir / "labels.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    r = run_real_benchmark(
        manifest,
        labels_dir,
        split="dev",
        base_settings=make_settings(tmp_path),
        cache_dir=cache,
        offline=True,
        n_boot=50,
        gates_path=gates[0],
        lock_path=gates[1],
    )
    hg = r["human_grounded"]
    assert hg["available"] and hg["cells_with_reference"] == 2 * len(dev)
    assert r["meta"]["labels_sha256"] and r["meta"]["human_label_rows"] == len(rows)
    render_markdown(r)


def test_test_split_is_guarded(
    tmp_path: Path, real_manifest: tuple[Path, Path], gates: tuple[Path, Path]
) -> None:
    manifest, cache = real_manifest
    kw: dict[str, Any] = {
        "base_settings": make_settings(tmp_path),
        "cache_dir": cache,
        "offline": True,
        "gates_path": gates[0],
        "lock_path": gates[1],
    }
    with pytest.raises(SplitGuardError, match="held out"):
        run_real_benchmark(manifest, None, split="test", **kw)
    labels_dir = tmp_path / "early"
    labels_dir.mkdir()
    early = _labels([("x", "depth", "r1", 3)], split="test")
    early[0]["submitted_at"] = "2020-01-01T00:00:00+00:00"
    (labels_dir / "labels.jsonl").write_text(json.dumps(early[0]))
    with pytest.raises(SplitGuardError, match="predate"):
        run_real_benchmark(manifest, labels_dir, split="test", confirm_test=True, **kw)


def test_video_urls_respect_platform_and_provenance() -> None:
    from avd.db.models import Video
    from avd.recsys.recommender import _video_url

    assert (
        _video_url(Video(id="abc123", platform="youtube", is_synthetic=False))
        == "https://www.youtube.com/watch?v=abc123"
    )
    licensed = Video(
        id="chl_x_1", platform="changelog.com", external_url="https://changelog.com/x/1", is_synthetic=False
    )
    assert _video_url(licensed) == "https://changelog.com/x/1"
    assert _video_url(Video(id="tpy_1", platform="talkpython.fm", is_synthetic=False)) is None
    assert _video_url(Video(id="syn_1", platform="youtube", is_synthetic=True)) is None


def test_recommendations_report_eligibility_intent_and_factual_explanations(tmp_path: Path) -> None:
    """Recommender v2: cold-start intent priors, duration range, eligibility accounting."""
    from sqlalchemy.orm import sessionmaker

    from avd.db.models import Base, User, UserPreferences
    from avd.db.session import make_engine
    from avd.embeddings.base import HashingEmbedder
    from avd.recsys.recommender import recommend
    from avd.services import seed_demo
    from avd.workers.context import build_context
    from avd.workers.runner import run_until_idle

    settings = make_settings(tmp_path)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as s:
        seed_demo(s, settings)
        s.commit()
    emb = HashingEmbedder()
    run_until_idle(build_context(settings=settings, session_factory=sf, embedder=emb))
    with sf() as s:
        u = User(token_hash="x")
        u.preferences = UserPreferences(
            interests=["photosynthesis and plant biology"],
            intent="learn",
            min_duration_minutes=3,
            max_duration_minutes=30,
            min_quality=2.5,
        )
        s.add(u)
        s.flush()
        rec = recommend(s, u.id, emb, limit=5, log_impressions=False)
    assert rec["eligibility"]["candidates"] >= rec["eligibility"]["eligible"] > 0
    assert any("intent 'learn'" in n for n in rec["notes"])
    for it in rec["items"]:
        d = it["video"]["duration_seconds"]
        assert d is None or 180 <= d <= 1800
        text = " ".join(it["explanations"])
        assert "Assessment:" in text and "Genre:" in text and "from your intent" in text
        q = it["intrinsic_quality"]["overall"]
        assert q is None or q >= 0  # quality is reported separately from relevance


def test_ltr_split_is_temporal_and_never_splits_a_request() -> None:
    from datetime import datetime, timedelta

    from avd.recsys.ltr_data import LabeledPair, temporal_split

    t0 = datetime(2026, 1, 1)
    pairs = [
        LabeledPair({"quality": 1.0}, {"quality": 0.0}, f"req{r}", f"user{r % 3}", t0 + timedelta(hours=r))
        for r in range(10)
        for _ in range(3)
    ]
    train, held = temporal_split(pairs, holdout=0.2)
    assert {p.request_id for p in train}.isdisjoint({p.request_id for p in held})
    assert max(p.shown_at for p in train) < min(p.shown_at for p in held)
    assert len(held) == 6 and len(train) == 24


def test_dry_run_call_plan_matches_pipeline_structure() -> None:
    """The throughput experiment records the real call plan without any API: one rubric call for
    a short transcript; map calls per chunk + reduce for a long one; 3 samples in Stage C."""
    from avd.bench.throughput import DryRunProvider, _input
    from avd.content.models import Transcript, TranscriptSegment
    from avd.evaluation.pipeline import build_pipeline
    from avd.llm.pricing import cost_usd
    from tests.conftest import LECTURE, make_transcript

    settings = make_settings(Path("/tmp"))
    llm = settings.model_copy(
        update={
            "llm_provider": "anthropic",
            "llm_single_pass_max_tokens": 2000,
            "llm_chunk_target_tokens": 600,
        }
    )
    dry = DryRunProvider()
    build_pipeline("B", llm, dry).run(_input("v", make_transcript(LECTURE)))
    assert [c.schema_name for c in dry.calls] == ["rubric_evaluation"]
    long_t = Transcript(
        segments=[
            TranscriptSegment(start=i, end=i + 1, text=f"Point {i}: " + "word " * 60) for i in range(80)
        ]
    )
    dry = DryRunProvider()
    build_pipeline("B", llm, dry).run(_input("v", long_t))
    kinds = [c.schema_name for c in dry.calls]
    assert kinds.count("chunk_notes") > 1 and kinds[-1] == "rubric_evaluation"
    assert not any("output contract" in c.user for c in dry.calls)  # no repair calls
    dry = DryRunProvider()
    build_pipeline("C", llm, dry).run(_input("v", make_transcript(LECTURE)))
    assert len(dry.calls) == llm.llm_self_consistency_samples_stage_c
    assert cost_usd("claude-haiku-5-5", 150_000, 0) > cost_usd("claude-haiku-5-5", 99_000, 0) * 1.5 * 5 - 1e-9
