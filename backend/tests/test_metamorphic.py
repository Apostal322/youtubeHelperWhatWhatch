"""Phase 3.3: metamorphic tests of the intrinsic quality evaluation.

Each test changes one thing that must not matter, or must matter only in a defined place, and
checks the evaluation:

* popularity stored in the database (views, likes, comments, subscribers);
* production quality (audio/visual scores) versus content and presentation;
* unavailable media: never a zero, never a penalty on content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from avd.evaluation.evaluators.base import EvaluatorOutput
from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.pipeline import build_pipeline
from avd.evaluation.schema import (
    Dimension,
    DimensionGroup,
    DimensionScore,
    DimensionStatus,
    Modality,
)
from tests.conftest import DB, _make_db, make_input, make_settings


def _technical() -> list[Dimension]:
    return [d for d in Dimension if d.group == DimensionGroup.TECHNICAL]


@pytest.fixture
def seeded(tmp_path: Path) -> DB:
    from avd.services import seed_demo

    db = _make_db(tmp_path, None)
    with db.sf() as s:
        seed_demo(s, db.settings)
        s.commit()
    return db


def _input_for(db: DB, vid: str) -> EvaluationInput:
    from avd.bench.evaluate_real import _input_for

    with db.sf() as s:
        return _input_for(s, vid)  # type: ignore[no-any-return]


class TestPopularityInDatabase:
    def test_stats_and_subscribers_change_nothing(self, seeded: DB) -> None:
        from avd.db.models import Channel, Video, VideoStats

        with seeded.sf() as s:
            vid = s.scalars(select(Video.id).order_by(Video.id)).first()
            assert vid
        before_inp = _input_for(seeded, vid)
        pipe = build_pipeline("B", seeded.settings, None)
        before = pipe.run(before_inp)
        with seeded.sf() as s:
            v = s.get(Video, vid)
            assert v is not None
            s.add(VideoStats(video_id=vid, view_count=10**9, like_count=10**8, comment_count=10**7))
            ch = s.get(Channel, v.channel_id)
            if ch is not None:
                ch.subscriber_count = 10**8
                ch.size_bucket = "huge"
            s.commit()
        after_inp = _input_for(seeded, vid)
        after = pipe.run(after_inp)
        assert after_inp.fingerprint() == before_inp.fingerprint()
        assert after_inp.model_dump() == before_inp.model_dump()
        assert after.aggregation.overall == before.aggregation.overall
        assert [(d.dimension, d.status, d.score) for d in after.dimensions] == [
            (d.dimension, d.status, d.score) for d in before.dimensions
        ]


def _with_production(score: float | None) -> Any:
    """Evaluator stand-in that scores every technical dimension `score` (None = abstain)."""

    class Production:
        name = "production_stub"
        version = "test"
        synthetic = True
        requires = frozenset({Modality.TRANSCRIPT})
        dimensions = tuple(_technical())

        def evaluate(self, inp: EvaluationInput, profile: Any) -> EvaluatorOutput:
            return EvaluatorOutput(
                dimensions=[
                    DimensionScore(
                        dimension=d,
                        status=DimensionStatus.SCORED,
                        score=score,
                        confidence=0.9,
                        justification="stub",
                        method="production_stub",
                    )
                    if score is not None
                    else DimensionScore.abstain(
                        d, DimensionStatus.REQUIRES_MEDIA, "no media", "production_stub"
                    )
                    for d in self.dimensions
                ]
            )

    return Production()


class TestProductionIndependence:
    def _run(self, tmp_path: Path, score: float | None) -> Any:
        from avd.evaluation.pipeline import EvaluationPipeline

        base = build_pipeline("B", make_settings(tmp_path), None)
        pipe = EvaluationPipeline([*base.evaluators, _with_production(score)], "B", base.mode)
        return pipe.run(make_input())

    def test_poor_production_never_lowers_content_or_presentation(self, tmp_path: Path) -> None:
        good, poor = self._run(tmp_path, 5.0), self._run(tmp_path, 1.0)
        for group in (DimensionGroup.CONTENT, DimensionGroup.PRESENTATION):
            assert good.aggregation.groups[group].score == poor.aggregation.groups[group].score
        content = [d for d in good.dimensions if d.dimension not in _technical()]
        assert [(d.dimension, d.score) for d in content] == [
            (d.dimension, d.score) for d in poor.dimensions if d.dimension not in _technical()
        ]
        # Production quality may move the overall only through the technical group.
        assert poor.aggregation.groups[DimensionGroup.TECHNICAL].score == 1.0
        assert good.aggregation.overall is not None and poor.aggregation.overall is not None
        assert good.aggregation.overall >= poor.aggregation.overall

    def test_unavailable_media_is_not_a_zero(self, tmp_path: Path) -> None:
        missing = self._run(tmp_path, None)
        poor = self._run(tmp_path, 1.0)
        tech = [d for d in missing.dimensions if d.dimension in _technical()]
        assert tech and all(d.status != DimensionStatus.SCORED and d.score is None for d in tech)
        assert missing.aggregation.groups[DimensionGroup.TECHNICAL].score is None
        # Missing media is excluded from the overall, so it cannot pull it down like a 1 (or a 0) would.
        assert missing.aggregation.overall is not None and poor.aggregation.overall is not None
        assert missing.aggregation.overall > poor.aggregation.overall
        assert (
            missing.aggregation.groups[DimensionGroup.CONTENT].score
            == poor.aggregation.groups[DimensionGroup.CONTENT].score
        )


def test_popularity_text_does_not_shape_retrieval_embeddings() -> None:
    """Retrieval documents get the same popularity stripping as evaluator inputs."""
    from types import SimpleNamespace

    from avd.embeddings.base import HashingEmbedder
    from avd.embeddings.documents import document_inputs

    base = {
        "title": "Hash tables from scratch",
        "tags": ["hashing"],
        "topics": [],
        "description": "Load factor.",
    }
    plain = SimpleNamespace(**base)
    hyped = SimpleNamespace(
        **{
            **base,
            "title": "Hash tables from scratch (1M views!)",
            "description": "Load factor. Over 2,000,000 views and trending #1 this week!",
            "tags": ["hashing"],
        }
    )
    emb = HashingEmbedder()
    assert document_inputs(emb, plain, None) == document_inputs(emb, hyped, None)  # type: ignore[arg-type]


def test_channel_cap_is_soft_when_only_one_creator_covers_the_topic() -> None:
    """Phase 3.3 finding: a hard cap of 2 per channel left slots empty when every matching video
    came from one show. The cap still spreads creators when there is a choice."""
    from types import SimpleNamespace

    from avd.recsys.recommender import MAX_PER_CHANNEL, _mmr

    def cand(i: int, channel: str, rank: float) -> dict[str, Any]:
        return {"video": SimpleNamespace(id=f"v{i}", channel_id=channel, embedding=None), "rank": rank}

    one_channel = [cand(i, "show", 1.0 - i / 10) for i in range(5)]
    picked = _mmr(one_channel, 5, 0.0)
    assert len(picked) == 5 and sum(bool(c.get("channel_cap_relaxed")) for c in picked) == 5 - MAX_PER_CHANNEL
    mixed = [cand(i, "show", 1.0 - i / 10) for i in range(4)] + [cand(9, "other", 0.1)]
    picked = _mmr(mixed, 3, 0.0)
    assert [c["video"].channel_id for c in picked].count(
        "show"
    ) == MAX_PER_CHANNEL  # cap holds when there is a choice
    assert not any(c.get("channel_cap_relaxed") for c in picked)
