from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from avd.config import get_settings
from avd.content.synthetic_audio import generate, write_wav
from avd.evaluation.aggregation import aggregate
from avd.evaluation.evaluators.heuristic import HeuristicTextEvaluator
from avd.evaluation.pipeline import EvaluationPipeline, build_pipeline
from avd.evaluation.profiles import PROFILES, get_profile
from avd.evaluation.rubric import RUBRIC
from avd.evaluation.schema import (
    Dimension,
    DimensionGroup,
    DimensionScore,
    DimensionStatus,
    EvaluationMode,
    Genre,
)
from tests.conftest import make_input, make_transcript


def scored(d: Dimension, s: float, c: float = 0.8) -> DimensionScore:
    return DimensionScore(
        dimension=d, status=DimensionStatus.SCORED, score=s, confidence=c, justification="j", method="test"
    )


class TestSchemaContract:
    def test_scored_requires_score_confidence_and_justification(self) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=Dimension.DEPTH, status=DimensionStatus.SCORED, method="m", justification="x"
            )
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=Dimension.DEPTH, status=DimensionStatus.SCORED, score=3, confidence=0.5, method="m"
            )

    def test_abstention_cannot_carry_a_score(self) -> None:
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=Dimension.DEPTH, status=DimensionStatus.INSUFFICIENT_EVIDENCE, score=3, method="m"
            )

    def test_score_range_enforced(self) -> None:
        with pytest.raises(ValidationError):
            scored(Dimension.DEPTH, 6.0)

    def test_every_dimension_has_rubric_and_every_genre_a_profile(self) -> None:
        assert set(RUBRIC) == set(Dimension)
        assert set(PROFILES) == set(Genre)


class TestAggregation:
    def test_contributions_sum_to_overall(self) -> None:
        dims = [
            scored(Dimension.DEPTH, 5),
            scored(Dimension.CLARITY, 3),
            scored(Dimension.INFORMATION_DENSITY, 4),
            scored(Dimension.FACTUAL_ACCURACY, 4),
            scored(Dimension.AUDIO_QUALITY, 2),
        ]
        agg = aggregate(dims, get_profile(Genre.SCIENCE))
        assert agg.overall is not None
        assert sum(c.contribution for c in agg.contributions) == pytest.approx(agg.overall, abs=1e-3)
        assert sum(c.effective_weight for c in agg.contributions) == pytest.approx(1.0, abs=1e-4)

    def test_missing_dimensions_are_renormalised_not_penalised(self) -> None:
        full = aggregate(
            [scored(Dimension.DEPTH, 4), scored(Dimension.FACTUAL_ACCURACY, 4), scored(Dimension.CLARITY, 4)],
            get_profile(Genre.SCIENCE),
        )
        assert full.overall == pytest.approx(4.0)
        assert full.coverage < 1.0  # but the information loss is reported

    def test_abstains_when_content_coverage_too_low(self) -> None:
        agg = aggregate(
            [scored(Dimension.CLARITY, 5), scored(Dimension.PACING, 5)], get_profile(Genre.SCIENCE)
        )
        assert agg.overall is None and agg.abstain_reason
        assert agg.groups[DimensionGroup.PRESENTATION].score == 5

    def test_not_applicable_dimensions_are_ignored(self) -> None:
        prof = get_profile(Genre.ENTERTAINMENT)
        assert not prof.applicable(Dimension.FACTUAL_ACCURACY)
        a = aggregate([scored(Dimension.ORIGINALITY, 4), scored(Dimension.FACTUAL_ACCURACY, 1)], prof)
        assert Dimension.FACTUAL_ACCURACY not in a.groups[DimensionGroup.CONTENT].scored_dimensions

    def test_divergent_groups_are_flagged(self) -> None:
        dims = [
            scored(Dimension.DEPTH, 5),
            scored(Dimension.INFORMATION_DENSITY, 5),
            scored(Dimension.FACTUAL_ACCURACY, 5),
            scored(Dimension.AUDIO_QUALITY, 1),
        ]
        agg = aggregate(dims, get_profile(Genre.SCIENCE))
        assert any("diverge" in n for n in agg.notes)


class TestPopularityBlindness:
    def test_evaluation_input_has_no_popularity_or_channel_fields(self) -> None:
        from avd.evaluation.inputs import FORBIDDEN_INPUT_TERMS, EvaluationInput

        for field in EvaluationInput.model_fields:
            assert not any(term in field for term in FORBIDDEN_INPUT_TERMS), field

    def test_prompt_contains_no_popularity_information(self) -> None:
        from avd.evaluation.evaluators.llm_rubric import SYSTEM_PROMPT, TEXT_DIMENSIONS, build_user_prompt

        prompt = build_user_prompt(make_input(), get_profile(Genre.SCIENCE), TEXT_DIMENSIONS)
        lowered = (SYSTEM_PROMPT + prompt).lower()
        for term in ("subscriber", "view count", "likes:", "views:"):
            assert term not in lowered


class TestMediaIndependence:
    """A poorly recorded but excellent lecture keeps its content and presentation scores."""

    def test_bad_audio_changes_only_technical_group(self, tmp_path: Path) -> None:
        clean = write_wav(tmp_path / "clean.wav", generate("clean"))
        noisy = write_wav(tmp_path / "noisy.wav", generate("noisy_clipped"))
        pipe = build_pipeline("B", get_settings(), None)
        r_clean = pipe.run(make_input(audio_path=str(clean)))
        r_noisy = pipe.run(make_input(audio_path=str(noisy)))
        assert (
            r_clean.dimension(Dimension.AUDIO_QUALITY).score
            > r_noisy.dimension(Dimension.AUDIO_QUALITY).score
        )  # type: ignore[operator]
        for g in (DimensionGroup.CONTENT, DimensionGroup.PRESENTATION):
            assert r_clean.aggregation.groups[g].score == r_noisy.aggregation.groups[g].score
        for d in Dimension:
            if d.group != DimensionGroup.TECHNICAL:
                assert r_clean.dimension(d).score == r_noisy.dimension(d).score

    def test_missing_media_is_reported_as_requires_media(self) -> None:
        r = build_pipeline("B", get_settings(), None).run(make_input())
        for d in (Dimension.AUDIO_QUALITY, Dimension.VISUAL_QUALITY, Dimension.EDITING):
            assert r.dimension(d).status == DimensionStatus.REQUIRES_MEDIA
        assert "audio" in r.missing_inputs and "video" in r.missing_inputs


class TestHeuristicBaseline:
    def test_deterministic_and_labelled_synthetic(self) -> None:
        pipe = build_pipeline("B", get_settings(), None)
        a, b = pipe.run(make_input()), pipe.run(make_input())
        assert a.model_dump(exclude={"usage"}) == b.model_dump(exclude={"usage"})
        assert a.mode == EvaluationMode.SYNTHETIC_DEMO and a.is_synthetic
        for d in a.dimensions:
            if d.status == DimensionStatus.SCORED and d.method.startswith("heuristic"):
                assert d.justification.startswith("[Synthetic heuristic baseline")

    def test_abstains_on_what_text_statistics_cannot_judge(self) -> None:
        r = build_pipeline("B", get_settings(), None).run(make_input())
        assert r.dimension(Dimension.ORIGINALITY).status == DimensionStatus.INSUFFICIENT_EVIDENCE
        assert r.dimension(Dimension.FACTUAL_ACCURACY).status == DimensionStatus.INSUFFICIENT_EVIDENCE

    def test_evidence_timestamps_come_from_transcript(self) -> None:
        inp = make_input()
        r = build_pipeline("B", get_settings(), None).run(inp)
        assert inp.transcript is not None
        for d in r.dimensions:
            for e in d.evidence:
                if e.segment_index is not None:
                    seg = inp.transcript.segments[e.segment_index]
                    assert (e.start_seconds, e.end_seconds) == (seg.start, seg.end)
                    assert e.verified

    def test_filler_heavy_transcript_scores_lower_density(self) -> None:
        filler = make_transcript(
            ["Um so like, you know, guys, this is basically insane, smash that like button and subscribe."]
            * 30
        )
        ev = HeuristicTextEvaluator()
        good = {
            d.dimension: d.score for d in ev.evaluate(make_input(), get_profile(Genre.SCIENCE)).dimensions
        }
        bad = {
            d.dimension: d.score
            for d in ev.evaluate(make_input(filler), get_profile(Genre.SCIENCE)).dimensions
        }
        assert good[Dimension.INFORMATION_DENSITY] > bad[Dimension.INFORMATION_DENSITY]  # type: ignore[operator]

    def test_no_transcript_abstains_everything(self) -> None:
        inp = make_input().model_copy(update={"transcript": None, "modalities": frozenset()})
        r = build_pipeline("B", get_settings(), None).run(inp)
        assert r.status == "abstained" and r.aggregation.overall is None
        assert all(d.status != DimensionStatus.SCORED for d in r.dimensions)


class TestPipelineIsolation:
    def test_crashing_evaluator_yields_failed_dimensions_not_scores(self) -> None:
        class Boom:
            name, version, synthetic = "boom", "boom_v1", False
            dimensions = (Dimension.DEPTH,)
            requires = frozenset()

            def evaluate(self, inp, profile):  # type: ignore[no-untyped-def]
                raise RuntimeError("kaboom")

        r = EvaluationPipeline([Boom()], "B", EvaluationMode.LLM).run(make_input())
        d = r.dimension(Dimension.DEPTH)
        assert d.status == DimensionStatus.FAILED and d.score is None
        assert any("crashed" in w for w in r.validation.warnings)
