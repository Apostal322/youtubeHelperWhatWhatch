"""Model output that must not look more complete or more certain than it is (audit 2026-10-23).

All model responses are scripted test doubles; nothing here is real inference.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS, LLMErrorAggregate, LLMRubricEvaluator
from avd.evaluation.pipeline import EvaluationPipeline
from avd.evaluation.schema import Dimension, EvaluationMode
from avd.llm.base import LLMOutputError, LLMRequest, LLMTransientError
from tests.conftest import LECTURE, ScriptedProvider, make_input, make_transcript

TOPIC_A = (
    "Photosynthesis converts light energy into chemical energy in chloroplasts using chlorophyll pigments."
)
TOPIC_B = "Volcanic eruptions release magma, ash and sulfur dioxide gases from beneath tectonic plates."


def _judgments(score: int, quote: str, segment: int) -> dict[str, Any]:
    return {
        "content_summary": "",
        "detected_issues": [],
        "judgments": [
            {
                "dimension": d.value,
                "assessable": True,
                "score": score,
                "confidence": 0.8,
                "justification": "ok",
                "evidence": [{"segment_index": segment, "quote": quote}],
                "counter_evidence": [],
                "unassessable_reason": "",
            }
            for d in TEXT_DIMENSIONS
        ],
    }


def _sampled(fail: Any) -> Any:
    def respond(req: LLMRequest, i: int) -> Any:
        return (
            _judgments(5, "Chlorophyll absorbs red and blue light", 1)
            if req.sample_index % 1000 == 0
            else fail
        )

    ev = LLMRubricEvaluator(ScriptedProvider(respond), "m", samples=3)
    return EvaluationPipeline([ev], "C", EvaluationMode.LLM).run(make_input(make_transcript(LECTURE)))


class TestSelfConsistency:
    def test_a_transient_sample_failure_retries_the_job(self) -> None:
        # Before: the failed samples were dropped and one sample was reported with full confidence.
        with pytest.raises(LLMErrorAggregate):
            _sampled(LLMTransientError("503 overloaded"))

    def test_a_permanently_failed_sample_lowers_confidence_and_asks_for_review(self) -> None:
        r = _sampled(LLMOutputError("invalid JSON"))
        d = r.dimension(Dimension.DEPTH)
        assert d.uncertainty["samples_succeeded"] == 1 and d.uncertainty["samples_requested"] == 3
        assert d.confidence == pytest.approx(0.8 / 3, abs=1e-3)  # was 0.8: as if all samples agreed
        assert r.needs_review and any("sample" in x for x in r.review_reasons)


def _long(respond_notes: Any) -> Any:
    t = make_transcript([TOPIC_A if i < 200 else TOPIC_B for i in range(400)])

    def respond(req: LLMRequest, i: int) -> Any:
        if req.schema_name == "chunk_notes":
            return respond_notes(req)
        return _judgments(3, TOPIC_B[:50], 250)

    provider = ScriptedProvider(respond)
    ev = LLMRubricEvaluator(provider, "m", single_pass_max_tokens=2000, chunk_target_tokens=1500)
    return EvaluationPipeline([ev], "B", EvaluationMode.LLM).run(make_input(t)), provider


class TestLongTranscripts:
    def test_a_failed_part_is_not_counted_as_evaluated_and_is_flagged(self) -> None:
        def notes(req: LLMRequest) -> Any:
            first = int(re.search(r"segments (\d+)-(\d+)", req.user).group(1))  # type: ignore[union-attr]
            return (
                {"bad": "x"}
                if first == 0
                else {"summary": "s", "points": [], "claims": [], "detected_issues": []}
            )

        r, _ = _long(notes)
        rep = r.coverage_report["evaluators"]["llm_rubric"]
        assert rep["chunks_failed"] == 1 and rep["failed_parts"]
        assert rep["segments_evaluated"] < 400  # was 400 of 400
        assert r.needs_review and any("partial" in x for x in r.review_reasons)

    def test_part_notes_with_invented_quotes_are_flagged_and_summaries_labelled(self) -> None:
        def notes(req: LLMRequest) -> Any:
            pts = [
                {
                    "segment_index": 5,
                    "quote": f"invented insight {k} on quantum gravity",
                    "kind": "explanation",
                    "note": "x",
                }
                for k in range(10)
            ]
            return {
                "summary": "A rigorous derivation of quantum gravity.",
                "points": pts,
                "claims": [],
                "detected_issues": [],
            }

        r, provider = _long(notes)
        rep = r.coverage_report["evaluators"]["llm_rubric"]
        assert rep["map_citations_verified"] == 0 and rep["map_grounding_failed"]
        assert r.needs_review and any("poorly grounded" in x for x in r.review_reasons)  # was not reviewed
        reduce_prompt = provider.calls[-1].user
        assert "summaries are unverified model wording" in reduce_prompt
