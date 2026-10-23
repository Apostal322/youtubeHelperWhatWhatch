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


def test_a_dimension_left_out_by_some_samples_is_not_reported_as_unanimous() -> None:
    def respond(req: LLMRequest, i: int) -> Any:
        out = _judgments(4, "Chlorophyll absorbs red and blue light", 1)
        if req.sample_index % 1000 != 0:  # samples 2 and 3 leave "depth" out
            out["judgments"] = [j for j in out["judgments"] if j["dimension"] != "depth"]
        return out

    ev = LLMRubricEvaluator(ScriptedProvider(respond), "m", samples=3)
    r = EvaluationPipeline([ev], "C", EvaluationMode.LLM).run(make_input(make_transcript(LECTURE)))
    d = r.dimension(Dimension.DEPTH)
    assert d.confidence == pytest.approx(0.8 / 3, abs=1e-3)  # before: 0.8, as if three samples agreed
    assert r.needs_review and any("depth (1 of 3)" in x for x in r.review_reasons)
    assert r.dimension(Dimension.CLARITY).confidence == pytest.approx(
        0.8, abs=1e-3
    )  # complete ones unchanged


def _points(req: LLMRequest, fabricate: bool) -> Any:
    first = int(re.search(r"segments (\d+)-(\d+)", req.user).group(1))  # type: ignore[union-attr]
    quote = TOPIC_A[:50] if first < 200 else TOPIC_B[:50]
    if fabricate:
        quote = "invented sentence about an unrelated miracle cure"
    pts = [{"segment_index": first, "quote": quote, "kind": "explanation", "note": "n"} for _ in range(3)]
    return {"summary": f"summary of part from {first}", "points": pts, "claims": [], "detected_issues": []}


class TestPartGrounding:
    def test_one_part_with_invented_quotes_is_flagged_and_its_summary_withheld(self) -> None:
        r, provider = _long(
            lambda req: _points(req, fabricate=" 0-" in req.user or "segments 0-" in req.user)
        )
        rep = r.coverage_report["evaluators"]["llm_rubric"]
        assert not rep["map_grounding_failed"]  # one part out of many: the video-wide ratio is fine
        assert len(rep["map_parts_ungrounded"]) == 1  # before: not detected
        assert r.needs_review and any("summaries were withheld" in x for x in r.review_reasons)
        reduce_prompt = provider.calls[-1].user
        assert "summary of part from 0" not in reduce_prompt and "summary withheld" in reduce_prompt

    def test_notes_with_summaries_but_no_quotes_are_flagged(self) -> None:
        r, provider = _long(
            lambda req: {
                "summary": "the speaker endorses a miracle cure",
                "points": [],
                "claims": [],
                "detected_issues": [],
            }
        )
        assert r.coverage_report["evaluators"]["llm_rubric"]["map_unquoted"]
        assert r.needs_review and any(
            "unverified summaries" in x for x in r.review_reasons
        )  # before: no review
        assert "You see verified notes" not in provider.calls[-1].user  # the preface no longer claims that
