"""Phase 2 P3: transcript processing and hierarchical (long-form) evaluation."""

from __future__ import annotations

from typing import Any

import pytest

from avd.content.models import Transcript, TranscriptSegment
from avd.content.processing import (
    chunk_transcript,
    estimate_tokens,
    identify_language,
    normalize_transcript,
    validate_transcript,
)
from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS, LLMRubricEvaluator
from avd.evaluation.pipeline import EvaluationPipeline
from avd.evaluation.schema import Dimension, DimensionStatus, EvaluationMode
from avd.llm.base import LLMRequest
from tests.conftest import ScriptedProvider, make_input, make_transcript

EN = "The cell membrane controls what enters and leaves the cell, and it is made of a lipid bilayer that is not permeable to most ions."
ES = "La membrana celular controla lo que entra y sale de la célula, y está formada por una bicapa lipídica que no es permeable a la mayoría de los iones."
DE = "Die Zellmembran kontrolliert, was in die Zelle hinein und aus ihr heraus gelangt, und sie besteht aus einer Lipiddoppelschicht, die für die meisten Ionen nicht durchlässig ist."
FR = "La membrane cellulaire contrôle ce qui entre et sort de la cellule, et elle est constituée d'une bicouche lipidique qui n'est pas perméable à la plupart des ions."


@pytest.mark.parametrize(("text", "lang"), [(EN, "en"), (ES, "es"), (DE, "de"), (FR, "fr")])
def test_language_identification(text: str, lang: str) -> None:
    assert identify_language(text * 2).language == lang


def test_unknown_language_is_not_guessed() -> None:
    assert identify_language("ok").language == "unknown"
    assert (
        identify_language("Привет мир это тест на неизвестный язык который мы не поддерживаем " * 3).language
        == "unknown"
    )


def test_normalization_merges_rolling_duplicates_and_fixes_overlaps() -> None:
    t = Transcript(
        language="en",
        segments=[
            TranscriptSegment(start=0, end=4, text="Hello  world."),
            TranscriptSegment(start=3, end=6, text="Hello world."),
            TranscriptSegment(start=5, end=8, text="   "),
            TranscriptSegment(start=5.5, end=9, text="Next idea."),  # overlaps merged segment ending at 6,
        ],
    )
    out, rep = normalize_transcript(t)
    assert [s.text for s in out.segments] == ["Hello world.", "Next idea."]
    assert rep.merged_duplicates == 1 and rep.removed_empty == 1 and rep.fixed_overlaps == 1
    assert all(b.start >= a.end for a, b in zip(out.segments, out.segments[1:], strict=False))


def test_validation_reports_language_mismatch_and_gaps() -> None:
    t = Transcript(
        language="de",
        segments=[TranscriptSegment(start=i * 5, end=i * 5 + 4, text=EN) for i in range(10)]
        + [TranscriptSegment(start=200, end=204, text=EN)],
    )
    rep = validate_transcript(t)
    assert rep.language.language == "en" and rep.language_mismatch and rep.gaps_over_30s == 1
    assert "declared_language_mismatch" in rep.issues


TOPIC_A = (
    "Photosynthesis converts light energy into chemical energy in chloroplasts using chlorophyll pigments."
)
TOPIC_B = "Volcanic eruptions release magma, ash and sulfur dioxide gases from beneath tectonic plates."


def long_transcript(n: int = 400) -> Transcript:
    return make_transcript([TOPIC_A if i < n // 2 else TOPIC_B for i in range(n)])


def test_chunking_is_deterministic_complete_and_budgeted() -> None:
    t = long_transcript()
    a, b = chunk_transcript(t, 1200), chunk_transcript(t, 1200)
    assert a == b and len(a) > 3
    owned = [i for c in a for i in range(c.first_segment, c.last_segment + 1)]
    assert owned == list(range(len(t.segments)))  # every segment owned exactly once
    assert all(c.estimated_tokens <= 1200 * 1.3 + 60 for c in a)
    assert all(
        c.start_seconds == t.segments[c.first_segment].start
        and c.end_seconds == t.segments[c.last_segment].end
        for c in a
    )
    assert all(c.context_from == c.first_segment - 1 for c in a[1:])  # one context-only overlap segment


def test_chunk_boundary_prefers_topic_shift() -> None:
    t = long_transcript(400)
    shift = 200  # first TOPIC_B segment
    chunks = chunk_transcript(t, estimate_tokens(TOPIC_A) * 190)
    assert any(c.last_segment == shift - 1 and c.boundary == "topic_shift" for c in chunks)


def _hierarchical_provider(cite_context: bool = False) -> ScriptedProvider:
    def respond(req: LLMRequest, i: int) -> dict[str, Any]:
        if req.schema_name == "chunk_notes":
            import re

            first = int(re.search(r"segments (\d+)-(\d+)", req.user).group(1))  # type: ignore[union-attr]
            seg = first - 1 if cite_context and first > 0 else first
            quote = (TOPIC_A if seg < 200 else TOPIC_B)[:60]
            return {
                "summary": "Part summary.",
                "points": [
                    {
                        "segment_index": seg,
                        "quote": quote,
                        "kind": "explanation",
                        "note": "explains mechanism",
                    }
                ],
                "claims": [
                    {
                        "segment_index": seg,
                        "quote": quote,
                        "claim": "a claim",
                        "claim_type": "factual_assertion",
                    }
                ],
                "detected_issues": [],
            }
        return {
            "content_summary": "Two-topic lecture.",
            "detected_issues": [],
            "judgments": [
                {
                    "dimension": d.value,
                    "assessable": True,
                    "score": 3,
                    "confidence": 0.6,
                    "justification": "Based on notes.",
                    "evidence": [{"segment_index": 250, "quote": TOPIC_B[:50]}],
                    "counter_evidence": [],
                    "unassessable_reason": "",
                }
                for d in TEXT_DIMENSIONS
            ],
        }

    return ScriptedProvider(respond)


def test_long_transcript_is_evaluated_hierarchically_not_failed() -> None:
    """Regression for audit defect D1: a transcript over the single-pass budget used to crash."""
    t = make_transcript(
        ["We now derive the result step by step, because each assumption matters for the proof."] * 6000
    )
    assert len(t.text) > 400_000
    p = ScriptedProvider(
        lambda req, i: (
            {"summary": "derivation", "points": [], "claims": [], "detected_issues": []}
            if req.schema_name == "chunk_notes"
            else {
                "content_summary": "",
                "detected_issues": [],
                "judgments": [
                    {
                        "dimension": d.value,
                        "assessable": True,
                        "score": 3,
                        "confidence": 0.5,
                        "justification": "ok",
                        "evidence": [{"segment_index": 10, "quote": "each assumption matters for the proof"}],
                        "counter_evidence": [],
                        "unassessable_reason": "",
                    }
                    for d in TEXT_DIMENSIONS
                ],
            }
        )
    )
    ev = LLMRubricEvaluator(p, "m", single_pass_max_tokens=60_000, chunk_target_tokens=6_000)
    r = EvaluationPipeline([ev], "B", EvaluationMode.LLM).run(make_input(t))
    rep = r.coverage_report["evaluators"]["llm_rubric"]
    assert rep["strategy"] == "hierarchical" and rep["chunks"] >= 20 and rep["segments_evaluated"] == 6000
    assert r.dimension(Dimension.DEPTH).status == DimensionStatus.SCORED
    assert sum(1 for c in p.calls if c.schema_name == "chunk_notes") == rep["chunks"]
    assert max(len(c.user) for c in p.calls) < 60_000 * 4  # every request fits the configured budget


def test_hierarchical_notes_are_verified_and_context_citations_dropped() -> None:
    t = long_transcript(400)
    for cite_context, expect_all in ((False, True), (True, False)):
        ev = LLMRubricEvaluator(
            _hierarchical_provider(cite_context), "m", single_pass_max_tokens=2000, chunk_target_tokens=1500
        )
        r = EvaluationPipeline([ev], "B", EvaluationMode.LLM).run(make_input(t))
        rep = r.coverage_report["evaluators"]["llm_rubric"]
        verified_all = rep["map_citations_verified"] == rep["map_citations_claimed"]
        assert verified_all is expect_all
        reduce_call_user = ev.provider.calls[-1].user  # type: ignore[attr-defined]
        assert "VERBATIM OPENING" in reduce_call_user and "VERBATIM CLOSING" in reduce_call_user
        assert r.dimension(Dimension.DEPTH).evidence[0].start_seconds == t.segments[250].start


def test_too_many_failed_chunks_fail_the_evaluation_visibly() -> None:
    t = long_transcript(400)

    def respond(req: LLMRequest, i: int) -> dict[str, Any]:
        return {"bad": "output"}  # never valid

    ev = LLMRubricEvaluator(
        ScriptedProvider(respond), "m", single_pass_max_tokens=2000, chunk_target_tokens=1500
    )
    r = EvaluationPipeline([ev], "B", EvaluationMode.LLM).run(make_input(t))
    assert all(r.dimension(d).status == DimensionStatus.FAILED for d in TEXT_DIMENSIONS)
