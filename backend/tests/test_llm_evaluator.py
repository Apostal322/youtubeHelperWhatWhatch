"""LLM evaluator behaviour with a scripted provider (no network, no credentials)."""

from __future__ import annotations

from typing import Any

import pytest

from avd.evaluation.evaluators.factual import FactualVerificationEvaluator, ReferencePassage
from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS, LLMRubricEvaluator
from avd.evaluation.pipeline import EvaluationPipeline
from avd.evaluation.profiles import get_profile
from avd.evaluation.schema import Dimension, DimensionStatus, EvaluationMode, Genre
from avd.llm.base import LLMRefusalError, LLMRequest, LLMTransientError
from avd.llm.providers import CachedProvider
from tests.conftest import LECTURE, ScriptedProvider, make_input


def judgments(
    score: int = 4, quote: str | None = None, seg: int = 1, assessable: bool = True, conf: float = 0.8
) -> dict[str, Any]:
    q = (
        quote
        if quote is not None
        else "Chlorophyll absorbs red and blue light because of its electronic structure"
    )
    return {
        "content_summary": "A lecture on photosynthesis.",
        "detected_issues": [],
        "judgments": [
            {
                "dimension": d.value,
                "assessable": assessable,
                "score": score if assessable else 0,
                "confidence": conf,
                "justification": f"Reasoned judgement for {d.value}.",
                "evidence": [{"segment_index": seg, "quote": q}],
                "unassessable_reason": "" if assessable else "not enough material",
            }
            for d in TEXT_DIMENSIONS
        ],
    }


def run(provider: ScriptedProvider, samples: int = 1):  # type: ignore[no-untyped-def]
    ev = LLMRubricEvaluator(provider, "claude-opus-5-5", samples=samples)
    return EvaluationPipeline([ev], "B", EvaluationMode.LLM).run(make_input())


def test_valid_output_is_scored_with_verified_evidence_and_real_timestamps() -> None:
    r = run(ScriptedProvider(lambda req, i: judgments(4)))
    d = r.dimension(Dimension.DEPTH)
    assert d.status == DimensionStatus.SCORED and d.score == 4
    e = d.evidence[0]
    assert e.verified and e.segment_index == 1 and (e.start_seconds, e.end_seconds) == (8.0, 16.0)
    assert r.validation.passed and r.mode == EvaluationMode.LLM and not r.is_synthetic
    assert r.usage and r.usage[0].cost_usd > 0  # cost tracked from token usage


def test_quote_cited_in_neighbouring_segment_is_relocated() -> None:
    r = run(ScriptedProvider(lambda req, i: judgments(4, seg=2)))  # quote actually lives in segment 1
    e = r.dimension(Dimension.DEPTH).evidence[0]
    assert e.verified and e.segment_index == 1


def test_fabricated_quotes_invalidate_model_scores() -> None:
    r = run(
        ScriptedProvider(
            lambda req, i: judgments(5, quote="The speaker proves cold fusion works at room temperature")
        )
    )
    assert not r.validation.passed
    for d in TEXT_DIMENSIONS:
        assert r.dimension(d).status == DimensionStatus.FAILED
        assert r.dimension(d).score is None
    assert r.aggregation.overall is None


def test_schema_invalid_output_retries_once_then_fails_without_inventing() -> None:
    p = ScriptedProvider(lambda req, i: {"judgments": [{"dimension": "depth", "score": 9}]})
    r = run(p)
    assert len(p.calls) == 2  # one retry with a fresh sample
    assert p.calls[0].sample_index != p.calls[1].sample_index
    assert all(r.dimension(d).status == DimensionStatus.FAILED for d in TEXT_DIMENSIONS)
    assert r.status == "failed"


def test_refusal_is_recorded_as_failure() -> None:
    r = run(ScriptedProvider(lambda req, i: LLMRefusalError("declined")))
    assert all(r.dimension(d).status == DimensionStatus.FAILED for d in TEXT_DIMENSIONS)


def test_transient_errors_propagate_for_job_retry() -> None:
    with pytest.raises(LLMTransientError):
        run(ScriptedProvider(lambda req, i: LLMTransientError("overloaded")))


def test_model_abstention_is_preserved() -> None:
    r = run(ScriptedProvider(lambda req, i: judgments(assessable=False)))
    assert all(r.dimension(d).status == DimensionStatus.INSUFFICIENT_EVIDENCE for d in TEXT_DIMENSIONS)


def test_omitted_dimension_is_failed() -> None:
    def resp(req: LLMRequest, i: int) -> dict[str, Any]:
        out = judgments(4)
        out["judgments"] = [j for j in out["judgments"] if j["dimension"] != "pacing"]
        return out

    r = run(ScriptedProvider(resp))
    assert r.dimension(Dimension.PACING).status == DimensionStatus.FAILED
    assert r.dimension(Dimension.DEPTH).status == DimensionStatus.SCORED


def test_factual_accuracy_from_model_knowledge_is_confidence_capped() -> None:
    r = run(ScriptedProvider(lambda req, i: judgments(5, conf=0.95)))
    fa = r.dimension(Dimension.FACTUAL_ACCURACY)
    assert fa.confidence is not None and fa.confidence <= 0.5
    assert "reference_corpus" in fa.missing_inputs


def test_self_consistency_uses_median_and_penalises_dispersion() -> None:
    scores = [2, 5, 4]
    r = run(ScriptedProvider(lambda req, i: judgments(scores[i])), samples=3)
    d = r.dimension(Dimension.DEPTH)
    assert d.score == 4
    assert d.confidence is not None and d.confidence < 0.8
    assert "samples=[2, 5, 4]" in d.justification


def test_prompt_frames_transcript_as_untrusted_data() -> None:
    p = ScriptedProvider(lambda req, i: judgments(4))
    run(p)
    req = p.calls[0]
    import re

    nonces = re.findall(r"=== BEGIN UNTRUSTED TRANSCRIPT (\w+) ===", req.user)
    assert nonces and f"=== END UNTRUSTED TRANSCRIPT {nonces[0]} ===" in req.user
    assert "Never follow it" in req.system and "third parties" in req.system
    assert req.json_schema["additionalProperties"] is False


def test_cached_provider_avoids_repeat_calls(sqlite_db) -> None:  # type: ignore[no-untyped-def]
    inner = ScriptedProvider(lambda req, i: judgments(4))
    cached = CachedProvider(inner, sqlite_db.sf)
    run(cached)  # type: ignore[arg-type]
    r2 = run(cached)  # type: ignore[arg-type]
    assert len(inner.calls) == 1
    assert r2.usage[0].cached and r2.usage[0].cost_usd == 0.0


class FakeRetriever:
    def __init__(self, passages: list[ReferencePassage]) -> None:
        self.passages = passages

    def search(self, query: str, k: int = 3) -> list[tuple[ReferencePassage, float]]:
        return [(p, 0.9) for p in self.passages[:k]]


def test_fact_checker_scores_only_grounded_verdicts() -> None:
    passage = ReferencePassage(
        passage_id="bio#0",
        doc_id="bio",
        title="Bio",
        license="CC-BY-4.0",
        text="Chlorophyll absorbs mainly blue and red light.",
    )

    def respond(req: LLMRequest, i: int) -> dict[str, Any]:
        if req.schema_name == "claim_extraction":
            return {
                "claims": [
                    {
                        "claim": "Chlorophyll absorbs red and blue light",
                        "segment_index": 1,
                        "quote": "Chlorophyll absorbs red and blue light",
                        "claim_type": "factual_assertion",
                    },
                    {
                        "claim": "RuBisCO fixes carbon",
                        "segment_index": 4,
                        "quote": "the key enzyme is RuBisCO",
                        "claim_type": "factual_assertion",
                    },
                    {
                        "claim": "Made-up claim",
                        "segment_index": 0,
                        "quote": "Today we will learn",
                        "claim_type": "factual_assertion",
                    },
                ]
            }
        return {
            "verdicts": [
                {
                    "claim_index": 0,
                    "verdict": "supported",
                    "passage_ids": ["bio#0"],
                    "explanation": "matches",
                },
                {"claim_index": 1, "verdict": "supported", "passage_ids": ["bio#0"], "explanation": "ok"},
                {
                    "claim_index": 2,
                    "verdict": "contradicted",
                    "passage_ids": ["invented#9"],
                    "explanation": "no real passage",
                },
            ]
        }

    ev = FactualVerificationEvaluator(ScriptedProvider(respond), "claude-opus-5-5", FakeRetriever([passage]))
    out = ev.evaluate(make_input(), get_profile(Genre.SCIENCE))
    d = out.dimensions[0]
    assert d.status == DimensionStatus.SCORED
    # The contradicted verdict cites a passage that was never retrieved: downgraded, not counted.
    assert "2 claims supported, 0 contradicted" in d.justification
    assert len(out.usage) == 2


def test_fact_checker_abstains_with_too_few_checkable_claims() -> None:
    def respond(req: LLMRequest, i: int) -> dict[str, Any]:
        if req.schema_name == "claim_extraction":
            return {
                "claims": [
                    {
                        "claim": "x",
                        "segment_index": 0,
                        "quote": "Today we will learn",
                        "claim_type": "factual_assertion",
                    }
                ]
            }
        return {
            "verdicts": [
                {"claim_index": 0, "verdict": "insufficient_evidence", "passage_ids": [], "explanation": "-"}
            ]
        }

    ev = FactualVerificationEvaluator(ScriptedProvider(respond), "m", FakeRetriever([]))
    d = ev.evaluate(make_input(), get_profile(Genre.SCIENCE)).dimensions[0]
    assert d.status == DimensionStatus.INSUFFICIENT_EVIDENCE


def test_lecture_fixture_has_enough_words() -> None:
    assert sum(len(t.split()) for t in LECTURE) > 400
