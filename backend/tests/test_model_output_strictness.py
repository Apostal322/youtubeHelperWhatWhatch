"""Model output and runtime responses are taken at face value only when they are what they claim
(audit 2026-10-23). All model responses are scripted test doubles or the Ollama protocol mock."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from avd.evaluation.evaluators.factual import FactualVerificationEvaluator, ReferencePassage
from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS, LLMEvaluationOutput, LLMRubricEvaluator
from avd.evaluation.pipeline import EvaluationPipeline
from avd.evaluation.schema import Dimension, EvaluationMode
from avd.llm.base import LLMRequest, LLMTransientError
from avd.llm.ollama import LLMContextOverflow, OllamaProvider
from tests.conftest import LECTURE, ScriptedProvider, make_input, make_transcript
from tests.ollama_mock import FakeOllama


def _j(dim: str, score: Any = 4) -> dict[str, Any]:
    return {
        "dimension": dim,
        "assessable": True,
        "score": score,
        "confidence": 0.7,
        "justification": "x",
        "evidence": [],
        "counter_evidence": [],
        "unassessable_reason": "",
    }


@pytest.mark.parametrize(
    "judgments",
    [
        [_j("depth", "5")],  # a string is not a score (lax parsing made it 5)
        [_j("depth", True)],  # nor is a boolean (lax parsing made it 1)
        [_j("depth", 5), _j("depth", 1)],  # two different scores for one dimension
    ],
)
def test_ambiguous_model_judgments_are_rejected(judgments: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError):
        LLMEvaluationOutput.model_validate_json(json.dumps({"content_summary": "", "judgments": judgments}))


def _ollama(fake: FakeOllama, **kw: Any) -> OllamaProvider:
    return OllamaProvider("http://localhost:11434", transport=fake.transport(), **kw)


def _req(user: str = "[0] (00:00) hello world") -> LLMRequest:
    return LLMRequest(
        system="sys",
        user=user,
        json_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        schema_name="t",
        model="qwen3.5:4b",
        max_tokens=512,
    )


@pytest.mark.parametrize("status", [408, 504])
def test_gateway_and_request_timeouts_are_retried(status: int) -> None:
    fake = FakeOllama()
    fake.chat_status = status  # e.g. a reverse proxy in front of Ollama timed out
    with pytest.raises(LLMTransientError):  # before: a permanent error failing every dimension
        _ollama(fake).complete_json(_req())


def test_truncation_is_detected_when_ollama_omits_the_prompt_count() -> None:
    fake = FakeOllama()
    fake.chat_override = {"prompt_eval_count": None, "eval_count": 5000}  # prompt served from cache
    with pytest.raises(LLMContextOverflow):  # before: accepted as complete output
        _ollama(fake, num_ctx=4096).complete_json(_req())


class _Refs:
    def search(self, q: str, k: int = 3) -> list[tuple[ReferencePassage, float]]:
        p = ReferencePassage(passage_id="p1", doc_id="d", title="t", license="cc", text="ten percent")
        return [(p, 0.9)]


def _fact_pipeline(verify: Any, claims: list[dict[str, Any]]) -> Any:
    def respond(req: LLMRequest, i: int) -> Any:
        if req.schema_name == "claim_extraction":
            return {"claims": claims}
        if req.schema_name == "claim_verification":
            return verify(len(claims))
        return {
            "content_summary": "",
            "detected_issues": [],
            "judgments": [
                {
                    **_j(d.value, 5),
                    "evidence": [{"segment_index": 1, "quote": "Chlorophyll absorbs red and blue light"}],
                }
                for d in TEXT_DIMENSIONS
            ],
        }

    p = ScriptedProvider(respond)
    pipe = EvaluationPipeline(
        [LLMRubricEvaluator(p, "m"), FactualVerificationEvaluator(p, "m", _Refs())], "C", EvaluationMode.LLM
    )
    return pipe.run(make_input(make_transcript(LECTURE)))


CLAIM = {
    "claim": "1% stored",
    "segment_index": 3,
    "quote": "roughly one percent of light is stored",
    "claim_type": "factual_assertion",
}


def test_a_failed_fact_check_falling_back_to_the_rubric_is_flagged() -> None:
    r = _fact_pipeline(lambda n: {"verdicts": "nope"}, [CLAIM])
    assert r.dimension(Dimension.FACTUAL_ACCURACY).method.startswith("llm_rubric")
    assert r.needs_review and any("fallback" in x for x in r.review_reasons)  # before: not reviewed


def test_one_claim_extracted_three_times_counts_once() -> None:
    contradicted = lambda n: {  # noqa: E731
        "verdicts": [
            {"claim_index": k, "verdict": "contradicted", "passage_ids": ["p1"], "explanation": "e"}
            for k in range(n)
        ]
    }
    once = _fact_pipeline(contradicted, [CLAIM]).dimension(Dimension.FACTUAL_ACCURACY)
    thrice = _fact_pipeline(contradicted, [CLAIM] * 3).dimension(Dimension.FACTUAL_ACCURACY)
    assert thrice.score == once.score  # before: the repeated contradiction weighed three times
