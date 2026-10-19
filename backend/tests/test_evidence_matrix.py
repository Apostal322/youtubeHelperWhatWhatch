"""Phase 3: evidence consistency matrix — quote existence (validation) vs support (support audit)."""

from __future__ import annotations

from typing import Any

from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS
from avd.evaluation.schema import Dimension, DimensionScore, DimensionStatus, EvidenceRef
from avd.evaluation.support import audit
from avd.evaluation.validation import verify_quote
from tests.conftest import ScriptedProvider, make_transcript
from tests.test_phase2_llm import judgments, run_llm

SEGS = [
    "Today we will learn how photosynthesis works in green plants.",
    "Chlorophyll absorbs red and blue light because of its electronic structure.",
    "The Calvin cycle fixes carbon dioxide using the enzyme RuBisCO.",
    "However the process is not always efficient in hot dry climates.",
    "In summary light reactions make ATP and the Calvin cycle builds sugars.",
]
T = make_transcript(SEGS, seg_seconds=10.0)


def ref(seg: int, quote: str, kind: str = "supporting") -> EvidenceRef:
    return EvidenceRef(segment_index=seg, quote=quote, kind=kind)  # type: ignore[arg-type]


class TestQuoteExistence:
    def test_nonexistent_quote(self) -> None:
        assert verify_quote(ref(1, "mitochondria are the powerhouse of the cell"), T) is None

    def test_quote_from_a_distant_wrong_segment(self) -> None:
        assert verify_quote(ref(4, "absorbs red and blue light because"), T) is None

    def test_nearby_segment_confusion_is_relocated_with_our_timestamps(self) -> None:
        v = verify_quote(ref(2, "absorbs red and blue light because"), T)
        assert v is not None and v.segment_index == 1 and (v.start_seconds, v.end_seconds) == (10.0, 20.0)

    def test_timestamps_never_come_from_the_model(self) -> None:
        shifted = EvidenceRef(segment_index=1, quote="absorbs red and blue light", start_seconds=999.0)
        v = verify_quote(shifted, T)
        assert v is not None and v.start_seconds == 10.0

    def test_quote_spanning_a_segment_boundary(self) -> None:
        assert verify_quote(ref(2, "dioxide using the enzyme RuBisCO. However the process"), T) is not None

    def test_translated_quote_is_not_verified(self) -> None:
        assert verify_quote(ref(1, "la clorofila absorbe la luz roja y azul"), T) is None

    def test_partial_quote_with_ellipsis_needs_every_part(self) -> None:
        ok = ref(4, "In summary light reactions ... the Calvin cycle builds sugars")
        bad = ref(4, "In summary light reactions ... produce oxygen for animals to breathe")
        assert verify_quote(ok, T) is not None and verify_quote(bad, T) is None


def scored(
    dim: Dimension, justification: str, evidence: list[EvidenceRef], score: float = 4.0
) -> DimensionScore:
    verified = [v for e in evidence if (v := verify_quote(e, T)) is not None]
    return DimensionScore(
        dimension=dim,
        status=DimensionStatus.SCORED,
        score=score,
        confidence=0.6,
        justification=justification,
        method="llm_rubric:rubric_prompt_v2",
        evidence=verified,
    )


class TestSupportAudit:
    def test_clean_judgement_is_not_flagged(self) -> None:
        d = scored(
            Dimension.DEPTH,
            "Explains why chlorophyll absorbs specific wavelengths.",
            [ref(1, "absorbs red and blue light")],
        )
        assert audit([d], T).flags == {"depth": []}

    def test_invented_quote_inside_the_justification(self) -> None:
        d = scored(
            Dimension.CLARITY,
            'The speaker says "light is made of tiny invisible particles of joy" which is clear.',
            [ref(1, "absorbs red and blue light")],
        )
        assert "invented_quote_in_justification" in audit([d], T).flags["clarity"]

    def test_real_quote_inside_the_justification_is_fine(self) -> None:
        d = scored(
            Dimension.CLARITY,
            'Clear: "the Calvin cycle fixes carbon dioxide" is stated plainly.',
            [ref(2, "The Calvin cycle fixes carbon dioxide")],
        )
        assert audit([d], T).flags["clarity"] == []

    def test_timestamp_outside_the_cited_evidence(self) -> None:
        d = scored(
            Dimension.STRUCTURE,
            "At 1:05 the summary ties it together.",
            [ref(1, "absorbs red and blue light")],
        )
        assert "timestamp_not_in_cited_evidence" in audit([d], T).flags["structure"]
        ok = scored(
            Dimension.STRUCTURE, "At 0:15 the mechanism is explained.", [ref(1, "absorbs red and blue light")]
        )
        assert audit([ok], T).flags["structure"] == []

    def test_contradictory_and_counter_only_evidence(self) -> None:
        both = scored(
            Dimension.DEPTH,
            "Deep.",
            [ref(3, "not always efficient"), ref(3, "not always efficient", kind="counter")],
        )
        counter_only = scored(
            Dimension.PACING, "Good pacing.", [ref(3, "not always efficient", kind="counter")]
        )
        flags = audit([both, counter_only], T).flags
        assert "contradictory_evidence" in flags["depth"] and "no_supporting_evidence" in flags["pacing"]

    def test_one_quote_supporting_many_dimensions_is_generic(self) -> None:
        dims = [scored(d, "Good.", [ref(0, "learn how photosynthesis works")]) for d in TEXT_DIMENSIONS[:4]]
        flags = audit(dims, T).flags
        assert all("generic_evidence" in f for f in flags.values())

    def test_heuristic_and_abstained_dimensions_are_not_audited(self) -> None:
        h = scored(Dimension.DEPTH, 'Says "nothing like this".', [])
        h = h.model_copy(update={"method": "heuristic_baseline_v1"})
        a = DimensionScore.abstain(
            Dimension.ORIGINALITY, DimensionStatus.INSUFFICIENT_EVIDENCE, "n/a", "llm_rubric"
        )
        assert audit([h, a], T).checked_dimensions == 0


def test_pipeline_routes_severe_support_flags_to_review() -> None:
    def resp(_req: Any, _i: int) -> dict[str, Any]:
        out = judgments(4)
        out["judgments"][0]["justification"] = (
            'The host claims "photosynthesis happens only at night in deserts" which is deep.'
        )
        return out

    result = run_llm(ScriptedProvider(resp))
    sa = result.coverage_report["support_audit"]
    assert sa["flag_counts"].get("invented_quote_in_justification") == 1
    assert sa["semantic_entailment"] == "not automated"
    assert result.needs_review and any("justification cites text" in r for r in result.review_reasons)
