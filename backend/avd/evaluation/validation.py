"""Quality validation of evaluator outputs.

* Evidence grounding: every model-supplied quote must be found (exactly after normalisation, or
  with high fuzzy similarity) in the cited transcript segment or its immediate neighbours. Verified
  evidence receives timestamps from *our* transcript data; unverifiable evidence is dropped.
* A scored dimension whose cited supporting evidence is all unverifiable becomes ``failed``
  (``validation_failed``): its justification is unsupported. A dimension scored without citing any
  evidence keeps its score but is marked ``grounded: false``. No numeric confidence penalty is
  applied (there is no empirical basis for its size).
* If >= 4 quotes were claimed and < 50 % verify, all model-scored dimensions are invalidated — a
  strong hallucination signal. They are not re-scored or replaced.
* Citations of segments flagged as prompt-injection attempts are rejected.
* Degenerate outputs (identical scores across many dimensions) are flagged.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from avd.content.models import Transcript
from avd.evaluation.schema import (
    DimensionScore,
    DimensionStatus,
    EvidenceRef,
    ReasonCode,
    ValidationReport,
)

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_ELLIPSIS = re.compile(r"\.\.\.|…")

GROUNDING_INVALIDATION_THRESHOLD = 0.5
MIN_CLAIMS_FOR_INVALIDATION = 4
FUZZY_THRESHOLD = 0.85


def _norm(s: str) -> str:
    return " ".join(_NON_WORD.sub(" ", s.lower()).split())


def _fuzzy_contains(needle: str, hay: str) -> bool:
    if not needle:
        return False
    if needle in hay:
        return True
    n = len(needle)
    if n > len(hay) + 10:
        return SequenceMatcher(None, needle, hay).ratio() >= FUZZY_THRESHOLD
    best = 0.0
    step = max(1, n // 4)
    for i in range(0, max(1, len(hay) - n + 1), step):
        best = max(best, SequenceMatcher(None, needle, hay[i : i + n]).ratio())
        if best >= FUZZY_THRESHOLD:
            return True
    return False


def verify_quote(ref: EvidenceRef, transcript: Transcript) -> EvidenceRef | None:
    if ref.segment_index is None or not ref.quote:
        return None
    parts = [_norm(p) for p in _ELLIPSIS.split(ref.quote)]
    parts = [p for p in parts if len(p.split()) >= 3] or [_norm(ref.quote)]
    n = len(transcript.segments)

    def located(idx: int) -> EvidenceRef:
        seg = transcript.segments[idx]
        return ref.model_copy(
            update={
                "segment_index": idx,
                "start_seconds": seg.start,
                "end_seconds": seg.end,
                "verified": True,
            }
        )

    # 1) the quote lies within a single segment: the cited one, else an immediate neighbour
    #    (models are often off by one); the timestamp is that of the segment that contains it.
    for idx in (ref.segment_index, ref.segment_index - 1, ref.segment_index + 1):
        if 0 <= idx < n and all(_fuzzy_contains(p, _norm(transcript.segments[idx].text)) for p in parts):
            return located(idx)
    # 2) the quote spans a segment boundary around the cited segment.
    if 0 <= ref.segment_index < n:
        lo, hi = max(0, ref.segment_index - 1), min(n, ref.segment_index + 2)
        window = _norm(" ".join(s.text for s in transcript.segments[lo:hi]))
        if all(_fuzzy_contains(p, window) for p in parts):
            return located(ref.segment_index)
    return None


def validate_dimensions(
    dims: list[DimensionScore],
    transcript: Transcript | None,
    blocked_segments: frozenset[int] = frozenset(),
) -> tuple[list[DimensionScore], ValidationReport]:
    """Verify model-supplied evidence. ``blocked_segments`` are segments flagged as prompt-injection
    attempts; citing them is rejected (an instruction to the grader is not evidence of quality)."""
    warnings: list[str] = []
    claimed = verified_total = injection_rejected = 0
    ungrounded: list[str] = []
    out: list[DimensionScore] = []
    model_scored: list[int] = []
    seg_prefix = transcript.content_hash()[:16] if transcript is not None else None

    def check(e: EvidenceRef) -> EvidenceRef | None:
        nonlocal injection_rejected
        if transcript is None:
            return None
        v = verify_quote(e, transcript)
        if v is None:
            return None
        if v.segment_index in blocked_segments:
            injection_rejected += 1
            return None
        return v.model_copy(update={"segment_id": f"{seg_prefix}#{v.segment_index}"})

    for d in dims:
        if d.status != DimensionStatus.SCORED:
            out.append(d)
            continue
        pending = [e for e in d.evidence if not e.verified]
        already = [
            e.model_copy(update={"segment_id": f"{seg_prefix}#{e.segment_index}"})
            if seg_prefix and e.segment_index is not None and e.segment_id is None
            else e
            for e in d.evidence
            if e.verified
        ]
        if not pending:
            if not d.evidence and d.method.startswith("llm"):
                ungrounded.append(d.dimension.value)
                d = d.model_copy(update={"uncertainty": {**d.uncertainty, "grounded": False}})
            out.append(d.model_copy(update={"evidence": already}))
            continue
        model_scored.append(len(out))
        claimed += len(pending)
        ok = [v for e in pending if (v := check(e)) is not None]
        verified_total += len(ok)
        supporting_claimed = sum(1 for e in pending if e.kind == "supporting")
        supporting_ok = sum(1 for e in ok if e.kind == "supporting")
        unc = {**d.uncertainty, "evidence_claimed": len(pending), "evidence_verified": len(ok)}
        if supporting_claimed and not supporting_ok:
            warnings.append(f"{d.dimension.value}: none of {supporting_claimed} supporting quotes verified")
            out.append(
                DimensionScore.abstain(
                    d.dimension,
                    DimensionStatus.FAILED,
                    "Unsupported: none of the cited supporting evidence could be verified against the transcript.",
                    d.method,
                    model=d.model,
                    reason_code=ReasonCode.VALIDATION_FAILED,
                ).model_copy(update={"uncertainty": unc})
            )
            continue
        if len(ok) < len(pending):
            warnings.append(
                f"{d.dimension.value}: {len(pending) - len(ok)} of {len(pending)} quotes not verified"
            )
        out.append(d.model_copy(update={"evidence": already + ok, "uncertainty": unc}))

    invalidated = []
    if claimed >= MIN_CLAIMS_FOR_INVALIDATION and verified_total / claimed < GROUNDING_INVALIDATION_THRESHOLD:
        warnings.append(
            f"evidence grounding failed: {verified_total}/{claimed} quotes verified; model-scored "
            "dimensions invalidated"
        )
        for i in model_scored:
            d = out[i]
            invalidated.append(d.dimension)
            if d.status == DimensionStatus.SCORED:
                out[i] = DimensionScore.abstain(
                    d.dimension,
                    DimensionStatus.FAILED,
                    "Invalidated: most evidence quotes in this evaluation could not be verified against the transcript.",
                    d.method,
                    model=d.model,
                    reason_code=ReasonCode.VALIDATION_FAILED,
                )
    if injection_rejected:
        warnings.append(f"{injection_rejected} citation(s) of injection-flagged segments rejected")
    if ungrounded:
        warnings.append(f"scored without any cited evidence: {', '.join(ungrounded)}")

    scores = [d.score for d in out if d.status == DimensionStatus.SCORED]
    if len(scores) >= 6 and len(set(scores)) == 1:
        warnings.append(f"degenerate output: all {len(scores)} scored dimensions = {scores[0]}")

    report = ValidationReport(
        passed=not invalidated,
        warnings=warnings,
        injection_citations_rejected=injection_rejected,
        ungrounded_dimensions=ungrounded,
        evidence_claimed=claimed,
        evidence_verified=verified_total,
        dimensions_invalidated=invalidated,
    )
    return out, report
