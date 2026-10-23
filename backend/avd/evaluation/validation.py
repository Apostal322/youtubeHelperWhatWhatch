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
* Citations of segments flagged as prompt-injection attempts are rejected, and so are quotes that
  are popularity statements ("10 million views", "most viewed"): popularity is not evidence.
* Degenerate outputs (identical scores across many dimensions) are flagged.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from avd.content.models import Transcript
from avd.evaluation.sanitize import is_popularity_statement
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
MIN_QUOTE_WORDS = 3  # "at", "not", "e" occur in almost any segment and prove nothing
MIN_FUZZY_WORDS = 6  # near-matches (transcription noise) only for quotes long enough to be specific
_NEGATIONS = frozenset({"not", "no", "never", "nothing", "none", "nobody", "neither", "nor", "t", "without"})


def _norm(s: str) -> str:
    return " ".join(_NON_WORD.sub(" ", s.lower()).split())


def _negations(s: str) -> int:
    return sum(1 for w in s.split() if w in _NEGATIONS)


def _fuzzy_contains(needle: str, hay: str, whole_words: bool = True) -> bool:
    """Whole-word containment; for long quotes, a near match that keeps every negation."""
    if not needle:
        return False
    if (f" {needle} " in f" {hay} ") if whole_words else (needle in hay):
        return True
    if len(needle.split()) < MIN_FUZZY_WORDS:
        return False
    n = len(needle)
    if n > len(hay) + 10:
        return SequenceMatcher(None, needle, hay).ratio() >= FUZZY_THRESHOLD
    step = max(1, n // 4)
    for i in range(0, max(1, len(hay) - n + 1), step):
        window = hay[i : i + n]
        # "does not depend" must never verify against "does depend": negations have to agree.
        if SequenceMatcher(None, needle, window).ratio() >= FUZZY_THRESHOLD and _negations(
            needle
        ) == _negations(window):
            return True
    return False


def verify_quote(ref: EvidenceRef, transcript: Transcript, *, strict: bool = True) -> EvidenceRef | None:
    """Locate a quote at (or next to) its cited segment. ``strict`` (model output): whole words,
    at least MIN_QUOTE_WORDS words. Not strict (a person's text selection, which may start or end
    mid-word): any normalised substring of the segment."""
    if ref.segment_index is None or not ref.quote:
        return None
    parts = [p for p in (_norm(x) for x in _ELLIPSIS.split(ref.quote)) if p]
    # Every part of an elided quote must be found (short parts exactly, as whole words); the
    # quote as a whole must be long enough to be specific to this transcript.
    if not parts or (strict and sum(len(p.split()) for p in parts) < MIN_QUOTE_WORDS):
        return None
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
        if 0 <= idx < n and all(
            _fuzzy_contains(p, _norm(transcript.segments[idx].text), strict) for p in parts
        ):
            return located(idx)
    # 2) the quote spans a segment boundary around the cited segment.
    if 0 <= ref.segment_index < n:
        lo, hi = max(0, ref.segment_index - 1), min(n, ref.segment_index + 2)
        window = _norm(" ".join(s.text for s in transcript.segments[lo:hi]))
        if all(_fuzzy_contains(p, window, strict) for p in parts):
            return located(ref.segment_index)
    return None


def _draws_on_blocked(e: EvidenceRef, transcript: Transcript, blocked: frozenset[int]) -> bool:
    """A quote attributed to a clean segment that is (partly) the text of a neighbouring
    injection-flagged segment: matched across the boundary, it would launder the injection."""
    if not blocked or e.segment_index is None or not e.quote:
        return False
    parts = [p for p in (_norm(x) for x in _ELLIPSIS.split(e.quote)) if len(p.split()) >= MIN_QUOTE_WORDS]
    for idx in (e.segment_index - 1, e.segment_index, e.segment_index + 1):
        if idx in blocked and 0 <= idx < len(transcript.segments):
            seg = _norm(transcript.segments[idx].text)
            words = [w for p in parts for w in p.split()]
            grams = {" ".join(words[i : i + 4]) for i in range(max(0, len(words) - 3))}
            if any(f" {g} " in f" {seg} " for g in grams):
                return True
    return False


def validate_dimensions(
    dims: list[DimensionScore],
    transcript: Transcript | None,
    blocked_segments: frozenset[int] = frozenset(),
) -> tuple[list[DimensionScore], ValidationReport]:
    """Verify model-supplied evidence. ``blocked_segments`` are segments flagged as prompt-injection
    attempts; citing them is rejected (an instruction to the grader is not evidence of quality)."""
    warnings: list[str] = []
    claimed = verified_total = injection_rejected = popularity_rejected = 0
    ungrounded: list[str] = []
    out: list[DimensionScore] = []
    model_scored: list[int] = []
    seg_prefix = transcript.content_hash()[:16] if transcript is not None else None

    def check(e: EvidenceRef) -> EvidenceRef | None:
        nonlocal injection_rejected, popularity_rejected
        if transcript is None:
            return None
        v = verify_quote(e, transcript)
        if v is None:
            return None
        if v.segment_index in blocked_segments or _draws_on_blocked(e, transcript, blocked_segments):
            injection_rejected += 1
            return None
        if e.quote and is_popularity_statement(e.quote):
            # "This video has 2M views" is never evidence of quality (popularity leakage).
            popularity_rejected += 1
            return None
        return v.model_copy(update={"segment_id": f"{seg_prefix}#{v.segment_index}"})

    for d in dims:
        if d.status != DimensionStatus.SCORED:
            out.append(d)
            continue
        pending, seen, dupes = [], set(), 0
        for e in d.evidence:
            if e.verified:
                continue
            key = (e.kind, e.segment_index, _norm(e.quote or ""))
            if key in seen:  # one quote cited four times is one piece of evidence, not four
                dupes += 1
                continue
            seen.add(key)
            pending.append(e)
        if dupes:
            warnings.append(f"{d.dimension.value}: {dupes} duplicate citation(s) ignored")
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
        popularity_citations_rejected=popularity_rejected,
        ungrounded_dimensions=ungrounded,
        evidence_claimed=claimed,
        evidence_verified=verified_total,
        dimensions_invalidated=invalidated,
    )
    return out, report
