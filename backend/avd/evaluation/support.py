"""Evidence-support audit: does the cited evidence plausibly support the judgement?

``validation.verify_quote`` establishes that a quote *exists* in the cited segment. This module
asks the second question, *does it support the judgement?*, but only through checks that are
deterministic and whose error modes are known. It flags; it never changes a score. Flags feed
the review queue and the run reports.

* ``invented_quote_in_justification``: the justification quotes text (in quotation marks, at
  least 4 words) that appears nowhere in the transcript.
* ``timestamp_not_in_cited_evidence``: the justification states a time (m:ss or h:mm:ss) that
  no verified cited segment covers. The prompt forbids timestamps, so any that appear must be
  grounded.
* ``contradictory_evidence``: the same segment is cited both for and against the score.
* ``no_supporting_evidence``: the score cites only counter-evidence.
* ``generic_evidence``: one quote is the only support for four or more dimensions, so it is not
  specific to what each dimension measures.

**Not automated: semantic entailment.** Whether a real quote actually supports a particular
claim about depth or clarity needs a judgement this module does not make. No offline NLI model
with a measured error rate on this task is available. Human annotators record their own evidence
for each dimension, and the benchmark compares it. Unflagged therefore means "no detectable
problem", not "supported".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from avd.content.models import Transcript
from avd.evaluation.schema import DimensionScore, DimensionStatus
from avd.evaluation.validation import _fuzzy_contains, _norm

SUPPORT_AUDIT_VERSION = "support_audit_v1"

_QUOTED = re.compile(r"[\"“”']([^\"“”']{12,400})[\"“”']")
_TIMESTAMP = re.compile(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b")
TIMESTAMP_TOLERANCE_S = 30.0
GENERIC_EVIDENCE_MIN_DIMS = 4


@dataclass
class SupportReport:
    flags: dict[str, list[str]] = field(default_factory=dict)  # dimension -> flag codes
    checked_dimensions: int = 0

    @property
    def flagged_dimensions(self) -> list[str]:
        return sorted(d for d, f in self.flags.items() if f)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": SUPPORT_AUDIT_VERSION,
            "checked_dimensions": self.checked_dimensions,
            "flags": {d: f for d, f in sorted(self.flags.items()) if f},
            "flag_counts": dict(Counter(code for f in self.flags.values() for code in f)),
            "semantic_entailment": "not automated",
        }


def _seconds(m: re.Match[str]) -> float:
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def audit(dims: list[DimensionScore], transcript: Transcript | None) -> SupportReport:
    rep = SupportReport()
    if transcript is None:
        return rep
    whole = _norm(" ".join(s.text for s in transcript.segments))
    model_dims = [d for d in dims if d.status == DimensionStatus.SCORED and d.method.startswith("llm")]
    sole_quotes: Counter[str] = Counter()
    for d in model_dims:
        support = [e for e in d.evidence if e.kind == "supporting" and e.verified]
        if len(support) == 1 and support[0].quote:
            sole_quotes[_norm(support[0].quote)] += 1
    for d in model_dims:
        rep.checked_dimensions += 1
        codes: list[str] = []
        text = d.justification or ""
        for m in _QUOTED.finditer(text):
            q = _norm(m.group(1))
            if len(q.split()) >= 4 and not _fuzzy_contains(q, whole):
                codes.append("invented_quote_in_justification")
                break
        spans = [
            (e.start_seconds, e.end_seconds) for e in d.evidence if e.verified and e.start_seconds is not None
        ]
        for m in _TIMESTAMP.finditer(text):
            t = _seconds(m)
            if not any(
                (lo or 0) - TIMESTAMP_TOLERANCE_S
                <= t
                <= (hi if hi is not None else lo or 0) + TIMESTAMP_TOLERANCE_S
                for lo, hi in spans
            ):
                codes.append("timestamp_not_in_cited_evidence")
                break
        sup = {e.segment_index for e in d.evidence if e.kind == "supporting" and e.verified}
        ctr = {e.segment_index for e in d.evidence if e.kind == "counter" and e.verified}
        if sup & ctr:
            codes.append("contradictory_evidence")
        if ctr and not sup:
            codes.append("no_supporting_evidence")
        only = [e for e in d.evidence if e.kind == "supporting" and e.verified]
        if (
            len(only) == 1
            and only[0].quote
            and sole_quotes[_norm(only[0].quote)] >= GENERIC_EVIDENCE_MIN_DIMS
        ):
            codes.append("generic_evidence")
        rep.flags[d.dimension.value] = codes
    return rep
