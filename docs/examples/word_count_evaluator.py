"""Example experimental evaluator: a documented template, not part of the official protocol.

It judges `information_density` from a transparent word-rate rule and abstains when the input
cannot support a judgement. Enable it with:

    AVD_EXPERIMENTAL_EVALUATORS='word_count_evaluator:WordRateEvaluator' PYTHONPATH=docs/examples avd start

Because experimental evaluators have the lowest precedence, it only fills `information_density`
when the official evaluators abstain on it, and every affected evaluation records
`versions["experimental_evaluators"] = "word_rate@0.1"`.
"""

from __future__ import annotations

from avd.evaluation.evaluators.base import EvaluatorOutput
from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.profiles import GenreProfile
from avd.evaluation.schema import (
    Dimension,
    DimensionScore,
    DimensionStatus,
    EvidenceRef,
    Modality,
    ReasonCode,
)


class WordRateEvaluator:
    name = "word_rate"
    version = "0.1"
    dimensions = (Dimension.INFORMATION_DENSITY,)
    requires = frozenset({Modality.TRANSCRIPT})
    synthetic = True  # a rule, not a model: never presented as AI

    def evaluate(self, inp: EvaluationInput, profile: GenreProfile) -> EvaluatorOutput:
        d = Dimension.INFORMATION_DENSITY
        t = inp.transcript
        if t is None or not inp.duration_seconds or len(t.segments) < 5:
            # Contract: one DimensionScore per dimension, abstaining explicitly when unsure.
            return EvaluatorOutput(
                dimensions=[
                    DimensionScore.abstain(
                        d,
                        DimensionStatus.INSUFFICIENT_EVIDENCE,
                        "Too little transcript or no duration to measure a word rate.",
                        self.version,
                        reason_code=ReasonCode.INSUFFICIENT_MATERIAL,
                    )
                ]
            )
        words = sum(len(s.text.split()) for s in t.segments)
        wpm = words / (inp.duration_seconds / 60)
        score = 1.0 + min(4.0, max(0.0, (wpm - 60) / 40))  # 60 wpm -> 1, 220+ wpm -> 5
        first = t.segments[0]
        return EvaluatorOutput(
            dimensions=[
                DimensionScore(
                    dimension=d,
                    status=DimensionStatus.SCORED,
                    score=round(score, 2),
                    confidence=0.2,  # a crude proxy: say so
                    justification=f"{wpm:.0f} words per minute over the transcript (rule, not a model).",
                    evidence=[EvidenceRef(segment_index=0, quote=" ".join(first.text.split()[:8]))],
                    method=f"{self.name}@{self.version}",
                )
            ]
        )
