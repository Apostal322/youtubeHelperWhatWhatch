"""Confidence adjustment.

Model self-reported confidence is known to be poorly calibrated, so it is treated as one input and
combined with structural factors. The resulting values are *uncalibrated indicators* until the
benchmark harness has measured calibration (ECE) against human labels; the API says so.
"""

from __future__ import annotations

from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.schema import Dimension, DimensionScore, DimensionStatus

# Upper bounds per method family.
METHOD_CAPS: list[tuple[str, float]] = [
    ("heuristic_baseline", 0.6),
    ("audio_signal_metrics", 0.7),
    ("retrieval_fact_check", 0.85),
    ("llm_rubric", 0.9),
]
# Factual accuracy judged from a model's internal knowledge (no retrieval) is capped hard.
LLM_FACTUAL_CAP = 0.5


def completeness_factor(inp: EvaluationInput) -> float:
    if inp.transcript is None:
        return 1.0
    words = inp.transcript.word_count
    if words < 150:
        return 0.5
    if words < 400:
        return 0.8
    return 1.0


def adjust_confidence(d: DimensionScore, inp: EvaluationInput) -> DimensionScore:
    """Structural adjustments; every factor applied is recorded in ``uncertainty.adjustments``."""
    if d.status != DimensionStatus.SCORED or d.confidence is None:
        return d
    conf = d.confidence
    adjustments: list[str] = []
    if d.dimension.group.value != "technical":
        factor = completeness_factor(inp)
        if factor < 1.0:
            conf *= factor
            adjustments.append(f"short_transcript_x{factor}")
    for prefix, cap in METHOD_CAPS:
        if d.method.startswith(prefix):
            if conf > cap:
                adjustments.append(f"method_cap_{cap}")
            conf = min(conf, cap)
            break
    missing = list(d.missing_inputs)
    if d.dimension == Dimension.FACTUAL_ACCURACY and d.method.startswith("llm_rubric"):
        if conf > LLM_FACTUAL_CAP:
            adjustments.append(f"unretrieved_factual_cap_{LLM_FACTUAL_CAP}")
        conf = min(conf, LLM_FACTUAL_CAP)
        if "reference_corpus" not in missing:
            missing.append("reference_corpus")
    uncertainty = {**d.uncertainty, "adjustments": adjustments, "calibrated": False}
    return d.model_copy(
        update={"confidence": round(conf, 4), "missing_inputs": missing, "uncertainty": uncertainty}
    )
