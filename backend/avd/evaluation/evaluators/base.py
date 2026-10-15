from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.profiles import GenreProfile
from avd.evaluation.schema import Dimension, DimensionScore, Modality, UsageRecord


@dataclass
class EvaluatorOutput:
    dimensions: list[DimensionScore]
    usage: list[UsageRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)


class Evaluator(Protocol):
    """A component that judges a subset of dimensions.

    Contract: return exactly one ``DimensionScore`` per dimension in ``self.dimensions`` (scored or
    an explicit abstention). Raising is allowed; the pipeline converts exceptions into ``failed``
    dimensions and never substitutes a score.
    """

    name: str
    version: str
    dimensions: tuple[Dimension, ...]
    requires: frozenset[Modality]
    synthetic: bool

    def evaluate(self, inp: EvaluationInput, profile: GenreProfile) -> EvaluatorOutput: ...
