"""Evaluation pipeline: evaluators -> merge -> validation -> confidence -> aggregation.

The pipeline is a pure function of an ``EvaluationInput`` (plus evaluator configuration), which
makes it testable and reproducible. Persistence and scheduling live in ``avd.evaluation.service``
and ``avd.workers``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from avd.config import Settings
from avd.evaluation import schema as _schema
from avd.evaluation.aggregation import aggregate
from avd.evaluation.confidence import adjust_confidence
from avd.evaluation.evaluators.audio import AudioSignalEvaluator
from avd.evaluation.evaluators.base import Evaluator
from avd.evaluation.evaluators.factual import (
    FactualVerificationEvaluator,
    LocalReferenceCorpus,
    ReferenceRetriever,
)
from avd.evaluation.evaluators.heuristic import HeuristicTextEvaluator
from avd.evaluation.evaluators.llm_rubric import LLMRubricEvaluator
from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.profiles import get_profile
from avd.evaluation.rubric import RUBRIC
from avd.evaluation.schema import (
    Dimension,
    DimensionScore,
    DimensionStatus,
    EvaluationMode,
    EvaluationResult,
    UsageRecord,
)
from avd.evaluation.validation import validate_dimensions
from avd.llm.base import LLMError, LLMProvider
from avd.logging import get_logger

log = get_logger(__name__)


class EvaluationPipeline:
    def __init__(self, evaluators: Sequence[Evaluator], stage: str, mode: EvaluationMode) -> None:
        # Later evaluators take precedence for dimensions they *score* (e.g. retrieval-based fact
        # checking overrides the rubric's knowledge-based factual judgement).
        self.evaluators = list(evaluators)
        self.stage = stage
        self.mode = mode

    def run(self, inp: EvaluationInput) -> EvaluationResult:
        profile = get_profile(inp.genre.genre)
        candidates: dict[Dimension, list[tuple[int, DimensionScore]]] = {d: [] for d in Dimension}
        usage: list[UsageRecord] = []
        models: list[str] = []
        notes: list[str] = []

        for prec, ev in enumerate(self.evaluators):
            missing = sorted(m.value for m in ev.requires - inp.modalities)
            if missing:
                for d in ev.dimensions:
                    candidates[d].append(
                        (
                            prec,
                            DimensionScore.abstain(
                                d,
                                DimensionStatus.REQUIRES_MEDIA,
                                f"Requires authorized {', '.join(missing)}, which is not available.",
                                ev.version,
                                missing_inputs=missing,
                            ),
                        )
                    )
                continue
            try:
                out = ev.evaluate(inp, profile)
            except LLMError as exc:
                if exc.retryable:
                    raise  # transient: the job runner retries; nothing is persisted
                out_dims = self._failed(ev, f"{type(exc).__name__}: {exc}")
                notes.append(f"{ev.name} failed: {exc}")
                for failed in out_dims:
                    candidates[failed.dimension].append((prec, failed))
                continue
            except Exception as exc:
                log.exception("evaluator crashed", extra={"evaluator": ev.name, "video_id": inp.video_id})
                for failed in self._failed(ev, f"{type(exc).__name__}: {exc}"):
                    candidates[failed.dimension].append((prec, failed))
                notes.append(f"{ev.name} crashed: {type(exc).__name__}")
                continue
            returned = {r.dimension for r in out.dimensions}
            for r in out.dimensions:
                candidates[r.dimension].append((prec, r))
            for d in set(ev.dimensions) - returned:
                candidates[d].append(
                    (
                        prec,
                        DimensionScore.abstain(
                            d, DimensionStatus.FAILED, "Evaluator returned no result.", ev.version
                        ),
                    )
                )
            usage.extend(out.usage)
            models.extend(m for m in out.models if m not in models)
            notes.extend(out.notes)

        merged: list[DimensionScore] = []
        for d in Dimension:
            if not profile.applicable(d):
                merged.append(
                    DimensionScore.abstain(
                        d,
                        DimensionStatus.NOT_APPLICABLE,
                        f"Weight 0 in the '{profile.genre.value}' genre profile.",
                        "genre_profile",
                    )
                )
                continue
            options = sorted(candidates[d], key=lambda x: -x[0])
            scored = [o for _, o in options if o.status == DimensionStatus.SCORED]
            if scored:
                merged.append(scored[0])
            elif options:
                merged.append(options[0][1])
            else:
                need = sorted(m.value for m in RUBRIC[d].requires - inp.modalities)
                if need:
                    merged.append(
                        DimensionScore.abstain(
                            d,
                            DimensionStatus.REQUIRES_MEDIA,
                            f"Requires authorized {', '.join(need)}; no such input is available.",
                            "pipeline",
                            missing_inputs=need,
                        )
                    )
                else:
                    merged.append(
                        DimensionScore.abstain(
                            d,
                            DimensionStatus.INSUFFICIENT_EVIDENCE,
                            "No evaluator for this dimension is configured.",
                            "pipeline",
                        )
                    )

        validated, report = validate_dimensions(merged, inp.transcript)
        report.warnings.extend(notes)
        final = [adjust_confidence(d, inp) for d in validated]
        agg = aggregate(final, profile)

        scored_any = any(d.status == DimensionStatus.SCORED for d in final)
        failed_any = any(d.status == DimensionStatus.FAILED for d in final)
        if agg.overall is not None:
            status = "complete"
        elif scored_any:
            status = "partial"
        elif failed_any:
            status = "failed"
        else:
            status = "abstained"
        missing_inputs = sorted({m for d in final for m in d.missing_inputs})
        return EvaluationResult(
            video_id=inp.video_id,
            stage=self.stage,
            framework_version=_schema.FRAMEWORK_VERSION,
            mode=self.mode,
            is_synthetic=self.mode == EvaluationMode.SYNTHETIC_DEMO,
            models=models,
            genre=inp.genre,
            input_fingerprint=inp.fingerprint(),
            content_source_ids=list(inp.content_source_ids),
            dimensions=final,
            aggregation=agg,
            validation=report,
            missing_inputs=missing_inputs,
            usage=usage,
            status=status,
        )

    @staticmethod
    def _failed(ev: Evaluator, reason: str) -> list[DimensionScore]:
        return [
            DimensionScore.abstain(d, DimensionStatus.FAILED, f"Evaluator failed: {reason}"[:500], ev.version)
            for d in ev.dimensions
        ]


def build_pipeline(
    stage: str,
    settings: Settings,
    provider: LLMProvider | None,
    retriever: ReferenceRetriever | None = None,
) -> EvaluationPipeline:
    if provider is None:
        return EvaluationPipeline(
            [HeuristicTextEvaluator(), AudioSignalEvaluator()], stage, EvaluationMode.SYNTHETIC_DEMO
        )
    if stage == "C":
        evaluators: list[Evaluator] = [
            LLMRubricEvaluator(
                provider,
                settings.llm_model_stage_c,
                effort=settings.llm_effort_stage_c,
                max_tokens=settings.llm_max_output_tokens,
                samples=settings.llm_self_consistency_samples_stage_c,
            ),
            AudioSignalEvaluator(),
        ]
        if retriever is not None:
            evaluators.append(FactualVerificationEvaluator(provider, settings.llm_model_stage_c, retriever))
        return EvaluationPipeline(evaluators, stage, EvaluationMode.LLM)
    return EvaluationPipeline(
        [
            LLMRubricEvaluator(
                provider,
                settings.llm_model_stage_b,
                effort=settings.llm_effort_stage_b,
                max_tokens=settings.llm_max_output_tokens,
            ),
            AudioSignalEvaluator(),
        ],
        stage,
        EvaluationMode.LLM,
    )


def load_reference_corpus(directory: Path | None) -> ReferenceRetriever | None:
    if directory is None or not directory.exists():
        return None
    from avd.embeddings.base import get_embedder

    corpus = LocalReferenceCorpus(directory, get_embedder())
    return corpus if corpus.passages else None
