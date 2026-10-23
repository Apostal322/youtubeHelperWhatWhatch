"""Evaluation pipeline: evaluators -> merge -> validation -> confidence -> aggregation.

The pipeline is a pure function of an ``EvaluationInput`` (plus evaluator configuration), which
makes it testable and reproducible. Persistence and scheduling live in ``avd.evaluation.service``
and ``avd.workers``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from avd.config import Settings
from avd.content.processing import validate_transcript
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
from avd.evaluation.evaluators.visual import VisualSignalEvaluator
from avd.evaluation.injection import scan as scan_injection
from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.profiles import profile_for
from avd.evaluation.prompts import get_prompt
from avd.evaluation.rubric import RUBRIC
from avd.evaluation.schema import (
    Dimension,
    DimensionScore,
    DimensionStatus,
    EvaluationMode,
    EvaluationResult,
    ReasonCode,
    UsageRecord,
)
from avd.evaluation.support import audit as audit_support
from avd.evaluation.validation import validate_dimensions
from avd.llm.base import LLMBudgetExceeded, LLMError, LLMProvider
from avd.logging import get_logger

log = get_logger(__name__)


class EvaluationPipeline:
    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        stage: str,
        mode: EvaluationMode,
        experimental: Sequence[Evaluator] = (),
    ) -> None:
        # Later evaluators take precedence for dimensions they *score* (e.g. retrieval-based fact
        # checking overrides the rubric's knowledge-based factual judgement). Experimental
        # (operator-configured) evaluators go first, so they can only fill official abstentions.
        self.experimental = [f"{ev.name}@{ev.version}" for ev in experimental]
        self.evaluators = [*experimental, *evaluators]
        self.stage = stage
        self.mode = mode

    def _versions(self) -> dict[str, str]:
        v = dict(self.versions())
        if self.experimental:
            v["experimental_evaluators"] = ",".join(self.experimental)
        return v

    @staticmethod
    def versions() -> dict[str, str]:
        return {
            "framework": _schema.FRAMEWORK_VERSION,
            "rubric": _schema.RUBRIC_VERSION,
            "prompt": _schema.PROMPT_VERSION,
            "aggregation": _schema.AGGREGATION_VERSION,
            "preprocessing": _schema.PREPROCESSING_VERSION,
            "injection_detector": _schema.INJECTION_DETECTOR_VERSION,
            "evidence_validation": _schema.EVIDENCE_VALIDATION_VERSION,
        }

    def config(self) -> dict[str, Any]:
        descriptors = []
        for ev in self.evaluators:
            desc = getattr(ev, "descriptor", None)
            descriptors.append(desc() if callable(desc) else {"evaluator": ev.name, "version": ev.version})
        return {
            "stage": self.stage,
            "mode": self.mode.value,
            "versions": self._versions(),
            "evaluators": descriptors,
        }

    def config_hash(self) -> str:
        canon = json.dumps(self.config(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canon.encode()).hexdigest()

    def run(self, inp: EvaluationInput) -> EvaluationResult:
        started = time.monotonic()
        profile = profile_for(inp.genre)
        candidates: dict[Dimension, list[tuple[int, DimensionScore]]] = {d: [] for d in Dimension}
        usage: list[UsageRecord] = []
        models: list[str] = []
        notes: list[str] = []
        evaluator_reports: dict[str, Any] = {}
        transcript_report = validate_transcript(inp.transcript).as_dict() if inp.transcript else None
        findings = scan_injection(
            [s.text for s in inp.transcript.segments] if inp.transcript else [],
            inp.title,
            inp.description,
            inp.tags,
        )
        blocked = frozenset(f.segment_index for f in findings if f.segment_index is not None)

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
                if exc.retryable or isinstance(exc, LLMBudgetExceeded):
                    raise  # transient or budget: the job is retried/deferred; nothing is persisted
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
            if out.report:
                evaluator_reports[ev.name] = out.report

        merged: list[DimensionScore] = []
        for d in Dimension:
            if d == Dimension.TITLE_CONSISTENCY and not inp.title_is_original:
                merged.append(
                    DimensionScore.abstain(
                        d,
                        DimensionStatus.INSUFFICIENT_EVIDENCE,
                        "No original title is available (the title is a placeholder), so title "
                        "consistency cannot be judged.",
                        "pipeline",
                        reason_code=ReasonCode.INSUFFICIENT_MATERIAL,
                    )
                )
                continue
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
                            reason_code=ReasonCode.NO_EVALUATOR,
                        )
                    )

        validated, report = validate_dimensions(merged, inp.transcript, blocked)
        report.warnings.extend(notes)
        report.injection_findings = [asdict(f) for f in findings[:20]]
        final = [
            adjust_confidence(d, inp).model_copy(
                update={"rubric_anchor": RUBRIC[d.dimension].anchor_for(d.score)}
            )
            if d.status == DimensionStatus.SCORED
            else d
            for d in validated
        ]
        agg = aggregate(final, profile)
        support = audit_support(final, inp.transcript)

        review: list[str] = []
        severe = sorted(
            d
            for d, codes in support.flags.items()
            if {"invented_quote_in_justification", "timestamp_not_in_cited_evidence"} & set(codes)
        )
        if severe:
            review.append(f"justification cites text or times not in the evidence: {', '.join(severe)}")
        if findings:
            review.append(f"suspected prompt injection ({len(findings)} finding(s))")
        if not report.passed:
            review.append("evidence grounding failed")
        short = next((d.uncertainty for d in final if "samples_succeeded" in d.uncertainty), None)
        if short:
            review.append(
                f"only {short['samples_succeeded']} of {short['samples_requested']} model samples succeeded; "
                "confidence lowered accordingly"
            )
        disagreeing = [d.dimension.value for d in final if d.uncertainty.get("disagreement")]
        if disagreeing:
            review.append(f"model samples disagree on: {', '.join(disagreeing)}")
        if transcript_report and transcript_report["language_mismatch"]:
            review.append("declared transcript language differs from detected language")
        mapped = evaluator_reports.get("llm_rubric") or {}
        if mapped.get("chunks_failed"):
            review.append(
                f"{mapped['chunks_failed']} of {mapped.get('chunks')} transcript parts could not be analysed "
                f"({', '.join(mapped.get('failed_parts', [])[:3])}): the evaluation is partial"
            )
        if mapped.get("map_grounding_failed"):
            review.append(
                f"part notes are poorly grounded: {mapped.get('map_citations_verified')} of "
                f"{mapped.get('map_citations_claimed')} quoted points found in the transcript"
            )
        high_stakes = (evaluator_reports.get("fact_check") or {}).get("high_stakes_contradicted", 0)
        if high_stakes:
            review.append(f"contradicted high-stakes claims (health/finance/legal/safety): {high_stakes}")

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
            config_hash=self.config_hash(),
            versions=self._versions(),
            coverage_report={
                "transcript": transcript_report,
                "evaluators": evaluator_reports,
                "modalities": sorted(m.value for m in inp.modalities),
                "sanitized_popularity_spans": inp.sanitized_spans,
                "support_audit": support.as_dict(),
            },
            needs_review=bool(review),
            review_reasons=review,
            latency_ms=int((time.monotonic() - started) * 1000),
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
    from avd.plugins import load_evaluators

    experimental = load_evaluators(settings.experimental_evaluators)
    if provider is None:
        return EvaluationPipeline(
            [HeuristicTextEvaluator(), AudioSignalEvaluator(), VisualSignalEvaluator()],
            stage,
            EvaluationMode.SYNTHETIC_DEMO,
            experimental,
        )
    if stage == "C":
        evaluators: list[Evaluator] = [
            LLMRubricEvaluator(
                provider,
                settings.llm_model_stage_c,
                effort=settings.llm_effort_stage_c,
                max_tokens=settings.llm_max_output_tokens,
                samples=settings.llm_self_consistency_samples_stage_c,
                temperature=settings.llm_temperature,
                single_pass_max_tokens=settings.llm_single_pass_max_tokens,
                chunk_target_tokens=settings.llm_chunk_target_tokens,
                prompt=get_prompt(settings.llm_prompt),
            ),
            AudioSignalEvaluator(),
            VisualSignalEvaluator(),
        ]
        if retriever is not None:
            evaluators.append(
                FactualVerificationEvaluator(
                    provider,
                    settings.llm_model_stage_c,
                    retriever,
                    chunk_target_tokens=settings.llm_chunk_target_tokens,
                    single_pass_max_tokens=settings.llm_single_pass_max_tokens,
                )
            )
        return EvaluationPipeline(evaluators, stage, EvaluationMode.LLM, experimental)
    return EvaluationPipeline(
        [
            LLMRubricEvaluator(
                provider,
                settings.llm_model_stage_b,
                effort=settings.llm_effort_stage_b,
                max_tokens=settings.llm_max_output_tokens,
                temperature=settings.llm_temperature,
                single_pass_max_tokens=settings.llm_single_pass_max_tokens,
                chunk_target_tokens=settings.llm_chunk_target_tokens,
                prompt=get_prompt(settings.llm_prompt),
            ),
            AudioSignalEvaluator(),
            VisualSignalEvaluator(),
        ],
        stage,
        EvaluationMode.LLM,
        experimental,
    )


def load_reference_corpus(directory: Path | None) -> ReferenceRetriever | None:
    if directory is None or not directory.exists():
        return None
    from avd.embeddings.base import get_embedder

    corpus = LocalReferenceCorpus(directory, get_embedder())
    return corpus if corpus.passages else None
