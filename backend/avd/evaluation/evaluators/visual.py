"""Visual signal evaluator: technical picture quality from sampled frames of authorized video.

Measured per sampled frame (grayscale, downscaled): sharpness (variance of the Laplacian),
exposure (mean luminance and the fraction of crushed or blown pixels) and contrast (luminance
standard deviation). Resolution comes from the container.

Only ``visual_quality`` is scored, and with low confidence: the thresholds are heuristics that
have not been calibrated against people. Static frames are reported, not penalised, because a
slide lecture is static by design. ``editing`` and ``visual_readability`` are about choices and
on-screen text, which pixel statistics cannot judge. They abstain (``method_cannot_judge``) until a
vision-language evaluator is configured (``FrameJudge`` protocol, not implemented here).

This evaluator never touches content or presentation scores.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Protocol

import numpy as np

from avd.content import media
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

METHOD = "visual_signal_v1"
N_FRAMES = 24


class FrameJudge(Protocol):
    """Extension point for a vision-language model judging editing and on-screen readability from
    sampled frames (plus the transcript). Not implemented: no authorized multimodal provider is
    configured in this repository."""

    def judge(
        self, frames: list[np.ndarray], timestamps: list[float], inp: EvaluationInput
    ) -> list[DimensionScore]: ...


def frame_metrics(frame: np.ndarray) -> dict[str, float]:
    f = frame.astype(np.float32)
    lap = f[1:-1, :-2] + f[1:-1, 2:] + f[:-2, 1:-1] + f[2:, 1:-1] - 4 * f[1:-1, 1:-1]
    return {
        "sharpness": float(lap.var()),
        "brightness": float(f.mean()),
        "contrast": float(f.std()),
        "dark_clip": float((f < 16).mean()),
        "bright_clip": float((f > 239).mean()),
    }


def summarize(frames: list[np.ndarray]) -> dict[str, float]:
    per = [frame_metrics(f) for f in frames]
    out = {k: float(np.median([p[k] for p in per])) for k in per[0]}
    diffs = [float(np.abs(a - b).mean()) for a, b in pairwise(frames)]
    out["static_fraction"] = float(np.mean([d < 1.0 for d in diffs])) if diffs else 0.0
    return out


def score_visual(m: dict[str, float], height: int | None) -> tuple[float, list[str]]:
    score, issues = 5.0, []
    if height is not None and height < 360:
        score -= 1.5
        issues.append(f"low resolution ({height}p)")
    elif height is not None and height < 720:
        score -= 0.5
        issues.append(f"standard resolution ({height}p)")
    if m["sharpness"] < 20:
        score -= 1.5
        issues.append("blurry (low edge energy)")
    elif m["sharpness"] < 60:
        score -= 0.5
        issues.append("slightly soft")
    if m["brightness"] < 40 or m["brightness"] > 215:
        score -= 1.0
        issues.append("under- or over-exposed")
    if m["dark_clip"] + m["bright_clip"] > 0.25:
        score -= 0.5
        issues.append("clipped shadows or highlights")
    if m["contrast"] < 15:
        score -= 0.5
        issues.append("very low contrast")
    return max(1.0, score), issues


class VisualSignalEvaluator:
    name = "visual_signal"
    version = METHOD
    synthetic = False
    requires = frozenset({Modality.VIDEO})
    dimensions: tuple[Dimension, ...] = (
        Dimension.VISUAL_QUALITY,
        Dimension.EDITING,
        Dimension.VISUAL_READABILITY,
    )

    def evaluate(self, inp: EvaluationInput, profile: GenreProfile) -> EvaluatorOutput:
        cannot = [
            DimensionScore.abstain(
                d,
                DimensionStatus.INSUFFICIENT_EVIDENCE,
                f"Pixel statistics cannot judge {d.value.replace('_', ' ')}; this needs a vision-language "
                "evaluator, which is not configured.",
                METHOD,
                reason_code=ReasonCode.METHOD_CANNOT_JUDGE,
            )
            for d in (Dimension.EDITING, Dimension.VISUAL_READABILITY)
        ]
        if inp.video_path is None or not media.available():
            return EvaluatorOutput(
                [
                    DimensionScore.abstain(
                        Dimension.VISUAL_QUALITY,
                        DimensionStatus.REQUIRES_MEDIA,
                        "No authorized video (or ffmpeg) available.",
                        METHOD,
                        missing_inputs=["video"],
                    ),
                    *cannot,
                ]
            )
        info = media.probe(Path(inp.video_path))
        frames = media.sample_frames(Path(inp.video_path), info, N_FRAMES)
        if len(frames) < 3:
            return EvaluatorOutput(
                [
                    DimensionScore.abstain(
                        Dimension.VISUAL_QUALITY,
                        DimensionStatus.INSUFFICIENT_EVIDENCE,
                        f"Only {len(frames)} frames could be decoded.",
                        METHOD,
                    ),
                    *cannot,
                ]
            )
        m = summarize(frames)
        score, issues = score_visual(m, info.height)
        summary = (
            f"{info.width}x{info.height}, sharpness {m['sharpness']:.0f}, brightness {m['brightness']:.0f}/255, "
            f"contrast {m['contrast']:.0f}, clipped {m['dark_clip'] + m['bright_clip']:.1%}, "
            f"static between samples {m['static_fraction']:.0%} ({len(frames)} frames)"
        )
        return EvaluatorOutput(
            [
                DimensionScore(
                    dimension=Dimension.VISUAL_QUALITY,
                    status=DimensionStatus.SCORED,
                    score=score,
                    confidence=0.35,
                    justification="Signal metrics: "
                    + summary
                    + (
                        ". Issues: " + "; ".join(issues) + "."
                        if issues
                        else ". No resolution, focus or exposure issues detected."
                    ),
                    evidence=[
                        EvidenceRef(
                            content_source_id=inp.video_source_id,
                            start_seconds=0.0,
                            end_seconds=round(info.duration_seconds, 2),
                            note=summary,
                            verified=True,
                        )
                    ],
                    method=METHOD,
                    model=None,
                ),
                *cannot,
            ],
            models=[METHOD],
            report={"frames": len(frames), **{k: round(v, 3) for k, v in m.items()}, "height": info.height},
        )
