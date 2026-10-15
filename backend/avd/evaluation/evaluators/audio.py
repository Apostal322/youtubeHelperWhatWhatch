"""Audio-quality evaluator using signal metrics on *authorized* audio (PCM WAV).

Metrics: clipping ratio, active speech level (90th percentile frame RMS, dBFS), noise floor
(10th percentile frame RMS) and the resulting SNR estimate. These capture level, noise and
distortion problems; they do not capture reverberation, intelligibility of accents, or music
mixing quality — confidence is capped accordingly.

Only runs when an authorized audio file is attached to the video; otherwise the pipeline marks
``audio_quality`` as ``requires_media``.
"""

from __future__ import annotations

import wave

import numpy as np

from avd.evaluation.evaluators.base import EvaluatorOutput
from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.profiles import GenreProfile
from avd.evaluation.schema import (
    Dimension,
    DimensionScore,
    DimensionStatus,
    EvidenceRef,
    Modality,
)

METHOD = "audio_signal_metrics_v1"
FRAME_SECONDS = 0.05


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        rate, channels = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def audio_metrics(samples: np.ndarray, rate: int) -> dict[str, float]:
    frame = max(1, int(rate * FRAME_SECONDS))
    n = len(samples) // frame
    if n < 10:
        raise ValueError("audio too short for analysis")
    frames = samples[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    db = 20 * np.log10(rms)
    speech = float(np.percentile(db, 90))
    floor = float(np.percentile(db, 10))
    return {
        "clipping_ratio": float(np.mean(np.abs(samples) >= 0.999)),
        "speech_level_dbfs": speech,
        "noise_floor_dbfs": floor,
        "snr_db": speech - floor,
        "duration_s": len(samples) / rate,
    }


def score_audio(m: dict[str, float]) -> tuple[float, list[str]]:
    score, issues = 5.0, []
    if m["clipping_ratio"] > 0.01:
        score -= 2.0
        issues.append(f"heavy clipping ({m['clipping_ratio']:.2%} of samples)")
    elif m["clipping_ratio"] > 0.001:
        score -= 1.0
        issues.append(f"some clipping ({m['clipping_ratio']:.2%} of samples)")
    if m["snr_db"] < 15:
        score -= 2.0
        issues.append(f"high background noise (SNR ~{m['snr_db']:.0f} dB)")
    elif m["snr_db"] < 25:
        score -= 1.0
        issues.append(f"audible background noise (SNR ~{m['snr_db']:.0f} dB)")
    if m["speech_level_dbfs"] < -35:
        score -= 1.0
        issues.append(f"low level ({m['speech_level_dbfs']:.0f} dBFS)")
    return max(1.0, score), issues


class AudioSignalEvaluator:
    name = "audio_signal"
    version = METHOD
    synthetic = False
    requires = frozenset({Modality.AUDIO})
    dimensions: tuple[Dimension, ...] = (Dimension.AUDIO_QUALITY,)

    def evaluate(self, inp: EvaluationInput, profile: GenreProfile) -> EvaluatorOutput:
        if inp.audio_path is None:
            return EvaluatorOutput(
                dimensions=[
                    DimensionScore.abstain(
                        Dimension.AUDIO_QUALITY,
                        DimensionStatus.REQUIRES_MEDIA,
                        "No authorized audio available.",
                        METHOD,
                        missing_inputs=["audio"],
                    )
                ]
            )
        samples, rate = read_wav_mono(inp.audio_path)
        m = audio_metrics(samples, rate)
        score, issues = score_audio(m)
        summary = (
            f"speech level {m['speech_level_dbfs']:.1f} dBFS, noise floor "
            f"{m['noise_floor_dbfs']:.1f} dBFS (SNR ~{m['snr_db']:.0f} dB), clipping "
            f"{m['clipping_ratio']:.3%}."
        )
        return EvaluatorOutput(
            dimensions=[
                DimensionScore(
                    dimension=Dimension.AUDIO_QUALITY,
                    status=DimensionStatus.SCORED,
                    score=score,
                    confidence=0.6,
                    justification=(
                        "Signal metrics: "
                        + summary
                        + (
                            " Issues: " + "; ".join(issues) + "."
                            if issues
                            else " No level, noise or clipping issues detected."
                        )
                    ),
                    evidence=[
                        EvidenceRef(
                            content_source_id=inp.audio_source_id,
                            start_seconds=0.0,
                            end_seconds=round(m["duration_s"], 2),
                            note=summary,
                            verified=True,
                        )
                    ],
                    method=METHOD,
                    model=None,
                )
            ],
            models=[METHOD],
        )
