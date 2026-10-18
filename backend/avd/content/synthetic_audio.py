"""Deterministic synthetic 'speech-like' audio for demo fixtures and tests.

The signal is a harmonic tone with a syllable-rate amplitude envelope and pauses — not real
speech. It exists only to exercise the audio-quality evaluator end-to-end without any media
downloads. Profiles: ``clean`` and ``noisy_clipped``.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def generate(profile: str, seconds: float = 20.0, rate: int = 16000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 + 15 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 6))
    envelope = np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 2
    pauses = (np.sin(2 * np.pi * 0.25 * t) > -0.6).astype(float)
    signal = 0.25 * voice * envelope * pauses
    if profile == "clean":
        signal = signal + 0.0005 * rng.standard_normal(len(t))
    elif profile == "noisy_clipped":
        signal = 4.0 * signal + 0.08 * rng.standard_normal(len(t))
        signal = np.clip(signal, -1.0, 1.0)
    else:
        raise ValueError(f"unknown audio profile {profile!r}")
    return np.asarray(signal, dtype=np.float64)


def write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path
