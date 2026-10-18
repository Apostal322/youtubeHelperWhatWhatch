"""Rankers.

The default ``LinearRanker`` is a transparent weighted sum over named features; every
recommendation exposes each feature's contribution. ``train_pairwise_ranker`` learns the same
kind of linear model from logged impressions + feedback (pairwise logistic loss), which is the
first Learning-to-Rank step: models are swappable through the ``Ranker`` protocol and record
their version on every impression for offline evaluation.

Note: popularity is not a ranking feature. ``small_creator`` is a user opt-in exposure boost
(default weight 0) and never touches intrinsic quality.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

FEATURES = (
    "relevance",
    "quality",
    "confidence",
    "genre_preference",
    "depth_match",
    "novelty",
    "feedback_affinity",
    "small_creator",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "relevance": 0.40,
    "quality": 0.30,
    "confidence": 0.05,
    "genre_preference": 0.10,
    "depth_match": 0.05,
    "novelty": 0.05,
    "feedback_affinity": 0.05,
    "small_creator": 0.0,
}


class Ranker(Protocol):
    version: str

    def score(
        self, features: dict[str, float], overrides: dict[str, float] | None = None
    ) -> tuple[float, dict[str, float]]: ...


@dataclass
class LinearRanker:
    weights: dict[str, float]
    version: str = "linear_v1"

    def score(
        self, features: dict[str, float], overrides: dict[str, float] | None = None
    ) -> tuple[float, dict[str, float]]:
        w = {**self.weights, **(overrides or {})}
        contrib = {k: round(w.get(k, 0.0) * features.get(k, 0.0), 6) for k in FEATURES}
        return round(sum(contrib.values()), 6), contrib

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"version": self.version, "weights": self.weights}, indent=2))

    @classmethod
    def load(cls, path: Path) -> LinearRanker:
        data = json.loads(path.read_text())
        return cls(weights=data["weights"], version=data["version"])


def train_pairwise_ranker(
    pairs: Sequence[tuple[dict[str, float], dict[str, float]]],
    *,
    epochs: int = 300,
    lr: float = 0.5,
    l2: float = 0.01,
    seed: int = 0,
) -> LinearRanker:
    """Pairwise logistic regression (RankNet-style with a linear scorer).

    ``pairs`` = [(preferred_features, other_features)]. Weights are L2-regularised towards zero
    and normalised to sum of absolute values 1 to stay on the same scale as the default ranker.
    """
    if not pairs:
        raise ValueError("no training pairs")
    x = np.array([[p[f] - q[f] for f in FEATURES] for p, q in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, len(FEATURES))
    for _ in range(epochs):
        z = x @ w
        grad = -(x * (1 / (1 + np.exp(z)))[:, None]).mean(axis=0) + l2 * w
        w -= lr * grad
    norm = float(np.abs(w).sum()) or 1.0
    weights = {f: round(float(v / norm), 6) for f, v in zip(FEATURES, w, strict=True)}
    digest = hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:8]
    return LinearRanker(weights=weights, version=f"ltr_pairwise_v1:{digest}")


def pairwise_accuracy(ranker: Ranker, pairs: Sequence[tuple[dict[str, float], dict[str, float]]]) -> float:
    if not pairs:
        return float("nan")
    ok = sum(1 for a, b in pairs if ranker.score(a)[0] > ranker.score(b)[0])
    return ok / len(pairs)
