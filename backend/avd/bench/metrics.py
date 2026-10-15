"""Evaluation metrics (numpy-only, no SciPy dependency)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np


def _rank(x: Sequence[float]) -> np.ndarray:
    """Average ranks with ties (1-based)."""
    a = np.asarray(x, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float("nan") if denom == 0 else (conc - disc) / denom


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(x) - np.asarray(y)))) if len(x) else float("nan")


def pairwise_accuracy(
    scores: dict[str, float | None], pairs: Sequence[tuple[str, str, str]]
) -> dict[str, float]:
    """pairs: (a, b, preferred in {'a','b','tie'}). Ties in the labels are excluded; pairs where
    the model abstained on either side count towards ``abstained``; model ties count as 0.5."""
    correct = 0.0
    decided = abstained = 0
    for a, b, pref in pairs:
        if pref == "tie":
            continue
        sa, sb = scores.get(a), scores.get(b)
        if sa is None or sb is None:
            abstained += 1
            continue
        decided += 1
        if sa == sb:
            correct += 0.5
        elif (sa > sb) == (pref == "a"):
            correct += 1
    total = decided + abstained
    return {
        "accuracy": correct / decided if decided else float("nan"),
        "decided": decided,
        "abstained": abstained,
        "coverage": decided / total if total else float("nan"),
    }


def krippendorff_alpha_interval(ratings: np.ndarray) -> float:
    """Krippendorff's alpha (interval metric). ``ratings``: raters x units, NaN = missing."""
    r = np.asarray(ratings, dtype=float)
    units = [r[:, u][~np.isnan(r[:, u])] for u in range(r.shape[1])]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    n = sum(len(u) for u in units)
    do = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in units) / n
    allv = np.concatenate(units)
    de = ((allv[:, None] - allv[None, :]) ** 2).sum() / (n * (n - 1))
    return float("nan") if de == 0 else 1.0 - do / de


def expected_calibration_error(
    conf: Sequence[float], correct: Sequence[bool], bins: int = 5
) -> dict[str, object]:
    c, k = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if len(c) == 0:
        return {"ece": float("nan"), "bins": []}
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in pairwise(edges):
        m = (c >= lo) & ((c < hi) | (hi == 1.0))
        if not m.any():
            continue
        gap = abs(c[m].mean() - k[m].mean())
        ece += m.mean() * gap
        table.append(
            {
                "range": [round(float(lo), 2), round(float(hi), 2)],
                "n": int(m.sum()),
                "mean_confidence": round(float(c[m].mean()), 3),
                "accuracy": round(float(k[m].mean()), 3),
            }
        )
    return {"ece": round(float(ece), 4), "bins": table}


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, v in enumerate(ranked[:k]) if v in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else float("nan")


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(1 for v in top if v in relevant) / k if top else float("nan")


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for v in ranked[:k] if v in relevant) / len(relevant) if relevant else float("nan")


def intra_list_diversity(vectors: Sequence[Sequence[float]]) -> float:
    if len(vectors) < 2:
        return float("nan")
    m = np.asarray(vectors, dtype=float)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    sims = m @ m.T
    iu = np.triu_indices(len(m), 1)
    return float(1 - sims[iu].mean())


def nanmean(xs: Sequence[float]) -> float:
    a = np.asarray([x for x in xs if x is not None and not math.isnan(x)], dtype=float)
    return float(a.mean()) if len(a) else float("nan")
