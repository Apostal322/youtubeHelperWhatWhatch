"""Agreement, uncertainty and selective-prediction statistics (numpy only).

Reference values used in tests:
* ICC — Shrout & Fleiss (1979), Table 2 (6 targets x 4 judges): ICC(1,1)=.17, ICC(2,1)=.29, ICC(3,1)=.71.
* Krippendorff's alpha — Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability",
  example with 4 coders x 12 units: nominal .743, ordinal .815, interval .849.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


# ------------------------------------------------------------------------------- grouped bootstrap


def grouped_bootstrap(
    units: Sequence[T],
    group_of: Callable[[T], Hashable],
    statistic: Callable[[Sequence[T]], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile CI where whole groups (e.g. shows / creators) are resampled with replacement, so
    items from the same group are never treated as independent evidence."""
    point = statistic(units)
    groups: dict[Hashable, list[T]] = {}
    for u in units:
        groups.setdefault(group_of(u), []).append(u)
    keys = list(groups)
    out: dict[str, Any] = {"value": _f(point), "n_units": len(units), "n_groups": len(keys), "seed": seed}
    if len(keys) < 2 or point is None or (isinstance(point, float) and math.isnan(point)):
        return {**out, "ci_low": None, "ci_high": None, "n_boot": 0}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [u for i in pick for u in groups[keys[i]]]
        v = statistic(sample)
        if v is not None and not math.isnan(v):
            stats.append(v)
    if len(stats) < n_boot * 0.5:  # statistic undefined on most resamples (e.g. constant scores)
        return {**out, "ci_low": None, "ci_high": None, "n_boot": len(stats)}
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {**out, "ci_low": _f(lo), "ci_high": _f(hi), "n_boot": len(stats)}


def _f(x: Any, nd: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, nd)


# ------------------------------------------------------------------------------------- agreement


def weighted_kappa(
    a: Sequence[int],
    b: Sequence[int],
    categories: Sequence[int] = (1, 2, 3, 4, 5),
    weights: str = "quadratic",
) -> float:
    """Cohen's weighted kappa for two raters on an ordinal scale."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[idx[round(x)], idx[round(y)]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = ((i - j) / (k - 1)) ** 2 if weights == "quadratic" else np.abs(i - j) / (k - 1)
    den = (w * exp).sum()
    return float("nan") if den == 0 else float(1 - (w * obs).sum() / den)


def icc(ratings: np.ndarray, kind: str = "1,1") -> float:
    """Intraclass correlation (Shrout & Fleiss). ``ratings``: targets x raters.

    * ``"1,1"`` one-way random, single rater; NaNs allowed (unbalanced designs: k0 correction);
      appropriate when different raters rate different items (our annotation design).
    * ``"2,1"`` two-way random, absolute agreement; ``"3,1"`` two-way mixed, consistency — both
      need a complete matrix.
    """
    r = np.asarray(ratings, dtype=float)
    if kind == "1,1":
        rows = [row[~np.isnan(row)] for row in r]
        rows = [x for x in rows if len(x) >= 2]
        n = len(rows)
        if n < 2:
            return float("nan")
        big_n = sum(len(x) for x in rows)
        grand = np.concatenate(rows).mean()
        msb = sum(len(x) * (x.mean() - grand) ** 2 for x in rows) / (n - 1)
        msw = sum(((x - x.mean()) ** 2).sum() for x in rows) / (big_n - n)
        k0 = (big_n - sum(len(x) ** 2 for x in rows) / big_n) / (n - 1)
        den = msb + (k0 - 1) * msw
        return float("nan") if den == 0 else float((msb - msw) / den)
    if np.isnan(r).any():
        r = r[~np.isnan(r).any(axis=1)]
    n, k = r.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = r.mean()
    msr = k * ((r.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((r.mean(0) - grand) ** 2).sum() / (k - 1)
    sse = ((r - r.mean(1, keepdims=True) - r.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    if kind == "2,1":
        den = msr + (k - 1) * mse + k * (msc - mse) / n
    elif kind == "3,1":
        den = msr + (k - 1) * mse
    else:
        raise ValueError(f"unsupported ICC kind {kind}")
    return float("nan") if den == 0 else float((msr - mse) / den)


def krippendorff_alpha(ratings: np.ndarray, level: str = "interval") -> float:
    """Krippendorff's alpha. ``ratings``: raters x units, NaN = missing. ``level``: nominal,
    ordinal or interval. Units with fewer than two values are not pairable and are ignored."""
    r = np.asarray(ratings, dtype=float)
    values = np.unique(r[~np.isnan(r)])
    if len(values) < 2:
        return float("nan")
    vi = {v: i for i, v in enumerate(values)}
    k = len(values)
    coinc = np.zeros((k, k))
    for u in range(r.shape[1]):
        col = r[:, u][~np.isnan(r[:, u])]
        m = len(col)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    coinc[vi[col[a]], vi[col[b]]] += 1 / (m - 1)
    n_c = coinc.sum(1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    delta = np.zeros((k, k))
    for c in range(k):
        for e in range(k):
            if level == "nominal":
                delta[c, e] = 0.0 if c == e else 1.0
            elif level == "interval":
                delta[c, e] = (values[c] - values[e]) ** 2
            elif level == "ordinal":
                lo, hi = min(c, e), max(c, e)
                delta[c, e] = (n_c[lo : hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
            else:
                raise ValueError(level)
    d_o = (coinc * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    return float("nan") if d_e == 0 else float(1 - d_o / d_e)


# ------------------------------------------------------------------------------- pairwise & ranks


def pairwise_tie_aware(
    scores: dict[str, float | None], pairs: Sequence[tuple[str, str, str]], tie_margin: float = 0.25
) -> dict[str, Any]:
    """Pairs ``(a, b, preferred)`` with preferred in a | b | equal | insufficient.

    * ``strict_accuracy``: decided label pairs (a/b) only; a model tie (|diff| <= margin) scores 0.5.
    * ``tie_aware_accuracy``: also includes label ties, correct when the model is within the margin.
    * Pairs where the model abstained on either side are counted, not guessed.
    """
    strict = strict_n = tie_ok = tie_n = abstained = insufficient = 0.0
    for a, b, pref in pairs:
        if pref == "insufficient":
            insufficient += 1
            continue
        sa, sb = scores.get(a), scores.get(b)
        if sa is None or sb is None:
            abstained += 1
            continue
        diff = sa - sb
        if pref == "equal":
            tie_n += 1
            tie_ok += abs(diff) <= tie_margin
            continue
        strict_n += 1
        if abs(diff) <= tie_margin:
            strict += 0.5
        elif (diff > 0) == (pref == "a"):
            strict += 1
    total = strict_n + tie_n
    return {
        "strict_accuracy": strict / strict_n if strict_n else None,
        "tie_aware_accuracy": (strict + tie_ok) / total if total else None,
        "decided_pairs": int(strict_n),
        "label_ties": int(tie_n),
        "model_abstained": int(abstained),
        "label_insufficient": int(insufficient),
        "tie_margin": tie_margin,
    }


# ------------------------------------------------------------------------------ selective prediction


def risk_coverage(confidence: Sequence[float], loss: Sequence[float], steps: int = 10) -> dict[str, Any]:
    """Selective risk: keep the most confident fraction ``c`` of predictions, report their mean
    loss. AURC is the area under the risk-coverage curve (lower is better); a well-ranked
    confidence makes risk fall as coverage falls."""
    c = np.asarray(confidence, dtype=float)
    ls = np.asarray(loss, dtype=float)
    if len(c) == 0:
        return {"n": 0, "curve": [], "aurc": None, "risk_at_full_coverage": None}
    order = np.argsort(-c, kind="mergesort")
    cum = np.cumsum(ls[order]) / np.arange(1, len(ls) + 1)
    curve = []
    for i in range(1, steps + 1):
        cov = i / steps
        m = max(1, round(cov * len(c)))
        curve.append({"coverage": round(cov, 2), "risk": _f(cum[m - 1]), "n": m})
    return {
        "n": len(c),
        "curve": curve,
        "aurc": _f(cum.mean()),
        "risk_at_full_coverage": _f(cum[-1]),
        "oracle_aurc": _f((np.cumsum(np.sort(ls)) / np.arange(1, len(ls) + 1)).mean()),
    }


# ------------------------------------------------------------------------------------ classification


def macro_f1(truth: Sequence[str], pred: Sequence[str]) -> dict[str, Any]:
    labels = sorted(set(truth) | set(pred))
    conf = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truth, pred, strict=True):
        conf[t][p] += 1
    per = {}
    for lab in sorted(set(truth)):
        tp = conf[lab][lab]
        fp = sum(conf[t][lab] for t in labels if t != lab)
        fn = sum(conf[lab][p] for p in labels if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = {
            "precision": _f(prec),
            "recall": _f(rec),
            "f1": _f(2 * prec * rec / (prec + rec) if prec + rec else 0.0),
            "support": tp + fn,
        }
    f1s = [v["f1"] or 0.0 for v in per.values()]
    return {
        "macro_f1": _f(sum(f1s) / len(f1s)) if f1s else None,
        "accuracy": _f(sum(1 for t, p in zip(truth, pred, strict=True) if t == p) / len(truth))
        if truth
        else None,
        "per_class": per,
        "confusion": conf,
        "n": len(truth),
    }


def percentiles(xs: Sequence[float], qs: Sequence[float] = (50, 95)) -> dict[str, float | None]:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if len(a) == 0:
        return {f"p{int(q)}": None for q in qs}
    return {f"p{int(q)}": _f(np.percentile(a, q), 2) for q in qs}
