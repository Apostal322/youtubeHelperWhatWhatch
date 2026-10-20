"""Benchmark on a real, licensed dataset against human annotations.

Runs the production pipeline on an isolated database built from a dataset manifest, then reports:

* **label-free** measurements (always available): coverage and abstention by reason, evidence
  validity (invalid-citation rate), injection flags, genre distribution, latency and cost
  percentiles, tokens per video, popularity invariance;
* **human-grounded** measurements (only when exported human labels exist for the split):
  model-human and human-human agreement (Spearman, Kendall, MAE, quadratic-weighted kappa) with
  bootstrap CIs that resample whole groups (shows), Krippendorff's alpha, ICC(1,1), abstention
  agreement, calibration (ECE), selective risk (risk-coverage, AURC) and tie-aware pairwise
  accuracy. Human-human agreement is the reference the model is compared with.

Release gates (``benchmarks/release_gates.json``) are evaluated only on the **test** split, only
if the gates file still matches its lock (``release_gates.lock``) and the lock predates every
test-split label. When there are no human labels, the human-grounded metrics and gates are
reported as *not evaluable*; nothing is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from avd.bench import metrics as M
from avd.bench import stats as S
from avd.bench.harness import _git_sha, _popularity_invariance
from avd.config import REPO_ROOT, LLMProviderName, Settings, get_settings
from avd.datasets.fetch import SourceFetcher
from avd.datasets.importer import import_manifest
from avd.datasets.pilot import load_manifest
from avd.db.models import Base, CostLedger, DatasetItem, Evaluation, Video
from avd.db.session import make_engine
from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS
from avd.evaluation.pipeline import EvaluationPipeline
from avd.evaluation.service import assessment_state
from avd.services import enqueue_stage_a
from avd.workers.context import build_context
from avd.workers.runner import run_until_idle

GATES_PATH = REPO_ROOT / "benchmarks" / "release_gates.json"
LOCK_PATH = REPO_ROOT / "benchmarks" / "release_gates.lock"
DIMENSIONS = tuple(d.value for d in TEXT_DIMENSIONS)
CONTENT_DIMENSIONS = (
    "information_density",
    "depth",
    "factual_accuracy",
    "originality",
    "informational_value",
)


class SplitGuardError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- run


def run_real_benchmark(
    manifest_path: Path,
    labels_dir: Path | None,
    split: str = "dev",
    evaluator: str = "heuristic",
    base_settings: Settings | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
    seed: int = 0,
    n_boot: int = 2000,
    confirm_test: bool = False,
    gates_path: Path = GATES_PATH,
    lock_path: Path = LOCK_PATH,
) -> dict[str, Any]:
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test (calibration items are for rater training)")
    if split == "test" and not confirm_test:
        raise SplitGuardError(
            "the test split is held out: pass --confirm-test for a deliberate, logged test run"
        )
    gates = load_gates(gates_path, lock_path)
    labels, pairs = _load_labels(labels_dir, split)
    if split == "test":
        _check_lock_predates_labels(gates, labels, pairs)
    base = base_settings or get_settings()
    t0 = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="avd-bench-real-"))
    overrides: dict[str, Any] = {
        "database_url": f"sqlite:///{tmp / 'bench.db'}",
        "media_cache_dir": tmp / "media",
        "redis_url": None,
    }
    if evaluator == "heuristic":
        overrides["llm_provider"] = LLMProviderName.HEURISTIC
    settings = base.model_copy(update=overrides)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    manifest = load_manifest(manifest_path)
    fetcher = SourceFetcher(cache_dir or settings.dataset_cache_dir, offline=offline)
    with sf() as s:
        imp = import_manifest(s, manifest_path, fetcher)
        items = {
            i.video_id: i
            for i in s.scalars(select(DatasetItem).where(DatasetItem.dataset_id == imp["dataset"])).all()
            if i.split == split
        }
        for vid in sorted(items):
            enqueue_stage_a(s, vid, force=True)
        s.commit()
    ctx = build_context(settings=settings, session_factory=sf)
    outcomes = run_until_idle(ctx)
    runtime = time.monotonic() - t0

    with sf() as s:
        evals = {
            e.video_id: e
            for e in s.scalars(select(Evaluation).where(Evaluation.is_current.is_(True))).all()
            if e.video_id in items
        }
        model = _model_table(evals)
        item_scores = {
            vid: {
                "overall": ev.overall_score,
                "content": (ev.group_scores or {}).get("content", {}).get("score"),
                "presentation": (ev.group_scores or {}).get("presentation", {}).get("score"),
            }
            for vid, ev in evals.items()
        }
        ledger = [row for row in s.scalars(select(CostLedger)).all() if row.video_id in items]
        videos = {v.id: v for v in s.scalars(select(Video)).all() if v.id in items}
        label_free = _label_free(evals, items, ledger, videos)
        label_free["popularity_invariance"] = _popularity_invariance(s, ctx, videos)
        pipeline_b: EvaluationPipeline = ctx.pipeline("B")
        config = pipeline_b.config()
        config_hash = pipeline_b.config_hash()

    groups = {vid: it.group_key for vid, it in items.items()}
    human = _human_metrics(model, item_scores, labels, pairs, groups, seed=seed, n_boot=n_boot)
    report: dict[str, Any] = {
        "label_free": label_free,
        "human_grounded": human,
        "job_outcomes": outcomes,
    }
    report["gates"] = evaluate_gates(gates, report, split, evaluator, load_exposures(manifest_path))
    report["meta"] = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": f"{manifest['name']}@{manifest['version']}",
        "split": split,
        "items": len(items),
        "groups": sorted(set(groups.values())),
        "manifest_items_sha256": manifest["items_sha256"],
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "labels_sha256": _sha_dir(labels_dir) if labels_dir else None,
        "human_label_rows": len(labels),
        "human_pair_rows": len(pairs),
        "git_commit": _git_sha(),
        "git_dirty": _git_dirty(),
        "evaluator": evaluator,
        "llm_provider": settings.llm_provider.value,
        "pipeline_config": config,
        "pipeline_config_hash": config_hash,
        "settings": {
            "llm_model_stage_b": settings.llm_model_stage_b,
            "llm_effort_stage_b": settings.llm_effort_stage_b,
            "llm_temperature": settings.llm_temperature,
            "llm_max_output_tokens": settings.llm_max_output_tokens,
            "llm_model_revision": settings.llm_model_revision,
            "llm_single_pass_max_tokens": settings.llm_single_pass_max_tokens,
            "llm_chunk_target_tokens": settings.llm_chunk_target_tokens,
        },
        "seed": seed,
        "n_boot": n_boot,
        "runtime_seconds": round(runtime, 2),
        "gates_file_sha256": gates["sha256"],
        "gates_lock_valid": gates["lock_valid"],
        "python": platform.python_version(),
        "warnings": _warnings(evaluator, labels, split),
    }
    return report


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=REPO_ROOT
        )
    except Exception:
        return None
    return bool(out.strip())


def _warnings(evaluator: str, labels: list[dict[str, Any]], split: str) -> list[str]:
    w = []
    if evaluator == "heuristic":
        w.append(
            "Evaluator is the HEURISTIC baseline (keyword/structure features), not an AI model. "
            "No live LLM was used in this run."
        )
    if not labels:
        w.append(
            f"No human labels for the {split} split: model-human and human-human agreement are not "
            "evaluable. Nothing here measures accuracy against people."
        )
    return w


# ----------------------------------------------------------------------------------------- inputs


def _load_labels(labels_dir: Path | None, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if labels_dir is None:
        return [], []

    def read(name: str) -> list[dict[str, Any]]:
        p = labels_dir / name
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    labels = [r for r in read("labels.jsonl") if r.get("split") == split]
    bad = {r.get("label_source") for r in labels} - {"human_annotation", "human_adjudication"}
    if bad:
        raise ValueError(
            f"refusing non-human label sources in a human-grounded benchmark: {sorted(map(str, bad))}"
        )
    pairs = [
        r
        for r in read("pairs.jsonl")
        if r.get("split") == split and r.get("label_source") == "human_annotation"
    ]
    return labels, pairs


def _sha_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.glob("*.jsonl")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _model_table(evals: dict[str, Evaluation]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for vid, ev in evals.items():
        for d in ev.dimensions:
            out[(vid, d.dimension)] = {
                "status": d.status,
                "score": d.score,
                "confidence": d.confidence,
                "reason_code": d.reason_code,
            }
    return out


# ------------------------------------------------------------------------------------- label-free


def _label_free(
    evals: dict[str, Evaluation],
    items: dict[str, DatasetItem],
    ledger: list[CostLedger],
    videos: dict[str, Video],
) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    states: dict[str, int] = defaultdict(int)
    dim_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    claimed = verified = ungrounded = injection_findings = 0
    flagged_items = 0
    for ev in evals.values():
        status_counts[ev.status] += 1
        states[assessment_state(ev)[0]] += 1
        for d in ev.dimensions:
            if d.dimension in DIMENSIONS:
                dim_status[d.dimension][
                    d.status if d.status == "scored" else f"{d.status}:{d.reason_code}"
                ] += 1
        v = ev.validation or {}
        claimed += int(v.get("evidence_claimed", 0))
        verified += int(v.get("evidence_verified", 0))
        ungrounded += len(v.get("ungrounded_dimensions") or [])
        f = v.get("injection_findings") or []
        injection_findings += len(f)
        flagged_items += bool(f)
    n = len(evals)
    tokens_in: dict[str, int] = defaultdict(int)
    tokens_out: dict[str, int] = defaultdict(int)
    for row in ledger:
        if row.video_id:
            tokens_in[row.video_id] += row.input_tokens
            tokens_out[row.video_id] += row.output_tokens
    costs = [ev.cost_usd for ev in evals.values()]
    genre_counts: dict[str, int] = defaultdict(int)
    for video in videos.values():
        genre_counts[video.genre or "unclassified"] += 1
    return {
        "items_in_split": len(items),
        "items_evaluated": n,
        "items_not_evaluated": sorted(set(items) - set(evals)),
        "evaluation_status": dict(status_counts),
        "assessment_state": dict(states),
        "overall_score_available_rate": S._f(
            sum(1 for e in evals.values() if e.overall_score is not None) / n
        )
        if n
        else None,
        "dimension_status": {k: dict(v) for k, v in sorted(dim_status.items())},
        "evidence": {
            "citations_claimed": claimed,
            "citations_verified": verified,
            "invalid_citation_rate": S._f(1 - verified / claimed) if claimed else None,
            "ungrounded_scored_dimensions": ungrounded,
        },
        "injection": {
            "findings": injection_findings,
            "items_flagged": flagged_items,
            "items_flagged_rate": S._f(flagged_items / n) if n else None,
            "note": "Dataset items are presumed benign (not verified); flags here are likely false positives.",
        },
        "needs_review_rate": S._f(sum(1 for e in evals.values() if e.needs_review) / n) if n else None,
        "predicted_genre": dict(genre_counts),
        "latency_ms": S.percentiles([e.latency_ms for e in evals.values() if e.latency_ms is not None]),
        "cost_usd": {
            "total": round(sum(costs), 6),
            "per_video_mean": S._f(float(np.mean(costs)), 6) if costs else None,
            **{k: v for k, v in S.percentiles(costs).items()},
        },
        "tokens_per_video": {
            "input": S.percentiles([tokens_in.get(v, 0) for v in evals]),
            "output": S.percentiles([tokens_out.get(v, 0) for v in evals]),
        },
        "llm_calls": {
            "total": len(ledger),
            "errors": sum(1 for r in ledger if r.status == "error"),
            "cached": sum(1 for r in ledger if r.status == "cached" or r.cached),
        },
    }


# ---------------------------------------------------------------------------------- human-grounded


def _reference(
    labels: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, float]]]:
    """Per (video, dimension): the reference label and the individual independent ratings."""
    ratings: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    statuses: dict[tuple[str, str], list[str]] = defaultdict(list)
    adjudicated: dict[tuple[str, str], dict[str, Any]] = {}
    for r in labels:
        key = (r["video_id"], r["dimension"])
        if r["label_source"] == "human_adjudication":
            adjudicated[key] = r
            continue
        statuses[key].append(r["status"])
        if r["status"] == "scored" and r.get("score") is not None:
            ratings[key][r["rater_id"]] = float(r["score"])
    ref: dict[tuple[str, str], dict[str, Any]] = {}
    for key in set(statuses) | set(adjudicated):
        if key in adjudicated:
            a = adjudicated[key]
            ref[key] = {
                "status": a["status"],
                "score": a.get("score"),
                "source": "adjudicated",
                "n": len(statuses.get(key, [])),
            }
            continue
        sts = statuses[key]
        scored = list(ratings[key].values())
        if len(scored) * 2 > len(sts) and scored:
            spread = max(scored) - min(scored)
            ref[key] = {
                "status": "scored" if spread < 2 else "disputed",
                "score": float(np.mean(scored)) if spread < 2 else None,
                "source": "mean_of_raters",
                "n": len(sts),
            }
        else:
            ref[key] = {"status": "not_scored", "score": None, "source": "majority_not_scored", "n": len(sts)}
    return ref, ratings


def _human_metrics(
    model: dict[tuple[str, str], dict[str, Any]],
    item_scores: dict[str, dict[str, float | None]],
    labels: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    groups: dict[str, str],
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    if not labels:
        return {"available": False, "reason": "no human labels for this split"}
    ref, ratings = _reference(labels)
    g = lambda u: groups.get(u["video"], u["video"])  # noqa: E731

    def boot(units: list[dict[str, Any]], fn: Callable[[Sequence[dict[str, Any]]], float]) -> dict[str, Any]:
        return S.grouped_bootstrap(units, g, fn, n_boot=n_boot, seed=seed)

    def spear(us: Sequence[dict[str, Any]]) -> float:
        return M.spearman([u["x"] for u in us], [u["y"] for u in us])

    def mae(us: Sequence[dict[str, Any]]) -> float:
        return M.mae([u["x"] for u in us], [u["y"] for u in us])

    def qwk(us: Sequence[dict[str, Any]]) -> float:
        return S.weighted_kappa([round(u["x"]) for u in us], [round(u["y"]) for u in us])

    # Model vs reference (cells with >= 2 independent ratings or an adjudication).
    mh_units: list[dict[str, Any]] = []
    abst = {
        "human_scored_model_scored": 0,
        "human_scored_model_abstained": 0,
        "human_not_scored_model_abstained": 0,
        "human_not_scored_model_scored": 0,
    }
    for (vid, dim), r in ref.items():
        if dim not in DIMENSIONS or vid not in item_scores or (r["n"] < 2 and r["source"] != "adjudicated"):
            continue
        m = model.get((vid, dim), {"status": "missing", "score": None})
        model_scored = m["status"] == "scored" and m["score"] is not None
        if r["status"] == "scored":
            abst["human_scored_model_scored" if model_scored else "human_scored_model_abstained"] += 1
            if model_scored:
                mh_units.append(
                    {
                        "video": vid,
                        "dim": dim,
                        "x": float(m["score"]),
                        "y": float(r["score"]),
                        "conf": m.get("confidence"),
                    }
                )
        elif r["status"] == "not_scored":
            abst["human_not_scored_model_scored" if model_scored else "human_not_scored_model_abstained"] += 1
    per_dim = {}
    for dim in DIMENSIONS:
        us = [u for u in mh_units if u["dim"] == dim]
        per_dim[dim] = {
            "n": len(us),
            "spearman": boot(us, spear),
            "mae": boot(us, mae),
            "qwk": boot(us, qwk),
            "kendall_tau_b": S._f(M.kendall_tau_b([u["x"] for u in us], [u["y"] for u in us])),
        }

    # Leave-one-rater-out: each rater vs the mean of the others, and the model vs the same mean.
    loo_h: list[dict[str, Any]] = []
    loo_m: list[dict[str, Any]] = []
    for (vid, dim), rs in ratings.items():
        if dim not in DIMENSIONS or len(rs) < 2:
            continue
        m = model.get((vid, dim), {})
        for rid, val in rs.items():
            others = [v for k, v in rs.items() if k != rid]
            y = float(np.mean(others))
            loo_h.append({"video": vid, "dim": dim, "x": val, "y": y})
            if m.get("status") == "scored" and m.get("score") is not None:
                loo_m.append({"video": vid, "dim": dim, "x": float(m["score"]), "y": y, "human": val})
    paired = loo_m

    def mae_gap(us: Sequence[dict[str, Any]]) -> float:  # model MAE - human MAE on the same references
        if not us:
            return float("nan")
        return float(
            np.mean([abs(u["x"] - u["y"]) for u in us]) - np.mean([abs(u["human"] - u["y"]) for u in us])
        )

    rater_ids = sorted({rid for rs in ratings.values() for rid in rs})
    cells = sorted(k for k in ratings if k[0] in groups and k[1] in DIMENSIONS)
    matrix = (
        np.array([[ratings[c].get(r, np.nan) for c in cells] for r in rater_ids])
        if cells
        else np.zeros((0, 0))
    )
    icc_per_dim = {}
    for dim in DIMENSIONS:
        rows = [list(ratings[c].values()) for c in cells if c[1] == dim]
        width = max((len(x) for x in rows), default=0)
        mat = np.array([x + [np.nan] * (width - len(x)) for x in rows]) if rows else np.zeros((0, 0))
        icc_per_dim[dim] = S._f(S.icc(mat, "1,1")) if len(rows) >= 2 else None

    conf_units = [u for u in mh_units if u["conf"] is not None]
    losses = [abs(u["x"] - u["y"]) for u in conf_units]
    pair_units: list[dict[str, Any]] = []
    by_crit = {
        crit: {vid: sc[crit] for vid, sc in item_scores.items()}
        for crit in ("overall", "content", "presentation")
    }
    for p in pairs:
        pair_units.append(
            {
                "video": p["video_a"],
                "a": p["video_a"],
                "b": p["video_b"],
                "pref": p["preferred"],
                "crit": p["criterion"],
            }
        )

    def pair_acc(us: Sequence[dict[str, Any]]) -> float:
        accs = []
        for crit, scores in by_crit.items():
            sub = [(u["a"], u["b"], u["pref"]) for u in us if u["crit"] == crit]
            if sub:
                r_ = S.pairwise_tie_aware(scores, sub)
                if r_["tie_aware_accuracy"] is not None:
                    accs.append((r_["tie_aware_accuracy"], r_["decided_pairs"] + r_["label_ties"]))
        tot = sum(n for _, n in accs)
        return sum(a * n for a, n in accs) / tot if tot else float("nan")

    return {
        "available": True,
        "reference_rule": "adjudicated score, else mean of >=2 independent ratings when their range < 2; disputed cells excluded",
        "cells_with_reference": sum(1 for r in ref.values() if r["status"] == "scored"),
        "cells_disputed_unadjudicated": sum(1 for r in ref.values() if r["status"] == "disputed"),
        "model_vs_human": {
            "pooled": {
                "n": len(mh_units),
                "spearman": boot(mh_units, spear),
                "mae": boot(mh_units, mae),
                "qwk": boot(mh_units, qwk),
            },
            "content_dimensions": {
                "n": sum(1 for u in mh_units if u["dim"] in CONTENT_DIMENSIONS),
                "spearman": boot([u for u in mh_units if u["dim"] in CONTENT_DIMENSIONS], spear),
                "mae": boot([u for u in mh_units if u["dim"] in CONTENT_DIMENSIONS], mae),
            },
            "per_dimension": per_dim,
            "abstention_agreement": {
                **abst,
                "model_scored_when_humans_did_not_rate": S._f(
                    abst["human_not_scored_model_scored"]
                    / (abst["human_not_scored_model_scored"] + abst["human_not_scored_model_abstained"])
                )
                if (abst["human_not_scored_model_scored"] + abst["human_not_scored_model_abstained"])
                else None,
            },
        },
        "human_vs_human": {
            "leave_one_rater_out": {"n": len(loo_h), "spearman": boot(loo_h, spear), "mae": boot(loo_h, mae)},
            "raters": len(rater_ids),
            "krippendorff_alpha_ordinal": S._f(S.krippendorff_alpha(matrix, "ordinal"))
            if matrix.size
            else None,
            "krippendorff_alpha_interval": S._f(S.krippendorff_alpha(matrix, "interval"))
            if matrix.size
            else None,
            "icc_1_1_per_dimension": icc_per_dim,
        },
        "model_minus_human_mae_same_reference": boot(paired, mae_gap),
        "calibration": {
            "criterion": "|model - reference| <= 1",
            **M.expected_calibration_error([u["conf"] for u in conf_units], [lo <= 1 for lo in losses]),
        },
        "selective_risk": _with_gain(S.risk_coverage([u["conf"] for u in conf_units], losses)),
        "pairwise": {"n_pairs": len(pair_units), "tie_aware_accuracy": boot(pair_units, pair_acc)},
    }


def _with_gain(rc: dict[str, Any]) -> dict[str, Any]:
    """``aurc_gain`` > 0 means confidence ranks errors usefully (selective risk below full risk)."""
    if rc.get("aurc") is None or rc.get("risk_at_full_coverage") is None:
        return {**rc, "aurc_gain": None}
    return {**rc, "aurc_gain": S._f(rc["risk_at_full_coverage"] - rc["aurc"])}


# ----------------------------------------------------------------------------------------- gates


def load_gates(gates_path: Path = GATES_PATH, lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    raw = gates_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    return {**json.loads(raw), "sha256": sha, "lock": lock, "lock_valid": lock.get("sha256") == sha}


def lock_gates(
    statement: str, gates_path: Path = GATES_PATH, lock_path: Path = LOCK_PATH, new_test_set: bool = False
) -> dict[str, Any]:
    """Record the gates file hash. Re-locking changed gates is only legitimate with a new,
    untouched test set (``new_test_set=True``), and the old lock is kept in the history."""
    sha = hashlib.sha256(gates_path.read_bytes()).hexdigest()
    old = json.loads(lock_path.read_text()) if lock_path.exists() else None
    if old and old["sha256"] != sha and not new_test_set:
        raise SplitGuardError("gates changed after locking; re-lock only together with a new test set")
    if old and old["sha256"] == sha:
        return dict(old)
    lock = {
        "sha256": sha,
        "locked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "statement": statement,
        "history": (
            [*old.get("history", []), {k: old[k] for k in ("sha256", "locked_at", "statement")}]
            if old
            else []
        ),
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def _check_lock_predates_labels(
    gates: dict[str, Any], labels: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> None:
    if not gates["lock_valid"]:
        raise SplitGuardError(
            "release gates changed after locking (sha mismatch): re-locking requires a new test set"
        )
    locked_at = datetime.fromisoformat(gates["lock"]["locked_at"])
    times = [datetime.fromisoformat(r["submitted_at"]) for r in labels + pairs if r.get("submitted_at")]
    early = [t for t in times if t < locked_at]
    if early:
        raise SplitGuardError(
            f"{len(early)} test-split labels predate the gate lock ({locked_at.isoformat()}); "
            "gates must be fixed before any test label exists"
        )


def _get(report: dict[str, Any], path: str) -> Any:
    cur: Any = report
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# Gates whose metric does not depend on which evaluator is the candidate: human reliability is
# about people, and the injection detector runs before any evaluator.
EVALUATOR_INDEPENDENT_GATES = {"human_reliability_floor", "injection_benign_flag_rate"}
GATE_STATUSES = (
    "pass",
    "fail",
    "blocked_missing_labels",
    "blocked_no_inference",
    "blocked_insufficient_n",
    "not_applicable_exposed",
    "metric_unavailable",
)


def load_exposures(manifest_path: Path | None) -> dict[str, list[str]]:
    """gate id -> exposure notes, from the dataset's exposure.json (if any)."""
    if manifest_path is None or not (manifest_path.parent / "exposure.json").exists():
        return {}
    data = json.loads((manifest_path.parent / "exposure.json").read_text())
    out: dict[str, list[str]] = {}
    for e in data.get("exposures", []):
        for gid in e.get("affected_gates", []):
            out.setdefault(gid, []).append(f"{e['component']} ({e['commit']})")
    return out


def evaluate_gates(
    gates: dict[str, Any],
    report: dict[str, Any],
    split: str,
    evaluator: str = "configured",
    exposures: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Every gate gets exactly one status from ``GATE_STATUSES``. Only ``pass``/``fail`` are
    evaluations; everything else says why the gate could not be evaluated *for the candidate*.
    A heuristic-baseline run never passes a candidate gate: its value is kept as ``baseline_value``."""
    results = []
    exposures = exposures or {}
    for gate in gates["gates"]:
        value = _get(report, gate["metric"])
        if isinstance(value, dict):  # bootstrap result: use the requested bound
            value = value.get(gate.get("use", "value"))
        n = _get(report, gate["n_metric"]) if gate.get("n_metric") else None
        human = gate["metric"].startswith("human_grounded.")
        missing = value is None or (isinstance(value, float) and math.isnan(value))
        baseline: Any = None
        if human and not _get(report, "human_grounded.available"):
            status, why = "blocked_missing_labels", "no human labels"
        elif split == "test" and gate["id"] in exposures:
            status, why = "not_applicable_exposed", "test data informed " + "; ".join(exposures[gate["id"]])
        elif evaluator != "configured" and gate["id"] not in EVALUATOR_INDEPENDENT_GATES:
            status, why = "blocked_no_inference", f"run used the {evaluator} evaluator, not the AI candidate"
            baseline = None if missing else value
        elif gate.get("min_n") and (n is None or n < gate["min_n"]):
            status, why = "blocked_insufficient_n", f"n={n} < min_n={gate['min_n']}"
        elif missing:
            status, why = "metric_unavailable", "metric unavailable"
        else:
            ok = {
                "<=": value <= gate["threshold"],
                ">=": value >= gate["threshold"],
                "==": value == gate["threshold"],
            }[gate["op"]]
            status, why = ("pass" if ok else "fail"), f"{value} {gate['op']} {gate['threshold']}"
        row = {
            "id": gate["id"],
            "status": status,
            "evaluable": status in ("pass", "fail"),
            "detail": why,
            "kind": gate["kind"],
        }
        if baseline is not None:
            row["baseline_value"] = baseline
        results.append(row)
    binding = split == "test" and gates["lock_valid"]
    counts = {st: sum(1 for r in results if r["status"] == st) for st in GATE_STATUSES}
    return {
        "binding": binding,
        "note": "binding only on a test-split run with a valid lock"
        if not binding
        else "binding test-split evaluation",
        "evaluator": evaluator,
        "counts": {k: v for k, v in counts.items() if v},
        "results": results,
        "release_ready": binding and all(r["status"] == "pass" for r in results),
    }


# ---------------------------------------------------------------------------------------- render


def _ci(x: dict[str, Any] | None) -> str:
    if not x or x.get("value") is None:
        return "—"
    if x.get("ci_low") is None:
        return f"{x['value']} (no CI)"
    return f"{x['value']} [{x['ci_low']}, {x['ci_high']}]"


def render_markdown(r: dict[str, Any]) -> str:
    m, lf, hg = r["meta"], r["label_free"], r["human_grounded"]
    out = [f"# Real-data benchmark — {m['dataset']} ({m['split']} split) — {m['evaluator']}", ""]
    out += [f"> **Warning:** {w}" for w in m["warnings"]]
    out += ["", "## Reproducibility", "", "| key | value |", "|---|---|"]
    for k in (
        "run_at",
        "git_commit",
        "git_dirty",
        "dataset",
        "split",
        "items",
        "manifest_items_sha256",
        "labels_sha256",
        "human_label_rows",
        "evaluator",
        "llm_provider",
        "pipeline_config_hash",
        "seed",
        "n_boot",
        "runtime_seconds",
        "gates_file_sha256",
        "gates_lock_valid",
        "python",
    ):
        out.append(f"| {k} | `{m[k]}` |")
    out.append(f"| versions | `{m['pipeline_config']['versions']}` |")
    out += [
        "",
        "## Label-free measurements",
        "",
        f"* Items evaluated: {lf['items_evaluated']}/{lf['items_in_split']}; status {lf['evaluation_status']}; assessment state {lf['assessment_state']}",
        f"* Overall score available: {lf['overall_score_available_rate']}; needs review: {lf['needs_review_rate']}",
        f"* Evidence: {lf['evidence']}",
        f"* Injection flags (presumed-benign data): {lf['injection']['items_flagged']} items, {lf['injection']['findings']} findings",
        f"* Predicted genre: {lf['predicted_genre']}",
        f"* Latency (ms): {lf['latency_ms']}; cost (USD): {lf['cost_usd']}; tokens/video: {lf['tokens_per_video']}; LLM calls: {lf['llm_calls']}",
        f"* Popularity invariance: {'PASSED' if lf['popularity_invariance']['passed'] else 'FAILED'} ({lf['popularity_invariance']})",
        "",
        "### Dimension status (scored / abstained:reason)",
        "",
    ]
    for d, c in lf["dimension_status"].items():
        out.append(f"* {d}: {c}")
    out += ["", "## Human-grounded measurements", ""]
    if not hg["available"]:
        out.append(f"**Not available:** {hg['reason']}. No model–human or human–human statistic is reported.")
    else:
        mv, hh = hg["model_vs_human"], hg["human_vs_human"]
        out += [
            f"Reference: {hg['reference_rule']}. Cells with reference: {hg['cells_with_reference']}; disputed: {hg['cells_disputed_unadjudicated']}. CIs: 95% percentile bootstrap resampling whole groups.",
            "",
            "| comparison | n | Spearman | MAE | QWK |",
            "|---|---:|---|---|---|",
            f"| model vs human (pooled) | {mv['pooled']['n']} | {_ci(mv['pooled']['spearman'])} | {_ci(mv['pooled']['mae'])} | {_ci(mv['pooled']['qwk'])} |",
            f"| human vs other humans (LOO) | {hh['leave_one_rater_out']['n']} | {_ci(hh['leave_one_rater_out']['spearman'])} | {_ci(hh['leave_one_rater_out']['mae'])} | |",
            "",
            f"Model MAE − human MAE (same references): {_ci(hg['model_minus_human_mae_same_reference'])}",
            f"Krippendorff α ordinal {hh['krippendorff_alpha_ordinal']}, interval {hh['krippendorff_alpha_interval']}; ICC(1,1) by dimension {hh['icc_1_1_per_dimension']}",
            f"Abstention agreement: {mv['abstention_agreement']}",
            f"Calibration ECE: {hg['calibration']['ece']}; selective risk AURC {hg['selective_risk']['aurc']} (oracle {hg['selective_risk']['oracle_aurc']}, full coverage {hg['selective_risk']['risk_at_full_coverage']})",
            f"Pairwise tie-aware accuracy: {_ci(hg['pairwise']['tie_aware_accuracy'])} over {hg['pairwise']['n_pairs']} human pairs",
            "",
            "| dimension | n | Spearman | MAE | QWK |",
            "|---|---:|---|---|---|",
        ]
        for d, x in mv["per_dimension"].items():
            out.append(f"| {d} | {x['n']} | {_ci(x['spearman'])} | {_ci(x['mae'])} | {_ci(x['qwk'])} |")
    g = r["gates"]
    out += [
        "",
        f"## Release gates ({'binding' if g['binding'] else 'informational'}; release ready: {g['release_ready']})",
        "",
        "| gate | kind | status | detail |",
        "|---|---|---|---|",
    ]
    for x in g["results"]:
        out.append(f"| {x['id']} | {x['kind']} | {x['status']} | {x['detail']} |")
    out.append("")
    return "\n".join(out)


def main(args: argparse.Namespace) -> None:
    report = run_real_benchmark(
        Path(args.manifest),
        Path(args.labels) if args.labels else None,
        split=args.split,
        evaluator=args.evaluator,
        offline=args.offline,
        seed=args.seed,
        n_boot=args.n_boot,
        confirm_test=args.confirm_test,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = (
        args.tag
        or f"{report['meta']['dataset'].replace('@', '-')}-{args.split}-{args.evaluator}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    (out / f"{run_id}.json").write_text(json.dumps(report, indent=2, default=str))
    (out / f"{run_id}.md").write_text(render_markdown(report))
    print(render_markdown(report))
    print(f"\nwrote {out / run_id}.json and .md")
