"""Reproducible benchmark harness for the quality evaluator and the recommender.

Runs the *production* pipeline (same code path as the workers) on an isolated temporary database
seeded from fixture videos, then compares outputs with labels:

* per-dimension agreement (Spearman, Kendall tau-b, MAE) and abstention behaviour;
* pairwise ranking accuracy against pairwise judgements;
* inter-rater agreement of the labels themselves (Krippendorff's alpha, when >= 2 raters);
* calibration of confidence values (ECE, |model - label| <= 1 counted as correct);
* popularity invariance (inflating views/subscribers x1000 must not change any score);
* genre classification accuracy;
* recommendation relevance (nDCG, precision, recall), diversity, low-quality exposure and
  small-creator exposure for persona profiles;
* cost per evaluated video.

Outputs ``<out>/<run_id>.json`` and ``<out>/<run_id>.md`` with a reproducibility block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from avd.bench import metrics as M
from avd.config import LLMProviderName, Settings, get_settings
from avd.content.store import load_bundle
from avd.db.models import Base, Channel, Evaluation, User, UserPreferences, Video, VideoStats
from avd.db.session import make_engine
from avd.embeddings.base import get_embedder
from avd.evaluation.aggregation import aggregate
from avd.evaluation.inputs import build_evaluation_input
from avd.evaluation.profiles import get_profile, profile_from_dict
from avd.evaluation.schema import (
    FRAMEWORK_VERSION,
    GENRE_PARENT,
    Dimension,
    DimensionScore,
    DimensionStatus,
    Genre,
)
from avd.evaluation.service import genre_or_other
from avd.recsys.recommender import recommend
from avd.services import seed_demo
from avd.workers.context import build_context
from avd.workers.runner import run_until_idle

GENRE_CODES = {
    "edu": "education",
    "sci": "science",
    "tech": "technology",
    "game": "gaming",
    "ent": "entertainment",
    "doc": "documentary",
    "art": "art",
    "music": "music",
    "howto": "howto",
    "news": "news_commentary",
}


def _sha256_dir(path: Path, patterns: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for pat in patterns:
        for p in sorted(path.glob(pat)):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_benchmark(
    dataset_dir: Path, fixtures_dir: Path, evaluator: str = "heuristic", base_settings: Settings | None = None
) -> dict[str, Any]:
    base = base_settings or get_settings()
    labels = _load_jsonl(dataset_dir / "labels.jsonl")
    pairs = _load_jsonl(dataset_dir / "pairs.jsonl")
    personas = json.loads((dataset_dir / "personas.json").read_text())
    label_sources = sorted({r.get("label_source", "unknown") for r in labels})

    tmp = Path(tempfile.mkdtemp(prefix="avd-bench-"))
    overrides: dict[str, Any] = {
        "database_url": f"sqlite:///{tmp / 'bench.db'}",
        "fixtures_dir": fixtures_dir,
        "media_cache_dir": tmp / "media",
        "redis_url": None,
    }
    if evaluator == "heuristic":
        overrides["llm_provider"] = LLMProviderName.HEURISTIC
    settings = base.model_copy(update=overrides)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)

    with sf() as s:
        seed_demo(s, settings)
        s.commit()
    ctx = build_context(settings=settings, session_factory=sf)
    outcomes = run_until_idle(ctx)

    report: dict[str, Any] = {"job_outcomes": outcomes}
    with sf() as s:
        evals = {
            e.video_id: e for e in s.scalars(select(Evaluation).where(Evaluation.is_current.is_(True))).all()
        }
        videos = {v.id: v for v in s.scalars(select(Video)).all()}
        report["evaluation"] = _evaluation_metrics(evals, labels, pairs)
        report["genre_classification"] = _genre_accuracy(videos)
        report["popularity_invariance"] = _popularity_invariance(s, ctx, videos)
        report["recommendation"] = _recommendation_metrics(s, personas, labels)
        report["cost"] = {
            "total_usd": round(sum(e.cost_usd for e in evals.values()), 6),
            "per_evaluated_video_usd": round(sum(e.cost_usd for e in evals.values()) / max(1, len(evals)), 6),
            "evaluated_videos": len(evals),
        }
        report["coverage"] = _coverage(s, evals)
        mode = next(iter(evals.values())).mode if evals else "none"

    report["meta"] = {
        "run_at": datetime.now(UTC).isoformat(),
        "framework_version": FRAMEWORK_VERSION,
        "evaluator": evaluator,
        "evaluation_mode": mode,
        "llm_provider": settings.llm_provider.value,
        "models": [] if settings.demo_mode else [settings.llm_model_stage_b, settings.llm_model_stage_c],
        "embedding_model": get_embedder().model_id,
        "git_commit": _git_sha(),
        "dataset": str(dataset_dir.name),
        "dataset_sha256": _sha256_dir(dataset_dir, ("*.jsonl", "*.json")),
        "fixtures_sha256": _sha256_dir(fixtures_dir / "videos", ("*.json",)),
        "label_sources": label_sources,
        "python": platform.python_version(),
        "warnings": _warnings(label_sources, labels),
    }
    return report


def _warnings(label_sources: list[str], labels: list[dict[str, Any]]) -> list[str]:
    w = []
    if any("synthetic" in s for s in label_sources):
        w.append(
            "Labels are SYNTHETIC design-intent labels written by the fixture author, not human "
            "annotations. These numbers test the harness and catch regressions; they say nothing "
            "about real-world accuracy."
        )
    if len({r["rater_id"] for r in labels}) < 2:
        w.append("Single rater: inter-rater agreement cannot be estimated.")
    return w


def _evaluation_metrics(
    evals: dict[str, Evaluation], labels: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    raters: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    null_labels: set[tuple[str, str]] = set()
    for r in labels:
        key = (r["video_id"], r["dimension"])
        if r["score"] is None:
            null_labels.add(key)
        else:
            by_key[key].append(float(r["score"]))
            raters[r["rater_id"]][key] = float(r["score"])

    model: dict[tuple[str, str], tuple[str, float | None, float | None]] = {}
    for vid, ev in evals.items():
        for d in ev.dimensions:
            model[(vid, d.dimension)] = (d.status, d.score, d.confidence)

    per_dim: dict[str, Any] = {}
    conf_all, correct_all = [], []
    for dim in Dimension:
        xs, ys = [], []
        abstained = scored_without_label = 0
        for vid in evals:
            key = (vid, dim.value)
            st, sc, cf = model.get(key, ("missing", None, None))
            if key in by_key:
                if st == DimensionStatus.SCORED.value and sc is not None:
                    lab = float(np.mean(by_key[key]))
                    xs.append(sc)
                    ys.append(lab)
                    if cf is not None:
                        conf_all.append(cf)
                        correct_all.append(abs(sc - lab) <= 1.0)
                else:
                    abstained += 1
            elif key in null_labels and st == DimensionStatus.SCORED.value:
                scored_without_label += 1
        per_dim[dim.value] = {
            "n_compared": len(xs),
            "spearman": _r(M.spearman(xs, ys)),
            "kendall_tau_b": _r(M.kendall_tau_b(xs, ys)),
            "mae": _r(M.mae(xs, ys)),
            "abstained_where_label_exists": abstained,
            "scored_where_label_is_null": scored_without_label,
        }

    # Reference "overall" from labels via the same genre profile and aggregation.
    ref_overall: dict[str, float | None] = {}
    model_overall: dict[str, float | None] = {}
    for vid, ev in evals.items():
        dims = [
            DimensionScore(
                dimension=Dimension(d),
                status=DimensionStatus.SCORED,
                score=float(np.mean(v)),
                confidence=1.0,
                justification="label",
                method="label",
            )
            for (v_id, d), v in by_key.items()
            if v_id == vid
        ]
        # Same weights as the evaluation itself (blended for genre mixtures), so the comparison is
        # like for like.
        stored = (ev.aggregation or {}).get("profile")
        prof = profile_from_dict(stored) if stored else get_profile(genre_or_other(ev.genre))
        ref_overall[vid] = aggregate(dims, prof).overall
        model_overall[vid] = ev.overall_score
    both: list[tuple[float, float]] = []
    for v in evals:
        mo, ro = model_overall[v], ref_overall[v]
        if mo is not None and ro is not None:
            both.append((mo, ro))

    pair_sets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for p in pairs:
        pair_sets[p["criterion"]].append((p["video_a"], p["video_b"], p["preferred"]))
    pairwise = {}
    for crit, ps in pair_sets.items():
        if crit == "overall":
            scores = dict(model_overall)
        else:
            scores = {v: (s if st == "scored" else None) for (v, d), (st, s, _) in model.items() if d == crit}
        pairwise[crit] = M.pairwise_accuracy(scores, ps)
    all_decided = sum(v["decided"] for v in pairwise.values())
    all_correct = sum(v["accuracy"] * v["decided"] for v in pairwise.values() if v["decided"])

    rater_ids = sorted(raters)
    keys = sorted({k for r in raters.values() for k in r})
    alpha = (
        M.krippendorff_alpha_interval(np.array([[raters[r].get(k, np.nan) for k in keys] for r in rater_ids]))
        if len(rater_ids) >= 2
        else None
    )
    statuses: dict[str, int] = defaultdict(int)
    for ev in evals.values():
        statuses[ev.status] += 1
    return {
        "per_dimension": per_dim,
        "overall_vs_label_reference": {
            "n": len(both),
            "spearman": _r(M.spearman([a for a, _ in both], [b for _, b in both])),
            "mae": _r(M.mae([a for a, _ in both], [b for _, b in both])),
            "overall_abstentions": sum(1 for v in model_overall.values() if v is None),
        },
        "pairwise": {
            **pairwise,
            "_all": {
                "decided": all_decided,
                "accuracy": _r(all_correct / all_decided) if all_decided else None,
            },
        },
        "inter_rater": {
            "raters": len(rater_ids),
            "krippendorff_alpha_interval": _r(alpha) if alpha is not None else None,
        },
        "calibration": {
            "criterion": "|model - label| <= 1",
            **M.expected_calibration_error(conf_all, correct_all),
        },
        "evaluation_status_counts": dict(statuses),
    }


def _genre_accuracy(videos: dict[str, Video]) -> dict[str, Any]:
    rows = []
    for vid, v in videos.items():
        code = vid.split("_")[1] if vid.startswith("syn_") else None
        if code in GENRE_CODES:
            rows.append((vid, GENRE_CODES[code], v.genre))
    # v2 genres are refinements; compare at the v1 level the fixture labels use.
    rows = [(v, t, GENRE_PARENT.get(Genre(p), Genre(p)).value if p else p) for v, t, p in rows]
    correct = sum(1 for _, t, p in rows if t == p)
    return {
        "accuracy": _r(correct / len(rows)) if rows else None,
        "n": len(rows),
        "errors": [{"video_id": v, "expected": t, "predicted": p} for v, t, p in rows if t != p],
    }


def _popularity_invariance(s: Session, ctx: Any, videos: dict[str, Video]) -> dict[str, Any]:
    """Inflate every popularity signal x1000 and re-run the pipeline: nothing may change."""
    pipeline = ctx.pipeline("B")
    max_delta, fp_changes, checked = 0.0, 0, 0
    for vid, v in videos.items():
        bundle = load_bundle(s, vid)
        if bundle.transcript is None:
            continue
        genre = ctx.classifier.classify(
            v.title,
            v.description or "",
            list(v.tags or []),
            v.youtube_category_id,
            transcript_text=bundle.transcript.text,
        )
        before = build_evaluation_input(v, bundle, genre)
        r1 = pipeline.run(before)
        for st in s.scalars(select(VideoStats).where(VideoStats.video_id == vid)).all():
            st.view_count = (st.view_count or 1) * 1000
            st.like_count = (st.like_count or 1) * 1000
        ch = s.get(Channel, v.channel_id)
        if ch is not None:
            ch.subscriber_count = (ch.subscriber_count or 1) * 1000
        s.flush()
        after = build_evaluation_input(v, load_bundle(s, vid), genre)
        r2 = pipeline.run(after)
        fp_changes += int(before.fingerprint() != after.fingerprint())
        for d1, d2 in zip(r1.dimensions, r2.dimensions, strict=True):
            if d1.score is not None or d2.score is not None:
                max_delta = max(max_delta, abs((d1.score or 0) - (d2.score or 0)))
        checked += 1
    s.rollback()
    return {
        "videos_checked": checked,
        "fingerprint_changes": fp_changes,
        "max_score_delta": max_delta,
        "passed": fp_changes == 0 and max_delta == 0.0,
    }


def _recommendation_metrics(
    s: Session, personas: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> dict[str, Any]:
    emb = get_embedder()
    by_video: dict[str, list[float]] = defaultdict(list)
    for r in labels:
        if r["score"] is not None and r["dimension"] not in ("audio_quality",):
            by_video[r["video_id"]].append(float(r["score"]))
    low_quality = {v for v, xs in by_video.items() if np.mean(xs) <= 2.5}
    results: list[dict[str, Any]] = []
    exposures: dict[str, int] = defaultdict(int)
    for p in personas:
        user = User(token_hash=hashlib.sha256(p["persona_id"].encode()).hexdigest())
        user.preferences = UserPreferences(
            interests=p["interests"],
            genre_weights=p.get("genre_weights", {}),
            depth_preference=p.get("depth_preference", "any"),
            languages=p.get("languages", []),
        )
        s.add(user)
        s.flush()
        rec = recommend(s, user.id, emb, limit=10, log_impressions=False)
        ranked = [it["video"]["id"] for it in rec["items"]]
        rel = set(p["relevant_video_ids"])
        vecs = [s.get(Video, v).embedding for v in ranked]  # type: ignore[union-attr]
        for v in ranked[:5]:
            ch = s.get(Channel, s.get(Video, v).channel_id)  # type: ignore[union-attr]
            exposures[ch.size_bucket if ch else "unknown"] += 1
        results.append(
            {
                "persona": p["persona_id"],
                "ndcg@5": _r(M.ndcg_at_k(ranked, rel, 5)),
                "ndcg@10": _r(M.ndcg_at_k(ranked, rel, 10)),
                "precision@5": _r(M.precision_at_k(ranked, rel, 5)),
                "recall@10": _r(M.recall_at_k(ranked, rel, 10)),
                "intra_list_diversity@10": _r(M.intra_list_diversity([v for v in vecs if v is not None])),
                "low_quality_in_top5": sum(1 for v in ranked[:5] if v in low_quality),
                "top5": ranked[:5],
            }
        )
    s.rollback()
    keys = ["ndcg@5", "ndcg@10", "precision@5", "recall@10", "intra_list_diversity@10"]
    return {
        "per_persona": results,
        "mean": {k: _r(M.nanmean([r[k] for r in results])) for k in keys},
        "low_quality_in_top5_total": sum(r["low_quality_in_top5"] for r in results),
        "top5_exposure_by_channel_size": dict(exposures),
    }


def _coverage(s: Session, evals: dict[str, Evaluation]) -> dict[str, Any]:
    by_genre: dict[str, int] = defaultdict(int)
    by_size: dict[str, int] = defaultdict(int)
    by_lang: dict[str, int] = defaultdict(int)
    for vid in evals:
        v = s.get(Video, vid)
        assert v is not None
        ch = s.get(Channel, v.channel_id)
        by_genre[v.genre or "other"] += 1
        by_size[ch.size_bucket if ch else "unknown"] += 1
        by_lang[v.language or "unknown"] += 1
    return {
        "by_genre": dict(by_genre),
        "by_channel_size": dict(by_size),
        "by_language": dict(by_lang),
        "genres_without_evaluations": sorted(set(g.value for g in Genre) - set(by_genre) - {"other"}),
    }


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def render_markdown(r: dict[str, Any]) -> str:
    m, ev, rec = r["meta"], r["evaluation"], r["recommendation"]
    lines = [f"# Benchmark report — {m['dataset']} — {m['evaluator']}", ""]
    for w in m["warnings"]:
        lines.append(f"> **Warning:** {w}")
    lines += ["", "## Reproducibility", "", "| key | value |", "|---|---|"]
    for k in (
        "run_at",
        "framework_version",
        "evaluator",
        "evaluation_mode",
        "llm_provider",
        "models",
        "embedding_model",
        "git_commit",
        "dataset_sha256",
        "fixtures_sha256",
        "label_sources",
        "python",
    ):
        lines.append(f"| {k} | `{m[k]}` |")
    lines += [
        "",
        "## Per-dimension agreement with labels",
        "",
        "| dimension | n | Spearman | Kendall τb | MAE | abstained (label exists) | scored (label null) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d, x in ev["per_dimension"].items():
        lines.append(
            f"| {d} | {x['n_compared']} | {x['spearman']} | {x['kendall_tau_b']} | {x['mae']} | "
            f"{x['abstained_where_label_exists']} | {x['scored_where_label_is_null']} |"
        )
    o = ev["overall_vs_label_reference"]
    lines += [
        "",
        f"**Overall vs label reference** (labels aggregated with the same genre profiles): n={o['n']}, "
        f"Spearman={o['spearman']}, MAE={o['mae']}, overall abstentions={o['overall_abstentions']}",
        "",
        "## Pairwise ranking accuracy",
        "",
        "| criterion | accuracy | decided | abstained |",
        "|---|---:|---:|---:|",
    ]
    for c, x in ev["pairwise"].items():
        if c != "_all":
            lines.append(f"| {c} | {_r(x['accuracy'])} | {x['decided']} | {x['abstained']} |")
    lines.append(
        f"| **all** | {ev['pairwise']['_all']['accuracy']} | {ev['pairwise']['_all']['decided']} | |"
    )
    cal = ev["calibration"]
    lines += [
        "",
        "## Calibration",
        "",
        f"ECE ({cal['criterion']}): **{cal['ece']}**",
        "",
        "| confidence bin | n | mean confidence | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for b in cal["bins"]:
        lines.append(f"| {b['range']} | {b['n']} | {b['mean_confidence']} | {b['accuracy']} |")
    ir = ev["inter_rater"]
    lines += [
        "",
        f"**Inter-rater agreement:** raters={ir['raters']}, Krippendorff α (interval) = {ir['krippendorff_alpha_interval']}",
        "",
        f"**Evaluation status counts:** {ev['evaluation_status_counts']}",
        "",
    ]
    g = r["genre_classification"]
    lines += [
        "## Genre classification",
        "",
        f"accuracy {g['accuracy']} (n={g['n']}); errors: "
        + (
            ", ".join(f"{e['video_id']}→{e['predicted']} (expected {e['expected']})" for e in g["errors"])
            or "none"
        ),
        "",
    ]
    pi = r["popularity_invariance"]
    lines += [
        "## Popularity invariance",
        "",
        f"views/likes/subscribers ×1000 on {pi['videos_checked']} videos → "
        f"fingerprint changes: {pi['fingerprint_changes']}, max score delta: {pi['max_score_delta']} → "
        f"**{'PASSED' if pi['passed'] else 'FAILED'}**",
        "",
        "## Recommendation (persona profiles)",
        "",
        "| persona | nDCG@5 | nDCG@10 | P@5 | R@10 | ILD@10 | low-quality in top5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in rec["per_persona"]:
        lines.append(
            f"| {p['persona']} | {p['ndcg@5']} | {p['ndcg@10']} | {p['precision@5']} | {p['recall@10']} | "
            f"{p['intra_list_diversity@10']} | {p['low_quality_in_top5']} |"
        )
    mn = rec["mean"]
    lines.append(
        f"| **mean** | {mn['ndcg@5']} | {mn['ndcg@10']} | {mn['precision@5']} | {mn['recall@10']} | "
        f"{mn['intra_list_diversity@10']} | {rec['low_quality_in_top5_total']} |"
    )
    lines += [
        "",
        f"Top-5 exposure by channel size: {rec['top5_exposure_by_channel_size']}",
        "",
        "## Coverage and cost",
        "",
        f"Evaluated by genre: {r['coverage']['by_genre']}",
        "",
        f"Evaluated by channel size: {r['coverage']['by_channel_size']}",
        "",
        f"Evaluated by language: {r['coverage']['by_language']}",
        "",
        f"Cost: total ${r['cost']['total_usd']}, per video ${r['cost']['per_evaluated_video_usd']} "
        f"({r['cost']['evaluated_videos']} videos)",
        "",
    ]
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    report = run_benchmark(Path(args.dataset), Path(args.fixtures), evaluator=args.evaluator)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_id = args.tag or f"{report['meta']['dataset']}-{args.evaluator}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    (out / f"{run_id}.json").write_text(json.dumps(report, indent=2, default=str))
    (out / f"{run_id}.md").write_text(render_markdown(report))
    print(render_markdown(report))
    print(f"\nwrote {out / run_id}.json and .md")
