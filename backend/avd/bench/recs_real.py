"""Held-out offline recommendation evaluation on the real pilot (test split).

Only the test-split items are evaluated and indexed (they were never used to tune anything in
the recommender). Each persona has interests written for this experiment. An item counts as
relevant to a persona when it belongs to the matching publisher show. That is a *topical* proxy
label from publisher metadata, not a human relevance or quality judgement. No quality labels
exist, so this measures topical retrieval and list composition, not whether the recommended
items are good.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from avd.bench import metrics as M
from avd.bench.harness import _git_sha
from avd.config import REPO_ROOT, LLMProviderName, get_settings
from avd.datasets.fetch import SourceFetcher
from avd.datasets.importer import import_manifest
from avd.datasets.pilot import load_manifest
from avd.db.models import Base, Dataset, DatasetItem, User, UserPreferences
from avd.db.session import make_engine
from avd.embeddings.base import Embedder, HashingEmbedder, calibrated_relevance
from avd.recsys.recommender import recommend
from avd.services import enqueue_stage_a
from avd.workers.context import build_context
from avd.workers.runner import run_until_idle

# Interests authored for this experiment (not taken from the items); relevance = show membership.
PERSONAS: list[dict[str, Any]] = [
    {
        "id": "frontend_dev",
        "interests": ["building web user interfaces with JavaScript frameworks"],
        "group": "changelog:jsparty",
    },
    {
        "id": "sre",
        "interests": ["running and deploying infrastructure reliably in production"],
        "group": "changelog:shipit",
    },
    {
        "id": "pythonista",
        "interests": ["Python libraries, tooling and data science workflows"],
        "group": "talkpython",
    },
    {
        "id": "mind_curious",
        "interests": ["psychology of habits, emotions and the brain"],
        "group": "changelog:brainscience",
    },
    {
        "id": "oss_generalist",
        "interests": ["open source maintainers and the software industry"],
        "group": "changelog:podcast",
    },
]


def run(
    manifest_path: Path, embedders: dict[str, Embedder], k: int = 5, offline: bool = True
) -> dict[str, Any]:
    settings = get_settings()
    manifest = load_manifest(manifest_path)
    results: dict[str, Any] = {}
    for name, emb in embedders.items():
        tmp = Path(tempfile.mkdtemp(prefix="avd-recs-"))
        s_over = settings.model_copy(
            update={
                "database_url": f"sqlite:///{tmp / 'r.db'}",
                "llm_provider": LLMProviderName.HEURISTIC,
                "redis_url": None,
            }
        )
        engine = make_engine(s_over.database_url)
        Base.metadata.create_all(engine)
        sf = sessionmaker(bind=engine, expire_on_commit=False)
        with sf() as s:
            imp = import_manifest(
                s, manifest_path, SourceFetcher(settings.dataset_cache_dir, offline=offline)
            )
            items = {
                i.video_id: i
                for i in s.scalars(select(DatasetItem).where(DatasetItem.dataset_id == imp["dataset"])).all()
            }
            test_ids = sorted(v for v, i in items.items() if i.split == "test")
            ds = s.get(Dataset, imp["dataset"])
            if ds is not None:
                ds.public_blinding = False  # offline benchmark: no raters; the recommender must see the items
            for vid in test_ids:
                enqueue_stage_a(s, vid, force=True)
            s.commit()
        run_until_idle(build_context(settings=s_over, session_factory=sf, embedder=emb))
        per: list[dict[str, Any]] = []
        with sf() as s:
            for p in PERSONAS:
                user = User(token_hash=hashlib.sha256(str(p["id"]).encode()).hexdigest())
                user.preferences = UserPreferences(
                    interests=p["interests"], include_synthetic=False, diversity=0.0
                )
                s.add(user)
                s.flush()
                rec = recommend(s, user.id, emb, limit=k, log_impressions=False)
                ranked = [it["video"]["id"] for it in rec["items"]]
                relevant = {v for v in test_ids if items[v].group_key == p["group"]}
                per.append(
                    {
                        "persona": p["id"],
                        "relevant": len(relevant),
                        f"ndcg@{k}": M.ndcg_at_k(ranked, relevant, k),
                        f"precision@{k}": M.precision_at_k(ranked, relevant, k),
                        f"recall@{k}": M.recall_at_k(ranked, relevant, k),
                        "top": ranked,
                        "shows_in_top": dict(Counter(items[v].group_key for v in ranked)),
                        "exploration_picks": sum(1 for it in rec["items"] if it["ranking"]["exploration"]),
                    }
                )
            s.rollback()
        keys = [f"ndcg@{k}", f"precision@{k}", f"recall@{k}"]
        results[name] = {
            "model_id": emb.model_id,
            "per_persona": per,
            "mean": {key: round(M.nanmean([r[key] for r in per]), 4) for key in keys},
        }
    return {
        "meta": {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_sha(),
            "dataset": f"{manifest['name']}@{manifest['version']}",
            "manifest_items_sha256": manifest["items_sha256"],
            "split": "test (held out)",
            "k": k,
            "evaluator": "heuristic baseline (quality features are heuristic, not AI or human)",
            "labels": "topical proxy: persona's matching publisher show; no human relevance or quality labels",
            "personas": PERSONAS,
        },
        "results": results,
    }


def render_markdown(r: dict[str, Any]) -> str:
    m = r["meta"]
    k = m["k"]
    out = [
        f"# Held-out recommendation check: {m['dataset']} {m['split']}",
        "",
        f"Commit `{m['git_commit']}`; manifest `{m['manifest_items_sha256'][:16]}…`; quality features: {m['evaluator']}.",
        f"> **Labels:** {m['labels']}. 5 personas × 17 items: indicative only, no confidence claimed.",
        "",
        f"| embedder | nDCG@{k} | P@{k} | R@{k} |",
        "|---|---:|---:|---:|",
    ]
    for name, x in r["results"].items():
        mn = x["mean"]
        out.append(f"| {name} | {mn[f'ndcg@{k}']} | {mn[f'precision@{k}']} | {mn[f'recall@{k}']} |")
    out += ["", "| embedder | persona | relevant | nDCG | shows in top-5 |", "|---|---|---:|---:|---|"]
    for name, x in r["results"].items():
        for p in x["per_persona"]:
            out.append(
                f"| {name} | {p['persona']} | {p['relevant']} | {round(p[f'ndcg@{k}'], 3)} | {p['shows_in_top']} |"
            )
    out.append("")
    return "\n".join(out)


def add_parser(sub: Any) -> None:
    p = sub.add_parser(
        "bench-recs-real", help="held-out topical recommendation check on the real pilot test split"
    )
    p.add_argument(
        "--manifest", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    )
    p.add_argument("--with-minilm", action="store_true", help="also evaluate the MiniLM ONNX embedder")
    p.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "reports"))
    p.add_argument("--tag", default="recs-real_pilot_v1-test")
    p.set_defaults(fn=main)


def main(args: argparse.Namespace) -> None:
    embedders: dict[str, Embedder] = {"hashing": HashingEmbedder()}
    if args.with_minilm:
        from avd.embeddings.onnx_minilm import OnnxMiniLMEmbedder

        embedders["minilm_onnx"] = OnnxMiniLMEmbedder(get_settings().embedding_model_dir)
    report = run(Path(args.manifest), embedders)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    (out / f"{args.tag}.md").write_text(render_markdown(report))
    print(render_markdown(report))


# ---------------------------------------------------------------------------------- diagnostic
# Phase 3, exploratory: *why* did hashing beat MiniLM on the test-split proxy? Run on the DEV
# split only (it informs no decision), with personas written for the dev shows before any dev
# result was seen. Three conditions separate the suspects:
#   full        - the production recommender (relevance calibrated and blended with quality);
#   similarity  - rank by raw cosine between interests and the stored video vector only;
#   recipe swap - MiniLM over the keyword-bag recipe hashing uses, instead of transcript windows.
DEV_PERSONAS: list[dict[str, Any]] = [
    {
        "id": "gopher",
        "group": "changelog:gotime",
        "interests": ["the Go programming language and its ecosystem"],
    },
    {
        "id": "ml_practitioner",
        "group": "changelog:practicalai",
        "interests": ["applying machine learning and AI models in practice"],
    },
    {
        "id": "founder",
        "group": "changelog:founderstalk",
        "interests": ["founding and growing a startup company"],
    },
    {
        "id": "podcast_insider",
        "group": "changelog:backstage",
        "interests": ["behind the scenes of making a podcast and running a media business"],
    },
]


def _index_split(manifest_path: Path, split: str, emb: Embedder) -> tuple[Any, dict[str, Any]]:
    """Import one split into a throwaway SQLite database, evaluate it with the heuristic baseline
    and embed it with ``emb``. Blinding is lifted inside this private copy only: an offline
    diagnostic has no raters, and the recommender must see the items."""
    settings = get_settings()
    tmp = Path(tempfile.mkdtemp(prefix="avd-recs-diag-"))
    s_over = settings.model_copy(
        update={
            "database_url": f"sqlite:///{tmp / 'r.db'}",
            "llm_provider": LLMProviderName.HEURISTIC,
            "redis_url": None,
        }
    )
    engine = make_engine(s_over.database_url)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as s:
        imp = import_manifest(s, manifest_path, SourceFetcher(settings.dataset_cache_dir, offline=True))
        items = {
            i.video_id: i
            for i in s.scalars(select(DatasetItem).where(DatasetItem.dataset_id == imp["dataset"])).all()
            if i.split == split
        }
        ds = s.get(Dataset, imp["dataset"])
        assert ds is not None
        ds.public_blinding = False
        for vid in sorted(items):
            enqueue_stage_a(s, vid, force=True)
        s.commit()
    run_until_idle(build_context(settings=s_over, session_factory=sf, embedder=emb))
    return sf, items


class _RecipeOverride:
    """An embedder that uses another document recipe (isolates representation from the model)."""

    def __init__(self, inner: Embedder, recipe: str) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.dim = inner.dim
        self.preprocessing = recipe
        self.relevance_floor = inner.relevance_floor
        self.relevance_ceiling = inner.relevance_ceiling

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(texts)


def diagnose(
    manifest_path: Path, embedders: dict[str, Embedder], split: str = "dev", k: int = 5
) -> dict[str, Any]:
    import numpy as np

    from avd.db.models import Video

    if split == "test":
        raise ValueError("the diagnostic must not use the held-out split")
    manifest = load_manifest(manifest_path)
    out: dict[str, Any] = {}
    for name, emb in embedders.items():
        sf, items = _index_split(manifest_path, split, emb)
        rows: dict[str, list[dict[str, Any]]] = {"full": [], "similarity": []}
        with sf() as s:
            vecs = {
                v.id: np.asarray(v.embedding, dtype=float)
                for v in s.scalars(select(Video).where(Video.id.in_(list(items)))).all()
                if v.embedding is not None
            }
            for p in DEV_PERSONAS:
                relevant = {v for v in items if items[v].group_key == p["group"]}
                user = User(token_hash=hashlib.sha256(f"{name}:{p['id']}".encode()).hexdigest())
                user.preferences = UserPreferences(
                    interests=p["interests"], include_synthetic=False, diversity=0.0
                )
                s.add(user)
                s.flush()
                full = [
                    it["video"]["id"]
                    for it in recommend(s, user.id, emb, limit=k, log_impressions=False)["items"]
                ]
                q = np.asarray(emb.embed(p["interests"])[0], dtype=float)
                sims = {
                    v: float(q @ x / ((np.linalg.norm(q) * np.linalg.norm(x)) or 1.0))
                    for v, x in vecs.items()
                }
                sim_rank = sorted(sims, key=lambda v: -sims[v])[:k]
                for cond, ranked in (("full", full), ("similarity", sim_rank)):
                    rows[cond].append(
                        {
                            "persona": p["id"],
                            "relevant": len(relevant),
                            "ndcg": M.ndcg_at_k(ranked, relevant, k),
                            "top": ranked,
                            "returned": len(ranked),
                        }
                    )
                rows.setdefault("similarity_margin", []).append(
                    {
                        "persona": p["id"],
                        "relevant_mean": round(float(np.mean([sims[v] for v in relevant if v in sims])), 4)
                        if relevant
                        else None,
                        "other_mean": round(float(np.mean([sims[v] for v in sims if v not in relevant])), 4),
                    }
                )
            s.rollback()
        out[name] = {
            "model_id": emb.model_id,
            "recipe": emb.preprocessing,
            "ndcg_mean": {
                c: round(M.nanmean([r["ndcg"] for r in rows[c]]), 4) for c in ("full", "similarity")
            },
            "full_returned_mean": round(sum(r["returned"] for r in rows["full"]) / len(rows["full"]), 2),
            "rows": rows,
        }
    return {
        "meta": {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_sha(),
            "dataset": f"{manifest['name']}@{manifest['version']}",
            "split": f"{split} (exploratory; informs no decision)",
            "k": k,
            "labels": "topical proxy: persona's matching publisher show",
            "personas": DEV_PERSONAS,
        },
        "results": out,
    }


def render_diagnostic(r: dict[str, Any]) -> str:
    m = r["meta"]
    lines = [
        f"# Why do the embedders disagree? Diagnostic on {m['dataset']} {m['split']}",
        "",
        f"Commit `{m['git_commit']}`. Labels: {m['labels']}. {len(m['personas'])} personas; tiny n,"
        " exploratory: no default is changed on this basis.",
        "",
        f"| embedder | recipe | nDCG@{m['k']} full recommender | nDCG@{m['k']} similarity only | items returned (full) |",
        "|---|---|---:|---:|---:|",
    ]
    for name, x in r["results"].items():
        lines.append(
            f"| {name} | {x['recipe']} | {x['ndcg_mean']['full']} | {x['ndcg_mean']['similarity']} | {x['full_returned_mean']} |"
        )
    lines += ["", "Mean cosine to relevant vs other items (per persona):", ""]
    for name, x in r["results"].items():
        lines.append(
            f"* {name}: "
            + "; ".join(
                f"{s['persona']} {s['relevant_mean']} vs {s['other_mean']}"
                for s in x["rows"]["similarity_margin"]
            )
        )
    lines.append("")
    return "\n".join(lines)


def add_diagnostic_parser(sub: Any) -> None:
    p = sub.add_parser(
        "bench-recs-diagnose", help="exploratory dev-split diagnostic: embedder vs recipe vs ranking"
    )
    p.add_argument(
        "--manifest", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    )
    p.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "reports"))
    p.add_argument("--tag", default="recs-diagnostic-real_pilot_v1-dev")
    p.set_defaults(fn=diagnose_main)


def diagnose_main(args: argparse.Namespace) -> None:
    from avd.embeddings.onnx_minilm import OnnxMiniLMEmbedder

    minilm = OnnxMiniLMEmbedder(get_settings().embedding_model_dir)
    embedders: dict[str, Embedder] = {
        "hashing": HashingEmbedder(),
        "minilm_chunks": minilm,
        "minilm_keywords": _RecipeOverride(minilm, "meta_keywords_v1"),
    }
    report = diagnose(Path(args.manifest), embedders)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    (out / f"{args.tag}.md").write_text(render_diagnostic(report))
    print(render_diagnostic(report))


def _rank_metrics(ranked: list[str], relevant: set[str], k: int) -> dict[str, float]:
    hits = [v in relevant for v in ranked[:k]]
    first = next((i for i, v in enumerate(ranked) if v in relevant), None)
    return {
        f"recall@{k}": sum(hits) / len(relevant) if relevant else float("nan"),
        f"precision@{k}": sum(hits) / k,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        f"ndcg@{k}": M.ndcg_at_k(ranked, relevant, k),
    }


def stage_analysis(
    manifest_path: Path,
    embedders: dict[str, Embedder],
    split: str = "dev",
    k: int = 5,
    diversities: tuple[float, ...] = (0.0, 0.3),
) -> dict[str, Any]:
    """Where do relevant items go? For each embedder, on the identical candidate pool (the whole
    split): retrieval quality by similarity alone, then the full recommender, with every relevant
    item's fate traced (never retrieved / ineligible / below the relevance floor / channel cap /
    not selected / selected). Labels are the topical proxy (persona's show), so the per-channel
    cap is *expected* to cost relevant items here; that is reported, not "fixed" by the label."""
    import time

    import numpy as np

    from avd.db.models import Video

    if split == "test":
        raise ValueError("stage analysis must not use the held-out split")
    manifest = load_manifest(manifest_path)
    results: dict[str, Any] = {}
    all_sims: dict[str, dict[str, dict[str, float]]] = {}  # embedder -> persona -> item -> cosine
    relevant_by_persona: dict[str, set[str]] = {}
    for name, emb in embedders.items():
        sf, items = _index_split(manifest_path, split, emb)
        per_persona = []
        with sf() as s:
            vecs = {
                v.id: np.asarray(v.embedding, dtype=float)
                for v in s.scalars(select(Video).where(Video.id.in_(list(items)))).all()
                if v.embedding is not None
            }
            channels = {
                v.id: v.channel_id for v in s.scalars(select(Video).where(Video.id.in_(list(items)))).all()
            }
            for p in DEV_PERSONAS:
                relevant = {v for v in items if items[v].group_key == p["group"]}
                q = np.asarray(emb.embed(p["interests"])[0], dtype=float)
                sims = {
                    v: float(q @ x / ((np.linalg.norm(q) * np.linalg.norm(x)) or 1.0))
                    for v, x in vecs.items()
                }
                sim_rank = sorted(sims, key=lambda v: -sims[v])
                all_sims.setdefault(name, {})[p["id"]] = sims
                relevant_by_persona[p["id"]] = relevant
                row: dict[str, Any] = {
                    "persona": p["id"],
                    "relevant": len(relevant),
                    "relevant_channels": len({channels[v] for v in relevant}),
                    "similarity_only": _rank_metrics(sim_rank, relevant, k),
                    "relevant_below_floor": sum(
                        1 for v in relevant if calibrated_relevance(emb, sims.get(v, -1.0)) <= 0
                    ),
                    "full": {},
                }
                for div in diversities:
                    user = User(token_hash=hashlib.sha256(f"{name}:{p['id']}:{div}".encode()).hexdigest())
                    user.preferences = UserPreferences(
                        interests=p["interests"], include_synthetic=False, diversity=div
                    )
                    s.add(user)
                    s.flush()
                    trace: dict[str, str] = {}
                    t0 = time.perf_counter()
                    res = recommend(s, user.id, emb, limit=k, log_impressions=False, trace=trace)
                    latency = (time.perf_counter() - t0) * 1000
                    ranked = [it["video"]["id"] for it in res["items"]]
                    explore = {it["video"]["id"] for it in res["items"] if it["ranking"]["exploration"]}
                    topical = [v for v in ranked if v not in explore]
                    fates = Counter(trace.get(v, "not_retrieved") for v in relevant)
                    row["full"][str(div)] = {
                        **_rank_metrics(ranked, relevant, k),
                        "returned": len(ranked),
                        "exploration_slots": len(explore),
                        "off_topic_rate_topical": round(
                            sum(v not in relevant for v in topical) / len(topical), 4
                        )
                        if topical
                        else None,
                        "relevant_fates": dict(fates),
                        "latency_ms": round(latency, 1),
                    }
                per_persona.append(row)
            s.rollback()

        def mean(path: list[str], rows: list[dict[str, Any]] = per_persona) -> float:
            vals = []
            for r in rows:
                x: Any = r
                for key in path:
                    x = x[key]
                vals.append(float(x))
            return round(M.nanmean(vals), 4)

        fates_total: dict[str, Counter[str]] = {
            str(d): sum((Counter(r["full"][str(d)]["relevant_fates"]) for r in per_persona), Counter())
            for d in diversities
        }
        results[name] = {
            "model_id": emb.model_id,
            "recipe": emb.preprocessing,
            "floor_ceiling": [emb.relevance_floor, emb.relevance_ceiling],
            "similarity_only": {m: mean(["similarity_only", m]) for m in per_persona[0]["similarity_only"]},
            "full": {
                str(d): {
                    **{
                        m: mean(["full", str(d), m])
                        for m in (f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}")
                    },
                    "returned": mean(["full", str(d), "returned"]),
                    "exploration_slots": mean(["full", str(d), "exploration_slots"]),
                    "relevant_fates": dict(fates_total[str(d)]),
                    "latency_ms_mean": mean(["full", str(d), "latency_ms"]),
                }
                for d in diversities
            },
            "personas": per_persona,
        }
    # Hybrid retrieval by score fusion: the mean of a lexical and a semantic cosine per item.
    # Embedding storage holds one 384-dimension vector per video, so a hybrid is evaluated on
    # retrieval only (no full-recommender run).
    fusions = {}
    lexical = "hashing" if "hashing" in all_sims else None
    for sem in [n for n in all_sims if n.startswith("minilm")]:
        if lexical is None:
            break
        rows = []
        for pid, rel in relevant_by_persona.items():
            a, b = all_sims[lexical][pid], all_sims[sem][pid]
            fused = {v: (a[v] + b[v]) / 2 for v in a.keys() & b.keys()}
            rows.append(_rank_metrics(sorted(fused, key=lambda v: -fused[v]), rel, k))
        fusions[f"hybrid({lexical}+{sem})"] = {m: round(M.nanmean([r[m] for r in rows]), 4) for m in rows[0]}
    return {
        "fusion_similarity_only": fusions,
        "meta": {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_sha(),
            "dataset": f"{manifest['name']}@{manifest['version']}",
            "split": f"{split} (exploratory; development data only)",
            "k": k,
            "labels": "topical proxy: persona's matching publisher show (not human relevance judgements)",
            "personas": DEV_PERSONAS,
            "quality": "heuristic baseline (no AI model available); quality features are not AI judgements",
        },
        "results": results,
    }


def render_stages(r: dict[str, Any]) -> str:
    m, k = r["meta"], r["meta"]["k"]
    lines = [
        f"# Recommendation stage analysis: {m['dataset']} {m['split']}",
        "",
        f"Commit `{m['git_commit']}`. Labels: {m['labels']}. Quality: {m['quality']}.",
        f"{len(m['personas'])} personas; small n: read directions, not decimals.",
        "",
        "## Retrieval only (similarity over the identical candidate pool)",
        "",
        f"| embedder | recall@{k} | precision@{k} | MRR | nDCG@{k} |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, x in r["results"].items():
        so = x["similarity_only"]
        lines.append(
            f"| {name} | {so[f'recall@{k}']} | {so[f'precision@{k}']} | {so['mrr']} | {so[f'ndcg@{k}']} |"
        )
    for name, so in r.get("fusion_similarity_only", {}).items():
        lines.append(
            f"| {name} (score fusion) | {so[f'recall@{k}']} | {so[f'precision@{k}']} | {so['mrr']} | {so[f'ndcg@{k}']} |"
        )
    for d in next(iter(r["results"].values()))["full"]:
        lines += [
            "",
            f"## Full recommender, diversity {d}",
            "",
            f"| embedder | recall@{k} | precision@{k} | MRR | nDCG@{k} | returned | exploration slots | relevant items' fate |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for name, x in r["results"].items():
            f = x["full"][d]
            fates = ", ".join(f"{a} {b}" for a, b in sorted(f["relevant_fates"].items()))
            lines.append(
                f"| {name} | {f[f'recall@{k}']} | {f[f'precision@{k}']} | {f['mrr']} | {f[f'ndcg@{k}']} | {f['returned']} | {f['exploration_slots']} | {fates} |"
            )
    lines.append("")
    return "\n".join(lines)


def add_stages_parser(sub: Any) -> None:
    p = sub.add_parser(
        "bench-recs-stages", help="dev-split: retrieval vs ranking, and where relevant items are lost"
    )
    p.add_argument(
        "--manifest", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    )
    p.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "reports"))
    p.add_argument("--tag", default="recs-stages-real_pilot_v1-dev")
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(fn=stages_main)


def stages_main(args: argparse.Namespace) -> None:
    from avd.embeddings.onnx_minilm import OnnxMiniLMEmbedder

    minilm = OnnxMiniLMEmbedder(get_settings().embedding_model_dir)
    hashing = HashingEmbedder()
    embedders: dict[str, Embedder] = {
        "hashing": hashing,
        "minilm_chunks": minilm,
        "minilm_keywords": _RecipeOverride(minilm, "meta_keywords_v1"),
    }
    report = stage_analysis(Path(args.manifest), embedders, k=args.k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    (out / f"{args.tag}.md").write_text(render_stages(report))
    print(render_stages(report))
