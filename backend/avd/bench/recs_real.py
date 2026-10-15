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
from avd.db.models import Base, DatasetItem, User, UserPreferences
from avd.db.session import make_engine
from avd.embeddings.base import Embedder, HashingEmbedder
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
