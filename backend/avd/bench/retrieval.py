"""Retrieval experiment: can an embedder find an episode from its creator-written title?

Corpus: a seeded sample of Talk Python To Me YouTube-edition transcripts (GPL-3.0, publisher
repository at a pinned commit). Documents contain **only the transcript**; titles are withheld.
Queries are the creator's own episode titles, taken from the file slugs. Each query has exactly
one relevant document.

Methods: the lexical hashing embedder (``meta_keywords_v1``), MiniLM ONNX (``chunk_mean_v1``),
and reciprocal-rank fusion of the two (RRF, k=60). Metrics: Recall@1/5/10, MRR, nDCG@10,
false-match rate, confident false matches, latency. CIs come from a query-level bootstrap.

Caveat: titles often share rare words with their transcripts (e.g. a library name), which favours
lexical matching. This measures known-item retrieval, not topical recommendation quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from avd.bench import stats as S
from avd.bench.harness import _git_sha
from avd.config import REPO_ROOT, get_settings
from avd.content.processing import normalize_transcript
from avd.datasets.fetch import SourceFetcher, sha256_bytes
from avd.datasets.pilot import list_candidates
from avd.datasets.sources import TALKPYTHON, item_metadata, parse
from avd.db.models import Video
from avd.embeddings import documents
from avd.embeddings.base import Embedder, HashingEmbedder

RRF_K = 60


def build_corpus(clone: Path, fetcher: SourceFetcher, n: int, seed: int) -> list[dict[str, Any]]:
    paths = list_candidates(TALKPYTHON, clone, "youtube_transcripts")
    random.Random(f"{seed}:retrieval").shuffle(paths)
    corpus: list[dict[str, Any]] = []
    for path in paths:
        if len(corpus) >= n:
            break
        raw = fetcher.get(TALKPYTHON, path)
        transcript, _ = parse(TALKPYTHON, path, raw.decode("utf-8"))
        transcript, _ = normalize_transcript(transcript)
        meta = item_metadata(TALKPYTHON, path)
        if transcript.word_count < 1000 or len(meta["title"].split()) < 2:
            continue  # too short, or a one-word slug that is not a usable query
        corpus.append(
            {"path": path, "sha256": sha256_bytes(raw), "query": meta["title"], "transcript": transcript}
        )
    return corpus


def _doc_vectors(embedder: Embedder, corpus: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    t0 = time.monotonic()
    vecs = []
    for i, d in enumerate(corpus):
        video = Video(id=f"doc{i}", title="", description="", tags=[], topics=[])  # titles withheld
        vecs.append(documents.embed_video(embedder, video, d["transcript"]).vector)
    return np.array(vecs), time.monotonic() - t0


def _rank_lists(
    embedder: Embedder, corpus: list[dict[str, Any]], docs: np.ndarray
) -> tuple[list[list[int]], list[list[float]], list[float]]:
    ranks, sims_sorted, latencies = [], [], []
    for d in corpus:
        t0 = time.monotonic()
        q = np.array(embedder.embed([d["query"]])[0])
        sims = docs @ q / (np.linalg.norm(docs, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-12)
        order = list(np.argsort(-sims))
        latencies.append((time.monotonic() - t0) * 1000)
        ranks.append([int(i) for i in order])
        sims_sorted.append([float(sims[i]) for i in order])
    return ranks, sims_sorted, latencies


def _rrf(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    out = []
    for ra, rb in zip(a, b, strict=True):
        score: dict[int, float] = {}
        for r in (ra, rb):
            for pos, doc in enumerate(r):
                score[doc] = score.get(doc, 0.0) + 1.0 / (RRF_K + pos + 1)
        out.append(sorted(score, key=lambda d: -score[d]))
    return out


def _metrics(ranks: list[list[int]], sims: list[list[float]] | None, seed: int) -> dict[str, Any]:
    positions = [r.index(i) + 1 for i, r in enumerate(ranks)]  # rank of the relevant document
    units = [{"q": i, "pos": p} for i, p in enumerate(positions)]

    def recall(k: int) -> Any:
        return S.grouped_bootstrap(
            units,
            lambda u: u["q"],
            lambda us: float(np.mean([u["pos"] <= k for u in us])),
            n_boot=1000,
            seed=seed,
        )

    out: dict[str, Any] = {
        "n_queries": len(ranks),
        "recall@1": recall(1),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr": S.grouped_bootstrap(
            units,
            lambda u: u["q"],
            lambda us: float(np.mean([1 / u["pos"] for u in us])),
            n_boot=1000,
            seed=seed,
        ),
        "ndcg@10": S._f(float(np.mean([1 / math.log2(p + 1) if p <= 10 else 0.0 for p in positions]))),
        "median_rank": float(np.median(positions)),
        "false_match_rate": S._f(float(np.mean([p != 1 for p in positions]))),
    }
    if sims is not None:
        correct_top = [s[0] for s, p in zip(sims, positions, strict=True) if p == 1]
        thresh = float(np.median(correct_top)) if correct_top else float("inf")
        wrong = [s[0] for s, p in zip(sims, positions, strict=True) if p != 1]
        out["confident_false_matches"] = sum(1 for w in wrong if w >= thresh)
        out["confident_threshold"] = S._f(thresh)
        matched = [s[r.index(i)] for i, (s, r) in enumerate(zip(sims, ranks, strict=True))]
        nonmatched = [
            x
            for i, (s, r) in enumerate(zip(sims, ranks, strict=True))
            for j, x in zip(r, s, strict=True)
            if j != i
        ]
        out["similarity"] = {
            "relevant_median": S._f(float(np.median(matched))),
            "non_relevant_median": S._f(float(np.median(nonmatched))),
            "non_relevant_p95": S._f(float(np.percentile(nonmatched, 95))),
        }
    return out


def run(clone: Path, n: int = 100, seed: int = 7, offline: bool = False) -> dict[str, Any]:
    settings = get_settings()
    fetcher = SourceFetcher(settings.dataset_cache_dir, offline=offline)
    corpus = build_corpus(clone, fetcher, n, seed)
    from avd.embeddings.onnx_minilm import OnnxMiniLMEmbedder

    embedders: dict[str, Embedder] = {
        "hashing": HashingEmbedder(),
        "minilm_onnx": OnnxMiniLMEmbedder(settings.embedding_model_dir),
    }
    results: dict[str, Any] = {}
    ranks: dict[str, list[list[int]]] = {}
    for name, emb in embedders.items():
        docs, t_docs = _doc_vectors(emb, corpus)
        r, s, lat = _rank_lists(emb, corpus, docs)
        ranks[name] = r
        results[name] = {
            "model_id": emb.model_id,
            "preprocessing": emb.preprocessing,
            **_metrics(r, s, seed),
            "doc_embedding_seconds_total": round(t_docs, 2),
            "doc_embedding_ms_per_doc": round(t_docs / len(corpus) * 1000, 1),
            "query_latency_ms": S.percentiles(lat),
        }
    results["hybrid_rrf"] = {
        "method": f"reciprocal rank fusion (k={RRF_K}) of hashing + minilm_onnx",
        **_metrics(_rrf(ranks["hashing"], ranks["minilm_onnx"]), None, seed),
    }
    return {
        "meta": {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_sha(),
            "source": f"{TALKPYTHON.repo_url}@{TALKPYTHON.commit} (youtube_transcripts/)",
            "license": TALKPYTHON.rights.license,
            "seed": seed,
            "n_documents": len(corpus),
            "documents": [{"path": d["path"], "sha256": d["sha256"]} for d in corpus],
            "queries": "creator episode titles from file slugs; titles withheld from documents",
            "caveat": __doc__.split("Caveat:")[1].strip() if __doc__ else "",
        },
        "results": results,
    }


def render_markdown(r: dict[str, Any]) -> str:
    m = r["meta"]
    out = [
        "# Retrieval experiment: episode title → transcript (known-item)",
        "",
        f"Source {m['source']} ({m['license']}); {m['n_documents']} documents; seed {m['seed']}; commit `{m['git_commit']}`.",
        f"Queries: {m['queries']}.",
        "",
        f"> **Caveat:** {m['caveat']}",
        "",
        "| method | R@1 [95% CI] | R@5 | R@10 | MRR [95% CI] | nDCG@10 | median rank | false-match rate | confident false matches | query ms p50 / p95 | doc embed ms/doc |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for name, x in r["results"].items():

        def ci(v: dict[str, Any]) -> str:
            return f"{v['value']} [{v['ci_low']}, {v['ci_high']}]"

        lat = x.get("query_latency_ms") or {}
        out.append(
            f"| {name} | {ci(x['recall@1'])} | {x['recall@5']['value']} | {x['recall@10']['value']} | {ci(x['mrr'])} | "
            f"{x['ndcg@10']} | {x['median_rank']} | {x['false_match_rate']} | {x.get('confident_false_matches', '—')} | "
            f"{lat.get('p50', '—')} / {lat.get('p95', '—')} | {x.get('doc_embedding_ms_per_doc', '—')} |"
        )
    out += ["", "Similarity distributions (cosine):", ""]
    for name, x in r["results"].items():
        if "similarity" in x:
            out.append(f"* {name}: {x['similarity']}")
    out.append("")
    return "\n".join(out)


def main(args: argparse.Namespace) -> None:
    report = run(Path(args.clone), n=args.n, seed=args.seed, offline=args.offline)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"retrieval-talkpython-n{report['meta']['n_documents']}-s{args.seed}"
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2))
    (out / f"{tag}.md").write_text(render_markdown(report))
    print(render_markdown(report))


def add_parser(sub: Any) -> None:
    p = sub.add_parser("bench-retrieval", help="known-item retrieval experiment (titles -> transcripts)")
    p.add_argument("--clone", required=True, help="local clone of the Talk Python transcripts repository")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "reports"))
    p.add_argument("--tag", default=None)
    p.set_defaults(fn=lambda a: main(a))
