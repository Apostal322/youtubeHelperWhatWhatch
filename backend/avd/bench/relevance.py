"""Human-judged query-video relevance benchmark (``avd bench-relevance``).

Replaces the show-membership proxy with graded human judgments (0/1/2) on the dev split. No
judgments exist yet: without a judgments file the command reports the benchmark as blocked and
computes nothing. See ``benchmarks/datasets/relevance_v1/README.md`` for the protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from avd.config import REPO_ROOT

DATA = REPO_ROOT / "benchmarks" / "datasets" / "relevance_v1"
PILOT = REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json"


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: str = Field(pattern=r"^q\d{2,4}$")
    text: str = Field(min_length=3, max_length=300)
    language: str = Field(max_length=8)
    audience: str = Field(max_length=40)
    depth: str = Field(max_length=20)
    # Who wrote it: the project author, who knew the shows' topics (relevance_v1 audit), or a rater
    # who wrote the need before seeing anything of the catalog. Results are reported per source.
    source: Literal["catalog_aware_author", "blind_rater"] = "catalog_aware_author"


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: str
    video_id: str
    rater: str = Field(min_length=2, max_length=40)
    grade: Literal[0, 1, 2]
    language_ok: bool | None = None
    depth: Literal["too_basic", "matches", "too_advanced", "unclear"] | None = None
    audience: Literal["too_basic", "matches", "too_advanced", "unclear"] | None = None
    minutes_spent: float | None = Field(default=None, ge=0)


class RelevanceError(ValueError):
    pass


def load_queries(path: Path = DATA / "queries.jsonl") -> list[Query]:
    return [Query.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def dev_items(manifest: Path = PILOT) -> dict[str, dict[str, Any]]:
    m = json.loads(manifest.read_text())
    return {it["item_id"]: it for it in m["items"] if it["split"] == "dev"}


def load_judgments(path: Path, queries: list[Query], items: dict[str, Any]) -> list[Judgment]:
    out: list[Judgment] = []
    qids = {q.query_id for q in queries}
    seen: set[tuple[str, str, str]] = set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            j = Judgment.model_validate_json(line)
        except ValidationError as exc:
            raise RelevanceError(f"line {n}: {exc.errors()[0]['msg']}") from exc
        if j.query_id not in qids:
            raise RelevanceError(f"line {n}: unknown query {j.query_id}")
        if j.video_id not in items:
            # Judging test or held-out items here would spend them; refuse.
            raise RelevanceError(f"line {n}: {j.video_id} is not a dev-split item")
        key = (j.query_id, j.video_id, j.rater)
        if key in seen:
            raise RelevanceError(f"line {n}: duplicate judgment {key}")
        seen.add(key)
        out.append(j)
    return out


def write_sheet(out: Path, queries: list[Query], items: dict[str, dict[str, Any]]) -> int:
    """A blank judging sheet: one row per query x item, with the publisher's link."""
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["query_id", "query", "audience", "depth", "video_id", "show", "episode", "link", "grade_0_1_2"]
        )
        n = 0
        for q in queries:
            for vid, it in sorted(items.items()):
                w.writerow(
                    [
                        q.query_id,
                        q.text,
                        q.audience,
                        q.depth,
                        vid,
                        it.get("show"),
                        it.get("episode"),
                        it.get("provenance_url"),
                        "",
                    ]
                )
                n += 1
    return n


def pooled_grades(judgments: list[Judgment]) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Median grade per (query, item), plus inter-rater agreement (ordinal Krippendorff alpha)."""
    import numpy as np

    from avd.bench.stats import krippendorff_alpha

    by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    raters = sorted({j.rater for j in judgments})
    for j in judgments:
        by_pair[(j.query_id, j.video_id)].append(j.grade)
    pairs = sorted(by_pair)
    matrix = np.full((len(raters), len(pairs)), np.nan)
    for j in judgments:
        matrix[raters.index(j.rater), pairs.index((j.query_id, j.video_id))] = j.grade
    alpha = krippendorff_alpha(matrix, level="ordinal") if len(raters) >= 2 else float("nan")
    raters_per_pair = [len(v) for v in by_pair.values()]
    return {p: float(statistics.median(g)) for p, g in by_pair.items()}, {
        "raters": len(raters),
        "pairs": len(pairs),
        "min_raters_per_pair": min(raters_per_pair) if raters_per_pair else 0,
        "krippendorff_alpha_ordinal": None if math.isnan(alpha) else round(alpha, 3),
    }


def graded_metrics(ranked: list[str], grades: dict[str, float], k: int = 5) -> dict[str, float]:
    """``grades``: pooled grade per judged item (0, 1, 2). Unjudged items count as not relevant.
    nDCG and Recall are NaN for a query with no relevant item; aggregate them with ``nanmean``
    and report how many queries had none, so one empty query cannot poison the mean."""
    gains = [grades.get(v, 0.0) for v in ranked[:k]]
    relevant = {v for v, g in grades.items() if g >= 1}
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    first = next((i for i, v in enumerate(ranked) if grades.get(v, 0) >= 1), None)
    return {
        f"ndcg@{k}": dcg / idcg if idcg else float("nan"),
        f"recall@{k}": sum(v in relevant for v in ranked[:k]) / len(relevant) if relevant else float("nan"),
        f"precision@{k}": sum(g >= 1 for g in gains) / k,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        f"off_topic@{k}": sum(g == 0 for g in gains) / max(1, len(gains)),
    }


def disputed_pairs(judgments: list[Judgment]) -> list[dict[str, Any]]:
    """(query, item) pairs whose grades span the whole scale (0 and 2): kept, reported, never averaged away."""
    by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
    for j in judgments:
        by_pair[(j.query_id, j.video_id)].append(j.grade)
    return [
        {"query_id": q, "video_id": v, "grades": sorted(g)}
        for (q, v), g in sorted(by_pair.items())
        if max(g) - min(g) >= 2
    ]


Index = tuple[Any, dict[str, list[float]], Any]  # embedder, dev vectors, session factory (or None)


def index_dev(embedders: dict[str, Any]) -> dict[str, Index]:
    """Real document vectors (transcripts, the production recipe) for every dev item, plus the
    database they live in, so the full recommender can rank the same pool."""
    from sqlalchemy import select

    from avd.bench.recs_real import _index_split
    from avd.db.models import Video

    out: dict[str, Index] = {}
    for name, emb in embedders.items():
        sf, items = _index_split(PILOT, "dev", emb)
        with sf() as s:
            vecs = {
                v.id: list(v.embedding)
                for v in s.scalars(select(Video).where(Video.id.in_(list(items)))).all()
                if v.embedding is not None
            }
        out[name] = (emb, vecs, sf)
    return out


def _cosines(emb: Any, vecs: dict[str, list[float]], text: str) -> dict[str, float]:
    import numpy as np

    qv = np.asarray(emb.embed([text])[0], dtype=float)
    out = {}
    for v, x in vecs.items():
        xv = np.asarray(x, dtype=float)
        out[v] = float(qv @ xv / ((np.linalg.norm(qv) * np.linalg.norm(xv)) or 1.0))
    return out


def _recommended(emb: Any, sf: Any, text: str, k: int) -> list[str]:
    """The final recommendation list for a user whose only stated interest is the query."""
    import hashlib

    from avd.db.models import User, UserPreferences
    from avd.recsys.recommender import recommend

    with sf() as s:
        user = User(token_hash=hashlib.sha256(f"relevance:{text}".encode()).hexdigest())
        user.preferences = UserPreferences(interests=[text], include_synthetic=False)
        s.add(user)
        s.flush()
        res = recommend(s, user.id, emb, limit=k, log_impressions=False)
        s.rollback()
    return [it["video"]["id"] for it in res["items"]]


def _summarise(rows: list[dict[str, float]]) -> dict[str, Any]:
    from avd.bench import metrics as M

    if not rows:
        return {}
    out: dict[str, Any] = {m: round(M.nanmean([r[m] for r in rows]), 4) for m in rows[0]}
    out["queries"] = len(rows)
    out["queries_without_relevant_item"] = sum(1 for r in rows if math.isnan(next(iter(r.values()))))
    return out


def evaluate(
    judgments_path: Path | None, indexes: dict[str, Index] | None = None, k: int = 5
) -> dict[str, Any]:
    queries, items = load_queries(), dev_items()
    if judgments_path is None or not judgments_path.exists():
        return {
            "status": "blocked_missing_labels",
            "detail": "no human relevance judgments exist yet (benchmarks/datasets/relevance_v1/README.md)",
            "queries": len(queries),
            "items": len(items),
        }
    judgments = load_judgments(judgments_path, queries, items)
    grades, agreement = pooled_grades(judgments)
    if agreement["min_raters_per_pair"] < 3:
        agreement["warning"] = (
            "fewer than 3 raters on some pairs: report, but do not compare retrievers on it"
        )
    judged = {
        q.query_id: {v: grades[(q.query_id, v)] for v in items if (q.query_id, v) in grades} for q in queries
    }
    judged = {q: g for q, g in judged.items() if g}
    results = _rankings(queries, judged, indexes or {}, k)
    by_source = {
        src: _rankings([q for q in queries if q.source == src], judged, indexes or {}, k)
        for src in sorted({q.source for q in queries if q.query_id in judged})
    }
    return {
        "status": "measured",
        "label_source": "human relevance judgments (relevance_v1)",
        "agreement": agreement,
        "disputed_pairs": disputed_pairs(judgments),
        "judgments": len(judgments),
        "judged_queries": len(judged),
        "judged_queries_by_source": {
            src: sum(1 for q in queries if q.source == src and q.query_id in judged) for src in by_source
        },
        "k": k,
        "results": results,
        # Catalog-aware queries favour lexical retrieval; only blind-rater queries test generalisation.
        "results_by_query_source": by_source,
    }


def _rankings(
    queries: list[Query], judged: dict[str, dict[str, float]], indexes: dict[str, Index], k: int
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    sims: dict[str, dict[str, dict[str, float]]] = {}
    for name, (emb, vecs, sf) in indexes.items():
        sim_rows, rec_rows = [], []
        for q in queries:
            if q.query_id not in judged:
                continue
            sims.setdefault(name, {})[q.query_id] = c = _cosines(emb, vecs, q.text)
            sim_rows.append(graded_metrics(sorted(c, key=lambda v: -c[v]), judged[q.query_id], k))
            if sf is not None:
                rec_rows.append(graded_metrics(_recommended(emb, sf, q.text, k), judged[q.query_id], k))
        results[name] = {"retrieval": _summarise(sim_rows)}
        if rec_rows:
            results[name]["final_recommendations"] = _summarise(rec_rows)
    if {"hashing", "minilm"} <= set(sims):
        # Score fusion (mean cosine): the stored vectors are fixed-width, so no concatenated hybrid.
        rows = []
        for qid in sims["minilm"]:  # the judged queries of this subset
            fused = {v: (sims["hashing"][qid][v] + sims["minilm"][qid][v]) / 2 for v in sims["minilm"][qid]}
            rows.append(graded_metrics(sorted(fused, key=lambda v: -fused[v]), judged[qid], k))
        results["hybrid_fusion"] = {"retrieval": _summarise(rows)}
    return results


def add_parser(sub: Any) -> None:
    p = sub.add_parser("bench-relevance", help="human-judged query-video relevance (dev split)")
    p.add_argument("action", nargs="?", choices=["run", "sheet"], default="run")
    p.add_argument("--judgments", default=str(DATA / "judgments.jsonl"))
    p.add_argument("--out", default="relevance_sheet.csv")
    p.set_defaults(fn=main)


def main(args: argparse.Namespace) -> None:
    if args.action == "sheet":
        n = write_sheet(Path(args.out), load_queries(), dev_items())
        print(json.dumps({"sheet": args.out, "rows": n}))
        return
    judgments = Path(args.judgments)
    if not judgments.exists():
        print(json.dumps(evaluate(None), indent=2))
        return
    from avd.config import get_settings
    from avd.embeddings.base import HashingEmbedder

    embedders: dict[str, Any] = {"hashing": HashingEmbedder()}
    try:
        from avd.embeddings.onnx_minilm import OnnxMiniLMEmbedder

        embedders["minilm"] = OnnxMiniLMEmbedder(get_settings().embedding_model_dir)
    except Exception as exc:  # model not fetched: compare what is available, say so
        print(f"MiniLM unavailable ({type(exc).__name__}); hashing only")
    print(json.dumps(evaluate(judgments, index_dev(embedders)), indent=2))
