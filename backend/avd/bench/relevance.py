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
    gains = [grades.get(v, 0.0) for v in ranked[:k]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    first = next((i for i, v in enumerate(ranked) if grades.get(v, 0) >= 1), None)
    return {
        f"ndcg@{k}": dcg / idcg if idcg else float("nan"),
        f"precision@{k}": sum(g >= 1 for g in gains) / k,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        f"off_topic@{k}": sum(g == 0 for g in gains) / max(1, len(gains)),
    }


def index_dev(embedders: dict[str, Any]) -> dict[str, tuple[Any, dict[str, list[float]]]]:
    """Real document vectors (transcripts, the production recipe) for every dev item."""
    from sqlalchemy import select

    from avd.bench.recs_real import _index_split
    from avd.db.models import Video

    out = {}
    for name, emb in embedders.items():
        sf, items = _index_split(PILOT, "dev", emb)
        with sf() as s:
            vecs = {
                v.id: list(v.embedding)
                for v in s.scalars(select(Video).where(Video.id.in_(list(items)))).all()
                if v.embedding is not None
            }
        out[name] = (emb, vecs)
    return out


def evaluate(
    judgments_path: Path | None,
    indexes: dict[str, tuple[Any, dict[str, list[float]]]] | None = None,
    k: int = 5,
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
    results: dict[str, Any] = {}
    import numpy as np

    for name, (emb, raw) in (indexes or {}).items():
        vecs = {vid: np.asarray(v, dtype=float) for vid, v in raw.items()}
        rows = []
        for q in queries:
            g = {vid: grades[(q.query_id, vid)] for vid in items if (q.query_id, vid) in grades}
            if not g:
                continue
            qv = np.asarray(emb.embed([q.text])[0], dtype=float)
            sims = {
                v: float(qv @ x / ((np.linalg.norm(qv) * np.linalg.norm(x)) or 1.0)) for v, x in vecs.items()
            }
            rows.append(graded_metrics(sorted(sims, key=lambda v: -sims[v]), g, k))
        results[name] = {m: round(statistics.fmean(r[m] for r in rows), 4) for m in rows[0]} if rows else {}
    return {"status": "measured", "agreement": agreement, "judgments": len(judgments), "retrieval": results}


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
