"""Command-line entry point: ``avd <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from avd.config import REPO_ROOT, get_settings
from avd.logging import configure_logging

if TYPE_CHECKING:
    from avd.datasets.fetch import SourceFetcher


def _migrate() -> None:
    from alembic import command
    from alembic.config import Config

    # Migrations ship inside the package, so this works for any install layout.
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parent / "migrations"))
    command.upgrade(cfg, "head")


def cmd_migrate(args: argparse.Namespace) -> None:
    _migrate()
    print("database migrated to head")


def cmd_seed_demo(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.services import seed_demo

    with session_scope() as s:
        n = seed_demo(s, get_settings())
    print(f"seeded {n} synthetic fixture videos and queued Stage A")


def cmd_worker(args: argparse.Namespace) -> None:
    from avd.workers.context import build_context
    from avd.workers.runner import run_forever, run_until_idle

    ctx = build_context()
    if args.once:
        print(json.dumps(run_until_idle(ctx, max_jobs=args.max_jobs, job_types=args.types)))
    else:
        run_forever(ctx)


def cmd_demo(args: argparse.Namespace) -> None:
    """Migrate, seed fixtures, process them, and print a summary table."""
    from sqlalchemy import select

    from avd.db.models import Evaluation, Video
    from avd.db.session import session_scope
    from avd.services import seed_demo
    from avd.workers.context import build_context
    from avd.workers.runner import run_until_idle

    settings = get_settings()
    _migrate()
    with session_scope() as s:
        seed_demo(s, settings)
    outcomes = run_until_idle(build_context())
    print(f"job outcomes: {outcomes}")
    mode = "SYNTHETIC DEMO (heuristic baseline, not AI)" if settings.demo_mode else "LLM"
    print(f"\nEvaluation mode: {mode}\n")
    print(f"{'video':16} {'genre':16} {'overall':>8} {'content':>8} {'present.':>8} {'technical':>9}  status")
    with session_scope() as s:
        rows = s.execute(
            select(Video.id, Video.genre, Evaluation).join(
                Evaluation, (Evaluation.video_id == Video.id) & Evaluation.is_current.is_(True)
            )
        ).all()
        for vid, genre, ev in sorted(rows, key=lambda r: -(r[2].overall_score or 0)):
            g = {k: v.get("score") for k, v in ev.group_scores.items()}

            def f(x: float | None) -> str:
                return "   —" if x is None else f"{x:.2f}"

            print(
                f"{vid:16} {genre or '?':16} {f(ev.overall_score):>8} {f(g.get('content')):>8} "
                f"{f(g.get('presentation')):>8} {f(g.get('technical')):>9}  {ev.status}"
            )


def cmd_discover(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.services import plan_discovery

    if get_settings().youtube_api_key is None:
        sys.exit("AVD_YOUTUBE_API_KEY is not set; discovery uses the official YouTube Data API")
    with session_scope() as s:
        print(f"enqueued {plan_discovery(s, args.n)} discovery jobs")


def cmd_import_dataset(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.ingest import import_licensed_dataset
    from avd.services import enqueue_stage_a

    with session_scope() as s:
        stats, video_ids = import_licensed_dataset(s, Path(args.path))
        for vid in video_ids:
            enqueue_stage_a(s, vid, force=True)
    print(json.dumps(stats))


def cmd_reevaluate_stale(args: argparse.Namespace) -> None:
    from datetime import UTC, datetime

    from avd.db.session import session_scope
    from avd.workers.queue import enqueue

    with session_scope() as s:
        enqueue(s, "reevaluate_stale", f"reevaluate_stale:{datetime.now(UTC):%Y%m%d%H%M%S}", priority=2.0)
    print("queued; run a worker to process")


def cmd_stats(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.workers.budget import llm_spent_today, youtube_units_today
    from avd.workers.queue import stats

    with session_scope() as s:
        print(
            json.dumps(
                {
                    "jobs": stats(s),
                    "llm_usd_today": llm_spent_today(s),
                    "youtube_units_today": youtube_units_today(s),
                },
                indent=2,
            )
        )


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("avd.api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_train_ranker(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.recsys.ltr_data import build_pairs
    from avd.recsys.ranker import DEFAULT_WEIGHTS, LinearRanker, pairwise_accuracy, train_pairwise_ranker

    with session_scope() as s:
        pairs = build_pairs(s)
    if len(pairs) < args.min_pairs:
        sys.exit(f"only {len(pairs)} preference pairs from feedback; need >= {args.min_pairs}")
    split = int(len(pairs) * 0.8)
    model = train_pairwise_ranker(pairs[:split])
    held = pairs[split:]
    baseline = LinearRanker(dict(DEFAULT_WEIGHTS))
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "version": model.version,
                "weights": model.weights,
                "heldout_pairwise_accuracy": pairwise_accuracy(model, held),
                "baseline_heldout_pairwise_accuracy": pairwise_accuracy(baseline, held),
            },
            indent=2,
        )
    )
    model.save(Path(args.out))


def cmd_llm_smoke(args: argparse.Namespace) -> None:
    from avd.smoke import main as smoke_main

    sys.exit(smoke_main(args))


def _fetcher(args: argparse.Namespace) -> SourceFetcher:
    from avd.datasets.fetch import SourceFetcher

    return SourceFetcher(Path(args.cache_dir or get_settings().dataset_cache_dir), offline=args.offline)


def cmd_dataset_build_pilot(args: argparse.Namespace) -> None:
    from avd.datasets.pilot import build_manifest

    clones = dict(c.split("=", 1) for c in args.clone)
    missing = {"changelog", "talkpython"} - clones.keys()
    if missing:
        sys.exit(f"--clone needed for: {', '.join(sorted(missing))} (git clone --filter=blob:none <repo>)")
    manifest = build_manifest(
        {k: Path(v) for k, v in clones.items()}, _fetcher(args), seed=args.seed, progress=print
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    splits: dict[str, int] = {}
    for it in manifest["items"]:
        splits[it["split"]] = splits.get(it["split"], 0) + 1
    print(
        json.dumps(
            {"items": len(manifest["items"]), "splits": splits, "items_sha256": manifest["items_sha256"]}
        )
    )


def cmd_dataset_import(args: argparse.Namespace) -> None:
    from avd.datasets.importer import import_manifest
    from avd.db.session import session_scope
    from avd.services import enqueue_stage_a

    with session_scope() as s:
        result = import_manifest(s, Path(args.manifest), _fetcher(args))
        if args.evaluate:
            for vid in result["video_ids"]:
                enqueue_stage_a(s, vid, force=True)
    print(json.dumps({k: v for k, v in result.items() if k != "video_ids"}))


def cmd_dataset_revoke(args: argparse.Namespace) -> None:
    from avd.datasets.importer import revoke_video_sources
    from avd.db.session import session_scope

    with session_scope() as s:
        print(json.dumps(revoke_video_sources(s, args.video_id, args.reason)))


def cmd_annotate(args: argparse.Namespace) -> None:
    from avd.annotation import service as svc
    from avd.db.session import session_scope

    with session_scope() as s:
        if args.annotate_cmd == "create-rater":
            rater, token = svc.create_rater(s, args.pseudonym, args.expertise, args.role)
            print(json.dumps({"id": rater.id, "pseudonym": rater.pseudonym, "role": rater.role}))
            print(f"token (shown once, give it to the rater privately): {token}", file=sys.stderr)
        elif args.annotate_cmd == "qc":
            print(json.dumps(svc.qc_report(s, args.dataset), indent=2))
        elif args.annotate_cmd == "export":
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            data = svc.export_labels(s, args.dataset)
            for name in ("labels", "repeats", "pairs", "items"):
                with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                    for row in data[name]:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({k: len(v) for k, v in data.items()}))


def cmd_bench_real(args: argparse.Namespace) -> None:
    from avd.bench.real import main as real_main

    real_main(args)


def cmd_gates(args: argparse.Namespace) -> None:
    from avd.bench.real import load_gates, lock_gates

    if args.gates_cmd == "lock":
        print(json.dumps(lock_gates(args.statement, new_test_set=args.new_test_set), indent=2))
    else:
        g = load_gates()
        print(json.dumps({"sha256": g["sha256"], "lock_valid": g["lock_valid"], "lock": g["lock"]}, indent=2))
        if not g["lock_valid"]:
            sys.exit(1)


def cmd_embeddings(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.emb_cmd == "fetch-model":
        from avd.embeddings.onnx_minilm import fetch_model

        print(f"verified model at {fetch_model(settings.embedding_model_dir)}")
        return
    from avd.db.session import get_sessionmaker
    from avd.embeddings.base import get_embedder
    from avd.embeddings.reindex import reindex

    print(json.dumps(reindex(get_sessionmaker(), get_embedder(), args.batch_size, args.dry_run), indent=2))


def cmd_bench(args: argparse.Namespace) -> None:
    from avd.bench.harness import main as bench_main

    bench_main(args)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    p = argparse.ArgumentParser(prog="avd", description="AI Video Discovery")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate").set_defaults(fn=cmd_migrate)
    sub.add_parser("seed-demo").set_defaults(fn=cmd_seed_demo)
    sub.add_parser("demo", help="migrate + seed fixtures + process + summary").set_defaults(fn=cmd_demo)
    w = sub.add_parser("worker")
    w.add_argument("--once", action="store_true", help="drain the queue and exit")
    w.add_argument("--max-jobs", type=int, default=None)
    w.add_argument("--types", nargs="*", default=None)
    w.set_defaults(fn=cmd_worker)
    d = sub.add_parser("discover")
    d.add_argument("--n", type=int, default=4)
    d.set_defaults(fn=cmd_discover)
    i = sub.add_parser("import-dataset")
    i.add_argument("path")
    i.set_defaults(fn=cmd_import_dataset)
    sub.add_parser("reevaluate-stale").set_defaults(fn=cmd_reevaluate_stale)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(fn=cmd_serve)
    t = sub.add_parser("train-ranker")
    t.add_argument("--out", default="ranker.json")
    t.add_argument("--min-pairs", type=int, default=50)
    t.set_defaults(fn=cmd_train_ranker)
    ls = sub.add_parser("llm-smoke", help="cost-capped live check of the real LLM evaluation path")
    ls.add_argument("--fixture", default="syn_sci_001", help="fixture video id to evaluate")
    ls.add_argument("--transcript", default=None, help="evaluate this VTT/SRT/JSON transcript instead")
    ls.add_argument("--max-cost", type=float, default=None, help="explicit spend cap in USD")
    ls.set_defaults(fn=cmd_llm_smoke)
    ds = sub.add_parser("dataset", help="licensed real-world datasets").add_subparsers(
        dest="dataset_cmd", required=True
    )
    bp = ds.add_parser("build-pilot", help="select items and write a manifest (needs local clones)")
    bp.add_argument(
        "--clone", action="append", default=[], help="source=DIR of a clone with the pinned commit"
    )
    bp.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1"))
    bp.add_argument("--seed", type=int, default=20261015)
    bp.set_defaults(fn=cmd_dataset_build_pilot)
    di = ds.add_parser("import", help="verify and import a manifest into the database")
    di.add_argument(
        "manifest",
        nargs="?",
        default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json"),
    )
    di.add_argument("--evaluate", action="store_true", help="queue Stage A for imported items")
    di.set_defaults(fn=cmd_dataset_import)
    for sp in (bp, di):
        sp.add_argument("--cache-dir", default=None)
        sp.add_argument("--offline", action="store_true", help="use only the local cache")
    rv = ds.add_parser("revoke", help="revoke a video's content sources and withdraw its evaluations")
    rv.add_argument("video_id")
    rv.add_argument("--reason", required=True)
    rv.set_defaults(fn=cmd_dataset_revoke)
    an = sub.add_parser("annotate", help="human annotation administration").add_subparsers(
        dest="annotate_cmd", required=True
    )
    cr = an.add_parser("create-rater")
    cr.add_argument("pseudonym")
    cr.add_argument("--expertise", nargs="*", default=[])
    cr.add_argument("--role", choices=["rater", "adjudicator"], default="rater")
    q = an.add_parser("qc")
    ex = an.add_parser("export")
    ex.add_argument("--out", required=True)
    for sp in (q, ex):
        sp.add_argument("--dataset", default="real_pilot@v1")
    for sp in (cr, q, ex):
        sp.set_defaults(fn=cmd_annotate)
    br = sub.add_parser("bench-real", help="benchmark on a real licensed dataset against human labels")
    br.add_argument(
        "--manifest", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    )
    br.add_argument("--labels", default=None, help="directory from 'avd annotate export'")
    br.add_argument("--split", choices=["dev", "test"], default="dev")
    br.add_argument(
        "--confirm-test", action="store_true", help="deliberately evaluate the held-out test split"
    )
    br.add_argument("--evaluator", choices=["heuristic", "configured"], default="heuristic")
    br.add_argument("--offline", action="store_true")
    br.add_argument("--seed", type=int, default=0)
    br.add_argument("--n-boot", type=int, default=2000)
    br.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "reports"))
    br.add_argument("--tag", default=None)
    br.set_defaults(fn=cmd_bench_real)
    gt = sub.add_parser("gates", help="release gates lock").add_subparsers(dest="gates_cmd", required=True)
    gl = gt.add_parser("lock")
    gl.add_argument("--statement", required=True)
    gl.add_argument("--new-test-set", action="store_true")
    gt.add_parser("check")
    for sp in gt.choices.values():
        sp.set_defaults(fn=cmd_gates)
    em = sub.add_parser("embeddings", help="embedding model and index maintenance").add_subparsers(
        dest="emb_cmd", required=True
    )
    em.add_parser("fetch-model", help="download and verify the pinned MiniLM ONNX model")
    ri = em.add_parser("reindex", help="re-embed rows whose model, recipe or content changed")
    ri.add_argument("--batch-size", type=int, default=64)
    ri.add_argument("--dry-run", action="store_true")
    for sp in em.choices.values():
        sp.set_defaults(fn=cmd_embeddings)
    from avd.bench.retrieval import add_parser as add_retrieval_parser

    add_retrieval_parser(sub)
    b = sub.add_parser("bench", help="run the evaluation benchmark harness")
    b.add_argument("--dataset", default=str(REPO_ROOT / "benchmarks" / "datasets" / "synthetic_v1"))
    b.add_argument("--fixtures", default=str(REPO_ROOT / "data" / "fixtures"))
    b.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "reports"))
    b.add_argument("--evaluator", choices=["heuristic", "configured"], default="heuristic")
    b.add_argument("--tag", default=None)
    b.set_defaults(fn=cmd_bench)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
