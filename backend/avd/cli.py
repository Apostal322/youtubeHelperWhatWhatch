"""Command-line entry point: ``avd <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _ensure_schema() -> None:
    """Create the schema on a brand-new database (a volunteer's fresh checkout). An existing
    database is left as it is: upgrading it stays an explicit `avd migrate`."""
    from sqlalchemy import inspect

    from avd.db.session import get_engine

    if not inspect(get_engine()).get_table_names():
        _migrate()


def cmd_eval_export(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.portability import export_evaluations, write_export

    with session_scope() as s:
        doc = export_evaluations(
            s,
            video_ids=args.video or None,
            include_quotes=args.include_quotes,
            include_synthetic=not args.no_synthetic,
        )
    digest = write_export(doc, Path(args.out))
    quoted = sum(1 for e in doc["evaluations"] if e["rights"]["quotes_included"])
    print(
        json.dumps(
            {
                "out": args.out,
                "evaluations": len(doc["evaluations"]),
                "with_quotes": quoted,
                "file_sha256": digest,
            }
        )
    )


def cmd_eval_import(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.portability import ImportRefused, import_evaluations

    raw = Path(args.file).read_bytes()
    try:
        with session_scope() as s:
            out = import_evaluations(s, raw, origin=args.origin)
    except ImportRefused as exc:
        raise SystemExit(f"refused: {exc} (nothing imported)") from exc
    print(json.dumps(out))


def cmd_data(args: argparse.Namespace) -> None:
    from avd import datactl

    settings = get_settings()
    try:
        if args.data_cmd == "backup":
            print(json.dumps(datactl.backup_sqlite(settings.database_url, Path(args.out))))
        elif args.data_cmd == "restore":
            print(json.dumps(datactl.restore_sqlite(Path(args.file), settings.database_url)))
        else:
            from avd.db.session import session_scope

            flags = {
                k: getattr(args, k)
                for k in ("evaluations", "model_cache", "preferences", "feedback", "imported")
            }
            if not any(flags.values()):
                raise SystemExit(
                    "choose what to clear: --evaluations --model-cache --preferences --feedback --imported"
                )
            if not args.yes:
                raise SystemExit(
                    f"this permanently deletes {[k for k, v in flags.items() if v]}; re-run with --yes"
                )
            with session_scope() as s:
                print(json.dumps(datactl.clear(s, **flags)))
    except datactl.DataError as exc:
        raise SystemExit(f"refused: {exc}") from exc


def cmd_doctor(args: argparse.Namespace) -> None:
    """One report for setup problems: configuration, database, worker queue, AI provider and
    hardware. Contains no secrets, so it can be pasted into an issue."""
    from sqlalchemy import func, select, text

    from avd.api.ops import _inference_status
    from avd.db.models import PipelineJob
    from avd.db.session import get_sessionmaker
    from avd.llm.local import detect_hardware

    s = get_settings()
    report: dict[str, Any] = {
        "deployment_mode": s.deployment_mode.value,
        "server_mode_problems": s.server_mode_problems(),
        "database": s.database_url.split("://", 1)[0] + "://…",
        "task_backend": s.task_backend.value,
        "redis": "configured" if s.redis_url else "not used",
        "registration": s.registration.value,
        "inference": _inference_status(s, check=True),
        "telemetry": "none",
    }
    try:
        with get_sessionmaker()() as session:
            session.execute(text("SELECT 1"))
            report["queue"] = dict(
                session.execute(select(PipelineJob.status, func.count()).group_by(PipelineJob.status)).all()
            )
        report["database_status"] = "ok"
    except Exception as exc:
        report["database_status"] = f"unavailable ({type(exc).__name__}); run `avd migrate` or `avd start`"
    hw = detect_hardware()
    report["hardware"] = {
        "os": hw.os,
        "cpu": f"{hw.cpu_model} x{hw.cpu_logical}",
        "ram_gb": hw.ram_total_gb,
        "accelerator": hw.accelerator,
        "disk_free_gb": hw.disk_free_gb,
        "ollama_binary": bool(hw.ollama_binary),
    }
    print(json.dumps(report, indent=2, default=str))


def cmd_jobs_cancel(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.workers import queue

    with session_scope() as s:
        print(json.dumps({"cancelled": queue.cancel(s, job_ids=args.ids, job_types=args.types)}))


def cmd_jobs_stats(args: argparse.Namespace) -> None:
    from avd.db.session import session_scope
    from avd.workers import queue

    with session_scope() as s:
        print(json.dumps(queue.stats(s), indent=2))


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


def _check_bind(host: str) -> None:
    from avd.api.guard import is_loopback
    from avd.config import DeploymentMode

    settings = get_settings()
    # Inside a container the API must bind 0.0.0.0; that is allowed only when the operator declared
    # the container bridge as trusted (the guard still refuses every other peer per request).
    container = bool(settings.local_peer_networks)
    if settings.deployment_mode == DeploymentMode.LOCAL and not (
        is_loopback(host) or host == "localhost" or container
    ):
        raise SystemExit(
            f"refused: a local instance binds to loopback only (asked for {host}). For a shared server set "
            "AVD_DEPLOYMENT_MODE=server with AVD_ADMIN_TOKEN, AVD_ALLOWED_HOSTS and a registration policy."
        )
    if problems := settings.server_mode_problems():
        raise SystemExit("refused: unsafe server configuration:\n  - " + "\n  - ".join(problems))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    _check_bind(args.host)
    uvicorn.run("avd.api.app:app", host=args.host, port=args.port, reload=args.reload)


def default_data_dir() -> Path:
    """Per-user data directory for `avd start` (the operator's data stays on their machine)."""
    import os
    import sys

    if env := os.environ.get("AVD_DATA_DIR"):
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ai-video-discovery"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ai-video-discovery"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "ai-video-discovery"


def prepare_local(data_dir: Path, provider: str | None, worker: bool) -> dict[str, Any]:
    """Configure this process for local mode (SQLite in ``data_dir``, embedded worker, no Redis or
    Celery). Explicit AVD_* environment variables still win."""
    import os

    data_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "AVD_DEPLOYMENT_MODE": "local",
        "AVD_DATABASE_URL": f"sqlite:///{data_dir / 'avd.db'}",
        "AVD_MEDIA_CACHE_DIR": str(data_dir / "media"),
        "AVD_DATASET_CACHE_DIR": str(data_dir / "datasets"),
        "AVD_TASK_BACKEND": "inline",
        "AVD_EMBEDDED_WORKER": "true" if worker else "false",
        "AVD_LOG_LEVEL": "WARNING",
    }
    if provider:
        env["AVD_LLM_PROVIDER"] = provider
    applied = {}
    for k, v in env.items():
        if k == "AVD_LLM_PROVIDER" or k not in os.environ:
            os.environ[k] = v
            applied[k] = v
    if os.environ.get("AVD_DEPLOYMENT_MODE") != "local":
        raise SystemExit("refused: `avd start` runs a local instance; use `avd serve` for server mode")
    from avd.db.session import get_engine, get_sessionmaker

    for f in (get_settings, get_engine, get_sessionmaker):
        f.cache_clear()
    return applied


def cmd_start(args: argparse.Namespace) -> None:
    """One command for a personal installation: SQLite, embedded worker, loopback only."""
    import time

    import uvicorn

    t0 = time.monotonic()
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    prepare_local(data_dir, args.provider, worker=not args.no_worker)
    _check_bind(args.host)
    _migrate()
    settings = get_settings()
    if args.demo:
        from avd.db.session import session_scope
        from avd.services import seed_demo

        with session_scope() as s:
            n = seed_demo(s, settings)
        print(f"queued {n} SYNTHETIC demo videos (heuristic baseline; not real videos, not AI)")
    from avd.api.ops import _inference_status

    inf = _inference_status(settings, check=True)
    print(f"AI Video Discovery — local instance\n  data:      {data_dir}")
    print(f"  API:       http://{args.host}:{args.port}  (loopback only)")
    print(f"  worker:    {'embedded' if settings.embedded_worker else 'off'}")
    if inf["provider"] == "heuristic":
        print("  evaluator: heuristic baseline (no AI). Use --provider ollama for local AI.")
    else:
        state = (
            "ready"
            if inf.get("model_installed")
            else f"NOT READY ({inf.get('error') or 'model not installed: avd local pull ' + inf['model']})"
        )
        print(f"  evaluator: {inf['location']} {inf['provider']} {inf['model']}: {state}")
    print(f"  started in {time.monotonic() - t0:.1f}s (before the server); Ctrl+C stops safely")
    uvicorn.run("avd.api.app:create_app", factory=True, host=args.host, port=args.port, log_level="warning")


def cmd_train_ranker(args: argparse.Namespace) -> None:
    """Pairwise LTR from logged impressions + explicit feedback, verified on a *temporal* hold-out
    (later requests), with a user-grouped bootstrap CI. The trained model is only described as an
    improvement when the CI of its accuracy gain over the default ranker excludes zero."""
    import hashlib
    from collections.abc import Sequence
    from datetime import UTC, datetime

    import numpy as np

    from avd.bench.stats import grouped_bootstrap
    from avd.db.session import session_scope
    from avd.recsys.ltr_data import build_labeled_pairs, temporal_split
    from avd.recsys.ranker import DEFAULT_WEIGHTS, LinearRanker, train_pairwise_ranker

    with session_scope() as s:
        pairs = build_labeled_pairs(s)
    if len(pairs) < args.min_pairs:
        sys.exit(f"only {len(pairs)} preference pairs from feedback; need >= {args.min_pairs}")
    train, held = temporal_split(pairs, args.holdout)
    if not train or len(held) < 10:
        sys.exit(f"temporal split too small (train {len(train)}, held-out {len(held)})")
    model = train_pairwise_ranker([(p.preferred, p.other) for p in train])
    baseline = LinearRanker(dict(DEFAULT_WEIGHTS))
    data_hash = hashlib.sha256(
        "|".join(
            f"{p.request_id}:{sorted(p.preferred.items())}:{sorted(p.other.items())}" for p in pairs
        ).encode()
    ).hexdigest()

    def gain(us: Sequence[tuple[float, float, str]]) -> float:
        return float(np.mean([m - b for m, b, _ in us])) if us else float("nan")

    units = [
        (
            float(model.score(p.preferred)[0] > model.score(p.other)[0]),
            float(baseline.score(p.preferred)[0] > baseline.score(p.other)[0]),
            p.user_id,
        )
        for p in held
    ]
    ci = grouped_bootstrap(units, lambda u: u[2], gain, n_boot=1000, seed=0)
    report = {
        "version": model.version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pairs": len(pairs),
        "train_pairs": len(train),
        "heldout_pairs": len(held),
        "split": "temporal by request (later requests held out)",
        "train_until": str(train[-1].shown_at),
        "data_sha256": data_hash,
        "weights": model.weights,
        "heldout_pairwise_accuracy": float(np.mean([u[0] for u in units])),
        "baseline_heldout_pairwise_accuracy": float(np.mean([u[1] for u in units])),
        "accuracy_gain_ci_by_user": ci,
        "improvement_supported": bool(ci["ci_low"] is not None and ci["ci_low"] > 0),
    }
    print(json.dumps(report, indent=2))
    model.save(Path(args.out))
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(report, indent=2))


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

    _ensure_schema()
    with session_scope() as s:
        result = import_manifest(s, Path(args.manifest), _fetcher(args))
        if args.evaluate:
            for vid in result["video_ids"]:
                enqueue_stage_a(s, vid, force=True)
    print(json.dumps({k: v for k, v in result.items() if k != "video_ids"}))


def cmd_dataset_build_holdout(args: argparse.Namespace) -> None:
    from avd.datasets.pilot import build_holdout, load_manifest

    clones = dict(c.split("=", 1) for c in args.clone)
    if "changelog" not in clones:
        sys.exit("--clone changelog=<path> is required")
    pilot = load_manifest(Path(args.pilot))
    manifest = build_holdout({"changelog": Path(clones["changelog"])}, _fetcher(args), pilot, progress=print)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"items": len(manifest["items"]), "items_sha256": manifest["items_sha256"]}))


def cmd_prompts(args: argparse.Namespace) -> None:
    from avd.evaluation.prompts import describe

    print(json.dumps(describe(), indent=2))


def cmd_dataset_rights(args: argparse.Namespace) -> None:
    from avd.datasets.rights import report

    print(json.dumps(report(), indent=2))


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
            from avd.annotation.offline import export_relevance

            data["relevance"] = export_relevance(s)
            for name in ("labels", "repeats", "pairs", "items", "enjoyment", "relevance"):
                with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                    for row in data[name]:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({k: len(v) for k, v in data.items()}))
        elif args.annotate_cmd in ("blind", "unblind"):
            from avd.db.models import Dataset

            ds = s.get(Dataset, args.dataset)
            if ds is None:
                sys.exit(f"unknown dataset {args.dataset}")
            ds.public_blinding = args.annotate_cmd == "blind"
            print(json.dumps({"dataset": ds.id, "public_blinding": ds.public_blinding}))
        elif args.annotate_cmd == "packet":
            from sqlalchemy import select

            from avd.annotation import offline
            from avd.db.models import Rater

            found = s.scalar(select(Rater).where(Rater.pseudonym == args.rater))
            if found is None:
                sys.exit(f"unknown rater {args.rater} (avd annotate create-rater)")
            rater = found
            consent = Path(args.consent_file).read_text(encoding="utf-8")
            try:
                packet, secret, payloads = offline.issue_packet(
                    s, rater, args.dataset, args.items, tuple(args.splits.split(",")), consent_text=consent
                )
            except svc.AnnotationError as exc:
                sys.exit(str(exc))
            out = Path(args.out or f"packet-{rater.pseudonym}-{packet.id[:8]}.html")
            out.write_text(offline.render_packet(packet, rater, payloads, secret, consent), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "packet_id": packet.id,
                        "rater": rater.pseudonym,
                        "items": [p["video_id"] for p in payloads],
                        "calibration": all(p["is_calibration"] for p in payloads),
                        "file": str(out),
                    }
                )
            )
        elif args.annotate_cmd == "import-packet":
            from avd.annotation import offline

            try:
                result = offline.import_return(s, json.loads(Path(args.file).read_text(encoding="utf-8")))
            except (svc.AnnotationError, ValueError) as exc:
                sys.exit(f"refused: {exc}")
            print(json.dumps(result, indent=2))
        elif args.annotate_cmd == "release-packet":
            from avd.annotation import offline

            print(json.dumps({"released_assignments": offline.release_packet(s, args.packet_id)}))


def cmd_bench_real(args: argparse.Namespace) -> None:
    from avd.bench.real import main as real_main

    real_main(args)


def cmd_gates(args: argparse.Namespace) -> None:
    from avd.bench.real import load_gates, lock_gates

    if args.gates_cmd == "lock":
        print(json.dumps(lock_gates(args.statement, new_test_set=args.new_test_set), indent=2))
    elif args.gates_cmd == "status":
        from avd.bench.real import evaluate_gates, load_exposures

        report = json.loads(Path(args.report).read_text())
        meta = report.get("meta", {})
        manifest = Path(args.manifest) if args.manifest else None
        res = evaluate_gates(
            load_gates(),
            report,
            meta.get("split", "dev"),
            meta.get("evaluator", "heuristic"),
            load_exposures(manifest),
        )
        out = {
            "report": args.report,
            "dataset": meta.get("dataset"),
            "split": meta.get("split"),
            "evaluator": res["evaluator"],
            "binding": res["binding"],
            "release_ready": res["release_ready"],
            "counts": res["counts"],
            "by_status": {st: [r["id"] for r in res["results"] if r["status"] == st] for st in res["counts"]},
            "results": res["results"],
        }
        print(json.dumps(out, indent=2))
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
    st = sub.add_parser("start", help="run a personal local instance (SQLite, embedded worker, no Redis)")
    st.add_argument("--data-dir", default=None, help="default: the per-user data directory")
    st.add_argument("--host", default="127.0.0.1")
    st.add_argument("--port", type=int, default=8000)
    st.add_argument(
        "--provider", choices=["heuristic", "ollama", "anthropic", "openai_compatible"], default=None
    )
    st.add_argument("--demo", action="store_true", help="queue the synthetic demo catalog (clearly labelled)")
    st.add_argument("--no-worker", action="store_true", help="API only; run `avd worker` separately")
    st.set_defaults(fn=cmd_start)
    sub.add_parser(
        "doctor", help="diagnose configuration, database, queue, AI provider and hardware"
    ).set_defaults(fn=cmd_doctor)
    ev = sub.add_parser("evaluations", help="portable evaluation export/import").add_subparsers(
        dest="eval_cmd", required=True
    )
    ex = ev.add_parser("export", help="write current evaluations to an avd-evaluation-export file")
    ex.add_argument("--out", required=True)
    ex.add_argument("--video", action="append", help="limit to these video ids")
    ex.add_argument(
        "--include-quotes", action="store_true", help="quote text only where every source permits"
    )
    ex.add_argument("--no-synthetic", action="store_true")
    ex.set_defaults(fn=cmd_eval_export)
    im = ev.add_parser("import", help="validate and store another instance's export as unverified")
    im.add_argument("file")
    im.add_argument(
        "--origin", required=True, help="where the file came from (shown with every imported record)"
    )
    im.set_defaults(fn=cmd_eval_import)
    da = sub.add_parser("data", help="backup, restore and clear local data").add_subparsers(
        dest="data_cmd", required=True
    )
    bk = da.add_parser("backup", help="consistent SQLite backup (PostgreSQL: pg_dump)")
    bk.add_argument("--out", required=True)
    bk.set_defaults(fn=cmd_data)
    rs = da.add_parser("restore", help="restore a SQLite backup into a new database file")
    rs.add_argument("file")
    rs.set_defaults(fn=cmd_data)
    cl = da.add_parser("clear", help="delete selected local data")
    for flag in ("--evaluations", "--model-cache", "--preferences", "--feedback", "--imported"):
        cl.add_argument(flag, action="store_true")
    cl.add_argument("--yes", action="store_true")
    cl.set_defaults(fn=cmd_data)
    jb = sub.add_parser("jobs", help="queue control").add_subparsers(dest="jobs_cmd", required=True)
    jc = jb.add_parser("cancel", help="cancel queued jobs (running jobs finish their attempt)")
    jc.add_argument("--type", action="append", dest="types")
    jc.add_argument("--id", action="append", dest="ids")
    jc.set_defaults(fn=cmd_jobs_cancel)
    jb.add_parser("stats").set_defaults(fn=cmd_jobs_stats)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(fn=cmd_serve)
    t = sub.add_parser("train-ranker")
    t.add_argument("--out", default="ranker.json")
    t.add_argument("--min-pairs", type=int, default=50)
    t.add_argument("--holdout", type=float, default=0.2, help="fraction of the latest requests held out")
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
    bh = ds.add_parser(
        "build-holdout", help="build the sealed hold-out manifest (unseen episodes of test shows)"
    )
    bh.add_argument("--clone", action="append", default=[], help="changelog=<local clone>")
    bh.add_argument(
        "--pilot", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json")
    )
    bh.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "datasets" / "real_holdout_v2"))
    bh.add_argument("--cache-dir", default=None)
    bh.add_argument("--offline", action="store_true")
    bh.set_defaults(fn=cmd_dataset_build_holdout)
    ds.add_parser("rights", help="use-time rights review: facts, assumptions, pending uses").set_defaults(
        fn=cmd_dataset_rights
    )
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
    bl = an.add_parser(
        "blind", help="withhold model assessments of the dataset's items from public endpoints"
    )
    ub = an.add_parser("unblind", help="annotation closed: publish model assessments of the dataset's items")
    pk = an.add_parser("packet", help="offline rating file for one rater (no server needed)")
    pk.add_argument("--rater", required=True, help="pseudonym from create-rater")
    pk.add_argument("--items", type=int, default=3, help="dev items per packet (calibration comes first)")
    pk.add_argument("--splits", default="dev", help="dev, or dev,test once the dev round is analysed")
    pk.add_argument("--out", default=None)
    pk.add_argument(
        "--consent-file",
        required=True,
        help="consent text approved by the project owner (template: docs/templates/rating-consent.txt)",
    )
    ip = an.add_parser("import-packet", help="validate and record a returned ratings file")
    ip.add_argument("file")
    rp = an.add_parser("release-packet", help="withdraw a packet's unreturned assignments")
    rp.add_argument("packet_id")
    for sp in (q, ex, bl, ub, pk):
        sp.add_argument("--dataset", default="real_pilot@v1")
    for sp in (cr, q, ex, bl, ub, pk, ip, rp):
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
    gs = gt.add_parser("status", help="classify every gate for a benchmark report (machine-readable)")
    gs.add_argument("--report", required=True, help="a bench-real JSON report")
    gs.add_argument("--manifest", default=None, help="dataset manifest (to apply its exposure ledger)")
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
    from avd.bench.recs_real import add_diagnostic_parser
    from avd.bench.recs_real import add_parser as add_recs_parser

    add_recs_parser(sub)
    from avd.bench.evaluate_real import add_parser as add_evaluate_real_parser
    from avd.bench.throughput import add_parser as add_throughput_parser

    add_throughput_parser(sub)
    add_evaluate_real_parser(sub)
    add_diagnostic_parser(sub)
    from avd.bench.recs_real import add_stages_parser
    from avd.bench.relevance import add_parser as add_relevance_parser

    add_stages_parser(sub)
    add_relevance_parser(sub)
    from avd.bench.local import add_parser as add_bench_local_parser
    from avd.llm.local import add_parser as add_local_parser

    add_local_parser(sub)
    add_bench_local_parser(sub)
    from avd.bench.community import add_parser as add_community_parser

    add_community_parser(sub)
    from avd.bench.evidence_review import add_parser as add_evidence_review_parser

    add_evidence_review_parser(sub)
    sub.add_parser("prompts", help="list registered evaluation prompts and their metadata").set_defaults(
        fn=cmd_prompts
    )
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
