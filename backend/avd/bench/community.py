"""Contributor-run local benchmarks: pack, verify and import (``avd community pack|verify|import``).

The project has no GPU and no inference budget, so real local-model results come from volunteers
who run ``avd bench-local`` on their own machines. This module turns such a run into a file that
can be shared safely, and checks a received file before it is recorded.

* **Pack** keeps only reports and summaries. It never includes run databases, media or cached
  model responses, because those contain transcript text.
* **Verify** checks structure and provenance claims:
  * real transport, not a mock;
  * model digests present;
  * the dev split of a known dataset whose manifest matches this repository;
  * internally consistent numbers;
  * a clean git commit.

  The verdict says what was checked and what cannot be.
* **Import** records an accepted report under ``benchmarks/community/`` with its verification
  verdict.

A report remains a **contributor-reported** result. Nothing here can prove that the model ran as
stated: the numbers are self-reported, and the file is not signed. Treat agreement between
independent contributors (same model digest, same commit, similar numbers) as the evidence, not
any single file.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from avd.config import REPO_ROOT

COMMUNITY_DIR = REPO_ROOT / "benchmarks" / "community"
REPORT_FORMAT = "avd-bench-local-report"
PACK_FORMAT = "avd-bench-pack"
MAX_PACK_BYTES = 20 * 1024 * 1024
# Files that may leave a contributor's machine. Everything else (run.db, media/, caches) stays.
ALLOWED = re.compile(r"^(report\.json|report\.md|runs/[\w.\-]+/(summary\.json|plan\.json|report\.md))$")
DIGEST = re.compile(r"@[0-9a-f]{12}:")


class CommunityError(ValueError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pack(bench_dir: Path, out: Path) -> dict[str, Any]:
    """Zip the shareable files of a ``bench-local`` output directory."""
    if not (bench_dir / "report.json").exists():
        raise CommunityError(f"{bench_dir} has no report.json (run `avd bench-local` first)")
    files = sorted(
        p.relative_to(bench_dir).as_posix()
        for p in bench_dir.rglob("*")
        if p.is_file() and ALLOWED.match(p.relative_to(bench_dir).as_posix())
    )
    skipped = sorted(
        p.relative_to(bench_dir).as_posix()
        for p in bench_dir.rglob("*")
        if p.is_file() and not ALLOWED.match(p.relative_to(bench_dir).as_posix())
    )
    manifest = {
        "format": PACK_FORMAT,
        "format_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {f: _sha((bench_dir / f).read_bytes()) for f in files},
        "excluded": skipped,
        "note": "Run databases, media and model caches are excluded: they contain transcript text.",
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(bench_dir / f, f)
        z.writestr("PACK.json", json.dumps(manifest, indent=2))
    return {"out": str(out), "files": len(files), "excluded": len(skipped), "sha256": _sha(out.read_bytes())}


def _read(source: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    raw = source.read_bytes()
    if len(raw) > MAX_PACK_BYTES:
        raise CommunityError("file larger than 20 MB")
    if source.suffix == ".json":
        return json.loads(raw), {"report.json": raw}
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise CommunityError("not a zip or json file") from exc
    contents: dict[str, bytes] = {}
    for info in z.infolist():
        name = info.filename
        if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
            raise CommunityError(f"unsafe path in pack: {name!r}")
        if name != "PACK.json" and not ALLOWED.match(name):
            raise CommunityError(
                f"unexpected file in pack: {name!r} (run databases and media must not be shared)"
            )
        if info.file_size > MAX_PACK_BYTES:
            raise CommunityError(f"{name} is too large")
        contents[name] = z.read(info)
    if "report.json" not in contents:
        raise CommunityError("pack has no report.json")
    if "PACK.json" in contents:
        listed = json.loads(contents["PACK.json"]).get("files", {})
        for name, digest in listed.items():
            if name not in contents or _sha(contents[name]) != digest:
                raise CommunityError(f"{name} does not match the pack manifest")
    return json.loads(contents["report.json"]), contents


def verify(source: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    """Check a contributed report or pack. Returns a verdict; never raises on a failed check."""
    from avd.datasets.pilot import load_manifest

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    try:
        report, contents = _read(source)
    except (CommunityError, ValueError) as exc:
        return {
            "accepted": False,
            "status": "refused",
            "checks": [{"check": "readable", "ok": False, "detail": str(exc)}],
        }
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    check(
        "format",
        report.get("format") == REPORT_FORMAT and str(report.get("format_version", "")).startswith("1."),
    )
    real = meta.get("transport_kind") == "real" and "NOT real" not in str(meta.get("transport", ""))
    check("real inference (not a mock)", real, str(meta.get("transport")))
    check("development split only", meta.get("split") == "dev", f"split={meta.get('split')!r}")
    manifest = load_manifest(
        manifest_path or REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json"
    )
    check(
        "dataset matches this repository",
        meta.get("manifest_items_sha256") == manifest["items_sha256"],
        f"{meta.get('dataset')} {str(meta.get('manifest_items_sha256'))[:12]}",
    )
    check("git commit recorded and clean", bool(meta.get("git_commit")) and meta.get("git_dirty") is False)
    models = report.get("models") or {}
    check("at least one model", bool(models))
    for name, entry in models.items():
        rt = (entry or {}).get("runtime", {})
        fp = ((rt.get("models") or {}).get(name) or {}).get("fingerprint", "")
        check(f"{name}: weights digest recorded", bool(DIGEST.search(str(fp))), str(fp))
        check(f"{name}: hardware recorded", bool((rt.get("hardware") or {}).get("cpu_model")))
        for ctx, w in (entry.get("windows") or {}).items():
            rates = [w.get(k) for k in ("structured_validity", "citation_correctness", "abstention_rate")]
            sane = all(r is None or 0 <= r <= 1 for r in rates) and w.get("calls", 0) >= 0
            consistent = (w.get("evidence_verified", 0) or 0) <= (w.get("evidence_claimed", 0) or 0)
            check(f"{name} ctx={ctx}: numbers in range and consistent", sane and consistent)
            check(f"{name} ctx={ctx}: cost is $0 (local)", (w.get("cost_usd") or 0) == 0)
    accepted = all(c["ok"] for c in checks)
    return {
        "accepted": accepted,
        "status": "contributor_reported" if accepted else "refused",
        "checks": checks,
        "report_sha256": _sha(contents["report.json"]),
        "cannot_verify": [
            "that the model actually ran as reported (self-reported numbers, unsigned file)",
            "the contributor's identity",
        ],
    }


def import_result(source: Path, contributor: str, dest_root: Path | None = None) -> dict[str, Any]:
    verdict = verify(source)
    if not verdict["accepted"]:
        failed = [c["check"] for c in verdict["checks"] if not c["ok"]]
        raise CommunityError(f"refused: {', '.join(failed)}")
    report, contents = _read(source)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{report['meta']['bench_id']}-{contributor}")[:100]
    dest = (dest_root or COMMUNITY_DIR) / slug
    if dest.exists():
        raise CommunityError(f"{dest} already exists")
    dest.mkdir(parents=True)
    for name, data in contents.items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    record = {
        **verdict,
        "contributor": contributor,
        "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_file_sha256": _sha(source.read_bytes()),
    }
    (dest / "VERIFICATION.json").write_text(json.dumps(record, indent=2))
    return {"imported_to": str(dest), "status": record["status"]}


def add_parser(sub: Any) -> None:
    p = sub.add_parser(
        "community", help="contributor benchmark results: pack, verify, import"
    ).add_subparsers(dest="community_cmd", required=True)
    pk = p.add_parser("pack", help="zip a bench-local output for sharing (no transcripts)")
    pk.add_argument("bench_dir")
    pk.add_argument("--out", required=True)
    pk.set_defaults(fn=lambda a: print(json.dumps(pack(Path(a.bench_dir), Path(a.out)))))
    vf = p.add_parser("verify", help="check a contributed report or pack")
    vf.add_argument("file")
    vf.set_defaults(fn=lambda a: print(json.dumps(verify(Path(a.file)), indent=2)))
    im = p.add_parser("import", help="record an accepted contribution under benchmarks/community/")
    im.add_argument("file")
    im.add_argument("--contributor", required=True, help="a pseudonym the contributor chose")
    im.set_defaults(fn=_import_main)


def _import_main(args: Any) -> None:
    try:
        print(json.dumps(import_result(Path(args.file), args.contributor)))
    except CommunityError as exc:
        raise SystemExit(str(exc)) from exc
