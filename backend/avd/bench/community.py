"""Contributor-run local benchmarks: inspect, pack, verify and import (``avd community …``).

The project has no GPU and no inference budget, so real local-model results come from volunteers
who run ``avd bench-local`` on their own machines. This module turns such a run into a file that
can be shared safely, and checks a received file before it is recorded.

* **Inspect** (dry run) shows exactly what a pack would contain, what is left out and why, what is
  redacted, and any content that blocks sharing. Nothing is written.
* **Pack** keeps only reports and summaries. Run databases, per-item evaluations, media and
  cached model responses never leave the machine, because they contain transcript text. The
  files that are kept are then *checked by content*, not just by name:
  * no transcript text: every 8-word run of every cached pilot transcript is compared against
    the packed text, and quote/transcript fields are refused;
  * no credentials, e-mail addresses or private network addresses;
  * no absolute local paths (they reveal user names); these are redacted, then re-checked;
  * only items whose source permits every external use (``avd dataset rights``);
  * no symlinks.

  Any finding stops the pack: nothing is written.
* **Verify** checks structure and provenance claims, and repeats the content checks on what it
  received. The verdict says what was checked and what cannot be.
* **Import** records an accepted report under ``benchmarks/community/`` with its verification
  verdict. It never overwrites an existing record, and it never extracts executables or anything
  outside the allow-list.

A report remains a **contributor-reported** result. Nothing here can prove that the model ran as
stated: the numbers are self-reported, and the file is not signed. A checksum proves the file was
not altered after packing, not who made it. Treat agreement between independent contributors
(same model digest, same commit, similar numbers) as the evidence, not any single file.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from avd.config import REPO_ROOT

COMMUNITY_DIR = REPO_ROOT / "benchmarks" / "community"
REPORT_FORMAT = "avd-bench-local-report"
PACK_FORMAT = "avd-bench-pack"
MAX_PACK_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_ENTRIES = 500
SHINGLE = 8  # words; a shared 8-word run is treated as copied transcript text
# Files that may leave a contributor's machine. Everything else (run.db, items/, media/, caches)
# stays.
ALLOWED = re.compile(r"^(report\.json|report\.md|runs/[\w.\-]+/(summary\.json|plan\.json|report\.md))$")
DIGEST = re.compile(r"@[0-9a-f]{12}:")
TEXT_KEYS = {
    "quote",
    "quotes",
    "excerpt",
    "transcript",
    "transcript_text",
    "segment_text",
    "segments",
    "text",
}
HUMAN_KEYS = {"human_reviews", "human_labels", "human_ratings", "annotations", "labels", "judgments"}
TRUST_VALUES = {"verified", "human_reviewed", "human_validated", "validated", "official"}
SECRETS = (
    ("API key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}")),
    (
        "credential",
        re.compile(
            r"(?i)(?:api[_-]?key|secret|password|passwd|admin[_-]?token|invite[_-]?code)"
            r"\s*[\"']?\s*[:=]\s*[\"']?(?!null\b|none\b|false\b)[^\s\"',}]{6,}"
        ),
    ),
    ("e-mail address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")),
)
ABS_PATH = re.compile(
    r"(?<![\w.:/])(?:/(?:home|Users|root|tmp|var|mnt|media|private|opt|srv|etc|run|data)(?:/[^\s\"'`<>|]*)?"
    r"|[A-Za-z]:\\[^\s\"'`<>|]*|~[/\\][^\s\"'`<>|]*)"
)
URL_HOST = re.compile(r"https?://\[?([A-Za-z0-9.\-:]+?)\]?(?::\d+)?(?=[/\s\"'`)>]|$)")
PUBLIC_HOSTS = {
    "localhost", "127.0.0.1", "::1", "github.com", "raw.githubusercontent.com", "changelog.com",
    "talkpython.fm", "ollama.com", "huggingface.co",
}  # fmt: skip
REDACTED_PATH = "<local path removed>"
REDACTED_HOST = "<non-loopback address removed>"


class CommunityError(ValueError):
    pass


def default_manifest() -> Path:
    return REPO_ROOT / "benchmarks" / "datasets" / "real_pilot_v1" / "manifest.json"


def default_cache_dir() -> Path:
    from avd.config import get_settings

    return get_settings().dataset_cache_dir


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- content checks


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def transcript_shingles(manifest_path: Path, cache_dir: Path) -> tuple[set[int], int]:
    """Hashes of every 8-word run in every cached transcript of the manifest (all splits).
    Returns the set and the number of transcripts read."""
    from avd.datasets.fetch import SourceFetcher
    from avd.datasets.pilot import load_manifest
    from avd.datasets.sources import SOURCES

    manifest = load_manifest(manifest_path)
    fetcher = SourceFetcher(cache_dir, offline=True)
    out: set[int] = set()
    read = 0
    for item in manifest["items"]:
        try:
            raw = fetcher.get(SOURCES[item["source"]], item["path"], item.get("raw_sha256"))
        except (FileNotFoundError, KeyError):
            continue
        read += 1
        w = _words(raw.decode("utf-8", errors="replace"))
        out.update(hash(" ".join(w[i : i + SHINGLE])) for i in range(max(0, len(w) - SHINGLE + 1)))
    return out, read


def _fields(obj: Any, path: str = "") -> list[tuple[str, str, Any]]:
    """(json path, key, value) for every key of every object, at any depth."""
    out: list[tuple[str, str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((f"{path}.{k}", str(k), v))
            out += _fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _fields(v, f"{path}[{i}]")
    return out


def _has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, list) and any(_has_text(v) or isinstance(v, dict) for v in value)


def scan(name: str, data: bytes, shingles: set[int] | None) -> list[str]:
    """Reasons this file must not be shared (empty when it is fine)."""
    text = data.decode("utf-8", errors="replace")
    findings: list[str] = []
    for label, rx in SECRETS:
        if rx.search(text):
            findings.append(f"{name}: contains a {label}")
    if m := ABS_PATH.search(text):
        findings.append(f"{name}: contains an absolute local path ({m.group(0)[:40]}…)")
    for host in URL_HOST.findall(text):
        if host.lower() not in PUBLIC_HOSTS:
            findings.append(f"{name}: contains a network address that is not loopback ({host})")
    if name.endswith(".json"):
        try:
            doc = json.loads(text)
        except ValueError:
            return [*findings, f"{name}: not valid JSON"]
        for jpath, key, value in _fields(doc):
            k = key.lower()
            if k in TEXT_KEYS and _has_text(value):
                findings.append(f"{name}: has a transcript/quote field {jpath}")
            elif k in HUMAN_KEYS and value not in (None, "", [], 0, False):
                findings.append(f"{name}: claims human labels at {jpath}; contributed runs carry none")
            elif (
                isinstance(value, str)
                and value.lower() in TRUST_VALUES
                and k in ("status", "trust", "verification")
            ):
                findings.append(f"{name}: claims trust status {value!r} at {jpath}")
    if shingles:
        w = _words(text)
        hits = sum(
            1 for i in range(max(0, len(w) - SHINGLE + 1)) if hash(" ".join(w[i : i + SHINGLE])) in shingles
        )
        if hits:
            findings.append(f"{name}: contains transcript text ({hits} matching {SHINGLE}-word runs)")
    return findings


def _redact(obj: Any) -> tuple[Any, int]:
    """Replace absolute paths and non-loopback base URLs in a parsed report."""
    n = 0
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "disk_path" and v:
                out[k], n = REDACTED_PATH, n + 1
                continue
            if k == "base_url" and isinstance(v, str):
                hosts = URL_HOST.findall(v)
                if hosts and hosts[0].lower() not in ("localhost", "127.0.0.1", "::1"):
                    out[k], n = REDACTED_HOST, n + 1
                    continue
            out[k], c = _redact(v)
            n += c
        return out, n
    if isinstance(obj, list):
        items = [_redact(v) for v in obj]
        return [i[0] for i in items], sum(i[1] for i in items)
    if isinstance(obj, str):
        new, c = ABS_PATH.subn(REDACTED_PATH, obj)
        return new, c
    return obj, 0


def _sanitized(name: str, raw: bytes) -> tuple[bytes, int]:
    if name.endswith(".json"):
        doc, n = _redact(json.loads(raw))
        return json.dumps(doc, indent=2).encode(), n
    text, n = ABS_PATH.subn(REDACTED_PATH, raw.decode("utf-8", errors="replace"))
    return text.encode(), n


def _item_ids(contents: dict[str, bytes]) -> set[str]:
    ids: set[str] = set()
    for name, data in contents.items():
        if not name.endswith(".json"):
            continue
        doc = json.loads(data)
        meta = doc.get("meta", {}) if isinstance(doc, dict) else {}
        ids.update(meta.get("items") or [])
        ids.update(
            str(it["video_id"]) for it in doc.get("items", []) if isinstance(it, dict) and it.get("video_id")
        )
        for m in (doc.get("models") or {}).values() if isinstance(doc.get("models"), dict) else []:
            for row in (m.get("heuristic_side_by_side") or {}).get("rows", []):
                if row.get("video_id"):
                    ids.add(str(row["video_id"]))
    return ids


def rights_check(ids: set[str], manifest_path: Path) -> list[str]:
    """Items must be development items whose source permits every external use."""
    from avd.datasets.pilot import load_manifest
    from avd.datasets.rights import pending_uses

    by_id = {it["item_id"]: it for it in load_manifest(manifest_path)["items"]}
    problems = []
    for vid in sorted(ids):
        it = by_id.get(vid)
        if it is None:
            problems.append(f"{vid}: not an item of this repository's pilot manifest")
        elif it["split"] != "dev":
            problems.append(f"{vid}: in the {it['split']} split; only development items may be shared")
        elif pending := pending_uses(it["source"]):
            problems.append(
                f"{vid}: source {it['source']!r} has unconfirmed uses ({', '.join(sorted(pending))})"
            )
    return problems


# ---------------------------------------------------------------- inspect / pack


def _plan(bench_dir: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    included, excluded, problems = [], [], []
    for p in sorted(bench_dir.rglob("*")):
        rel = p.relative_to(bench_dir).as_posix()
        if p.is_symlink():
            problems.append(f"{rel}: is a symbolic link (refused: it could point at private files)")
            continue
        if not p.is_file():
            continue
        if ALLOWED.match(rel):
            included.append(rel)
        else:
            why = (
                "run database (contains transcript text)"
                if ".db" in p.name
                else "per-item evaluation (contains evidence quotes)"
                if "/items/" in f"/{rel}"
                else "media"
                if "/media/" in f"/{rel}"
                else "not on the allow-list"
            )
            excluded.append({"file": rel, "reason": why})
    return included, excluded, problems


def inspect(
    bench_dir: Path, cache_dir: Path | None = None, manifest_path: Path | None = None
) -> dict[str, Any]:
    """Dry run of ``pack``: what would be shared, what is excluded or redacted, and what blocks it."""
    manifest_path = manifest_path or default_manifest()
    if not (bench_dir / "report.json").exists():
        raise CommunityError(f"{bench_dir} has no report.json (run `avd bench-local` first)")
    included, excluded, problems = _plan(bench_dir)
    cache = cache_dir or default_cache_dir()
    shingles, read = transcript_shingles(manifest_path, cache) if cache.exists() else (set(), 0)
    if not read:
        problems.append(
            f"no cached pilot transcripts under {cache}: cannot check for transcript text "
            "(pass --cache-dir, the directory used by `avd bench-local`)"
        )
    contents: dict[str, bytes] = {}
    redactions: dict[str, int] = {}
    for rel in included:
        data, n = _sanitized(rel, (bench_dir / rel).read_bytes())
        contents[rel] = data
        if n:
            redactions[rel] = n
        problems += scan(rel, data, shingles)
    problems += rights_check(_item_ids(contents), manifest_path)
    report = json.loads(contents["report.json"])
    return {
        "ok": not problems,
        "would_include": [
            {"file": f, "bytes": len(contents[f]), "redactions": redactions.get(f, 0)} for f in included
        ],
        "would_exclude": excluded,
        "transcripts_checked": read,
        "blocking_findings": problems,
        "transport_kind": report.get("meta", {}).get("transport_kind"),
        "note": "Nothing was written. `avd community pack` writes exactly the files listed above.",
        "_contents": contents,
    }


def pack(
    bench_dir: Path, out: Path, cache_dir: Path | None = None, manifest_path: Path | None = None
) -> dict[str, Any]:
    """Zip the shareable, redacted files of a ``bench-local`` output directory."""
    plan = inspect(bench_dir, cache_dir, manifest_path)
    if not plan["ok"]:
        raise CommunityError("not packed:\n  " + "\n  ".join(plan["blocking_findings"]))
    contents: dict[str, bytes] = plan["_contents"]
    manifest = {
        "format": PACK_FORMAT,
        "format_version": "1.1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {f: _sha(d) for f, d in contents.items()},
        "excluded": [e["file"] for e in plan["would_exclude"]],
        "transcripts_checked": plan["transcripts_checked"],
        "note": "Run databases, per-item evaluations, media and model caches are excluded: they contain "
        "transcript text. Local paths are redacted.",
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f, d in contents.items():
            z.writestr(f, d)
        z.writestr("PACK.json", json.dumps(manifest, indent=2))
    return {
        "out": str(out),
        "files": len(contents),
        "excluded": len(plan["would_exclude"]),
        "redactions": sum(i["redactions"] for i in plan["would_include"]),
        "sha256": _sha(out.read_bytes()),
    }


# ---------------------------------------------------------------- verify / import


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
    infos = z.infolist()
    if len(infos) > MAX_ENTRIES:
        raise CommunityError(f"too many entries ({len(infos)})")
    if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
        raise CommunityError("pack expands to more than 50 MB")
    contents: dict[str, bytes] = {}
    for info in infos:
        name = info.filename
        if name.startswith("/") or ".." in Path(name).parts or "\\" in name or ":" in name:
            raise CommunityError(f"unsafe path in pack: {name!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise CommunityError(f"symbolic link in pack: {name!r}")
        if info.is_dir():
            continue
        if name != "PACK.json" and not ALLOWED.match(name):
            raise CommunityError(
                f"unexpected file in pack: {name!r} (run databases and media must not be shared)"
            )
        if name in contents:
            raise CommunityError(f"duplicate entry in pack: {name!r}")
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
        unlisted = set(contents) - set(listed) - {"PACK.json"}
        if unlisted:
            raise CommunityError(f"files not in the pack manifest: {sorted(unlisted)}")
    return json.loads(contents["report.json"]), contents


def _commit_known(sha: str) -> bool | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.returncode == 0


def verify(source: Path, manifest_path: Path | None = None, cache_dir: Path | None = None) -> dict[str, Any]:
    """Check a contributed report or pack. Returns a verdict; never raises on a failed check."""
    from avd.datasets.pilot import load_manifest
    from avd.evaluation.prompts import get_prompt

    manifest_path = manifest_path or default_manifest()
    checks: list[dict[str, Any]] = []
    notes: list[str] = []

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
    manifest = load_manifest(manifest_path)
    check(
        "dataset matches this repository",
        meta.get("manifest_items_sha256") == manifest["items_sha256"],
        f"{meta.get('dataset')} {str(meta.get('manifest_items_sha256'))[:12]}",
    )
    sha = str(meta.get("git_commit") or "")
    check("git commit recorded and clean", bool(sha) and meta.get("git_dirty") is False, sha[:12])
    known = _commit_known(sha) if sha else None
    if known is None:
        notes.append("could not check whether the commit exists in this repository (no git here)")
    else:
        check("git commit exists in this repository", known, sha[:12])
    try:
        get_prompt(str(meta.get("prompt")))
        prompt_ok = bool(meta.get("rubric_version"))
    except (KeyError, ValueError):
        prompt_ok = False
    check(
        "prompt and rubric versions recorded and known",
        prompt_ok,
        f"prompt={meta.get('prompt')!r} rubric={meta.get('rubric_version')!r}",
    )
    models = report.get("models") or {}
    check("at least one model", bool(models))
    for name, entry in models.items():
        rt = (entry or {}).get("runtime", {})
        info = (rt.get("models") or {}).get(name) or {}
        fp = info.get("fingerprint", "")
        check(f"{name}: weights digest recorded", bool(DIGEST.search(str(fp))), str(fp))
        check(f"{name}: quantization recorded", bool(info.get("quantization")), str(info.get("quantization")))
        version = str(rt.get("ollama_version") or "")
        check(f"{name}: runtime version recorded", bool(version) and "mock" not in version, version)
        hw = rt.get("hardware") or {}
        check(f"{name}: hardware recorded", bool(hw.get("cpu_model")) and bool(hw.get("ram_total_gb")))
        for ctx, w in (entry.get("windows") or {}).items():
            rates = [w.get(k) for k in ("structured_validity", "citation_correctness", "abstention_rate")]
            sane = all(r is None or 0 <= r <= 1 for r in rates) and w.get("calls", 0) >= 0
            consistent = (w.get("evidence_verified", 0) or 0) <= (w.get("evidence_claimed", 0) or 0)
            check(f"{name} ctx={ctx}: numbers in range and consistent", sane and consistent)
            check(
                f"{name} ctx={ctx}: model calls made",
                (w.get("calls") or 0) > 0 and (w.get("items_evaluated_by_llm") or 0) > 0,
                f"calls={w.get('calls')} items={w.get('items_evaluated_by_llm')}",
            )
            check(
                f"{name} ctx={ctx}: structured-output validity reported",
                w.get("structured_validity") is not None,
            )
            check(
                f"{name} ctx={ctx}: duration recorded",
                (w.get("wall_seconds") or 0) > 0 and (w.get("latency_ms_p50") or 0) > 0,
                f"wall={w.get('wall_seconds')}s p50={w.get('latency_ms_p50')}ms",
            )
            check(f"{name} ctx={ctx}: cost is $0 (local)", (w.get("cost_usd") or 0) == 0)
            if w.get("server_rss_peak_gb") is None and w.get("vram_peak_gb") is None:
                notes.append(f"{name} ctx={ctx}: no memory measurement (sampling is Linux-only)")
    cache = cache_dir or default_cache_dir()
    shingles, read = transcript_shingles(manifest_path, cache) if cache.exists() else (set(), 0)
    if not read:
        notes.append("transcript-text check not run: no pilot cache on this machine")
    findings = [
        f for name, data in contents.items() if name != "PACK.json" for f in scan(name, data, shingles)
    ]
    check(
        "no transcript text, credentials, private addresses or local paths", not findings, "; ".join(findings)
    )
    rights = rights_check(_item_ids(contents), manifest_path)
    check("every item is a shareable development item", not rights, "; ".join(rights))
    accepted = all(c["ok"] for c in checks)
    return {
        "accepted": accepted,
        "status": "contributor_reported" if accepted else "refused",
        "checks": checks,
        "notes": notes,
        "report_sha256": _sha(contents["report.json"]),
        "cannot_verify": [
            "that the model actually ran as reported (self-reported numbers, unsigned file)",
            "the contributor's identity (a checksum proves integrity, not authorship)",
        ],
    }


def import_result(source: Path, contributor: str, dest_root: Path | None = None) -> dict[str, Any]:
    verdict = verify(source)
    if not verdict["accepted"]:
        failed = [c["check"] for c in verdict["checks"] if not c["ok"]]
        raise CommunityError(f"refused: {', '.join(failed)}")
    report, contents = _read(source)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{report['meta']['bench_id']}-{contributor}")[:100].strip(".-")
    root = (dest_root or COMMUNITY_DIR).resolve()
    dest = (root / slug).resolve()
    if root not in dest.parents:
        raise CommunityError("invalid destination")
    if dest.exists():
        raise CommunityError(f"{dest} already exists (records are never overwritten)")
    dest.mkdir(parents=True)
    for name, data in contents.items():
        target = (dest / name).resolve()
        if dest not in target.parents:
            raise CommunityError(f"unsafe path in pack: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    record = {
        **verdict,
        # Always contributor_reported: nothing in a submitted file can raise its trust level.
        "status": "contributor_reported",
        "independently_reproduced": False,
        "human_labels": 0,
        "contributor": contributor,
        "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_file_sha256": _sha(source.read_bytes()),
    }
    (dest / "VERIFICATION.json").write_text(json.dumps(record, indent=2))
    return {"imported_to": str(dest), "status": record["status"]}


def smallest_dev_item(manifest_path: Path | None = None) -> dict[str, Any]:
    from avd.datasets.pilot import load_manifest
    from avd.datasets.rights import pending_uses

    items = [
        it
        for it in load_manifest(manifest_path or default_manifest())["items"]
        if it["split"] == "dev" and not pending_uses(it["source"])
    ]
    if not items:
        raise CommunityError("the manifest has no shareable development items")
    return min(items, key=lambda it: (it.get("words") or 0, it["item_id"]))


def smoke(
    model: str,
    out: Path,
    num_ctx: int | None = None,
    cache_dir: Path | None = None,
    manifest_path: Path | None = None,
    base: Any = None,
    provider_factory: Any = None,
) -> dict[str, Any]:
    """One real evaluation of the shortest shareable development transcript, then a package dry
    run. Use it before committing hours to the full benchmark."""
    from avd.bench import local as bl

    manifest_path = manifest_path or default_manifest()
    item = smallest_dev_item(manifest_path)
    cfg = bl.LocalBenchConfig(
        manifest=manifest_path,
        models=[model],
        out_dir=out,
        repeats=1,
        num_ctx=[num_ctx] if num_ctx else [],
        item_ids=(item["item_id"],),
        probes=False,
        offline=True,
        cache_dir=cache_dir or default_cache_dir(),
        bench_id=f"smoke-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
    )
    report = bl.bench_local(cfg, base, provider_factory)
    entry = report["models"][model]
    window = next(iter(entry["windows"].values()))
    plan = inspect(out, cfg.cache_dir, manifest_path)
    return {
        "item": {"id": item["item_id"], "words": item.get("words"), "title": item.get("title")},
        "transport": report["meta"]["transport_kind"],
        "model": (entry["runtime"]["models"].get(model) or {}).get("fingerprint"),
        "runtime_version": entry["runtime"].get("ollama_version"),
        "accelerator": (entry["runtime"].get("hardware") or {}).get("accelerator"),
        "model_calls": window.get("calls"),
        "structured_validity": window.get("structured_validity"),
        "citation_correctness": window.get("citation_correctness"),
        "abstention_rate": window.get("abstention_rate"),
        "errors": window.get("errors"),
        "wall_seconds": window.get("wall_seconds"),
        "output_tokens_per_second": window.get("output_tokens_per_wall_second"),
        "server_rss_peak_gb": window.get("server_rss_peak_gb"),
        "vram_peak_gb": window.get("vram_peak_gb"),
        "pack_dry_run_ok": plan["ok"],
        "pack_blocking_findings": plan["blocking_findings"],
        "next": "If this looks right, run the full benchmark (docs/volunteer-pilot.md, step 8).",
    }


def add_parser(sub: Any) -> None:
    p = sub.add_parser(
        "community", help="contributor benchmark results: inspect, pack, verify, import"
    ).add_subparsers(dest="community_cmd", required=True)
    ins = p.add_parser("inspect", help="dry run: show what a pack would contain and what blocks it")
    ins.add_argument("bench_dir")
    ins.add_argument("--cache-dir", help="pilot transcript cache used by bench-local")
    ins.set_defaults(fn=_inspect_main)
    pk = p.add_parser("pack", help="zip a bench-local output for sharing (no transcripts)")
    pk.add_argument("bench_dir")
    pk.add_argument("--out", required=True)
    pk.add_argument("--cache-dir", help="pilot transcript cache used by bench-local")
    pk.set_defaults(fn=_pack_main)
    sm = p.add_parser("smoke", help="one real evaluation of the shortest development transcript")
    sm.add_argument("--model", required=True, help="an installed Ollama tag (avd local doctor)")
    sm.add_argument("--num-ctx", type=int, default=None)
    sm.add_argument("--out", default="smoke-run")
    sm.add_argument("--cache-dir", default=None)
    sm.set_defaults(fn=_smoke_main)
    vf = p.add_parser("verify", help="check a contributed report or pack")
    vf.add_argument("file")
    vf.add_argument("--cache-dir")
    vf.set_defaults(
        fn=lambda a: print(
            json.dumps(verify(Path(a.file), cache_dir=Path(a.cache_dir) if a.cache_dir else None), indent=2)
        )
    )
    im = p.add_parser("import", help="record an accepted contribution under benchmarks/community/")
    im.add_argument("file")
    im.add_argument("--contributor", required=True, help="a pseudonym the contributor chose")
    im.set_defaults(fn=_import_main)


def _inspect_main(args: Any) -> None:
    try:
        out = inspect(Path(args.bench_dir), Path(args.cache_dir) if args.cache_dir else None)
    except CommunityError as exc:
        raise SystemExit(str(exc)) from exc
    out.pop("_contents")
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        raise SystemExit(1)


def _smoke_main(args: Any) -> None:
    from avd.bench.evaluate_real import PreflightError

    try:
        out = smoke(
            args.model, Path(args.out), args.num_ctx, Path(args.cache_dir) if args.cache_dir else None
        )
    except (CommunityError, PreflightError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(out, indent=2))


def _pack_main(args: Any) -> None:
    try:
        print(
            json.dumps(
                pack(Path(args.bench_dir), Path(args.out), Path(args.cache_dir) if args.cache_dir else None)
            )
        )
    except CommunityError as exc:
        raise SystemExit(str(exc)) from exc


def _import_main(args: Any) -> None:
    try:
        print(json.dumps(import_result(Path(args.file), args.contributor)))
    except CommunityError as exc:
        raise SystemExit(str(exc)) from exc
