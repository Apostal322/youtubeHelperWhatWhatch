"""Build the real-world pilot dataset manifest (``real_pilot_v1``).

Selection is deterministic given the seed and the pinned commits:

1. Enumerate candidate files per group (show) from a local clone at the pinned commit
   (``git ls-tree``), excluding out-of-scope prefixes.
2. Shuffle each group's sorted candidate list with ``random.Random(f"{seed}:{group}")``.
3. Fetch candidates in that order and keep the first ones that pass eligibility until the group's
   quota is met. Every rejection and its reason is logged in the manifest.
4. Assign whole groups (shows) to splits — calibration / dev / test — so no creator appears in two
   splits (leakage-safe). Groups are taken in seeded order and each goes to the split furthest
   below its target fraction.

Nothing here assigns quality labels. Labels come only from human raters via the annotation
workflow (``avd.annotation``).
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from avd.content.processing import normalize_transcript, validate_transcript
from avd.datasets.fetch import SourceFetcher, sha256_bytes
from avd.datasets.sources import CHANGELOG, SOURCES, TALKPYTHON, LicensedSource, item_metadata, parse

DATASET_NAME = "real_pilot"
DATASET_VERSION = "v1"
DEFAULT_SEED = 20261015

# Per-group quotas (items). Small shows get fewer items; total 30.
QUOTAS: dict[tuple[str, str], int] = {
    ("changelog", "podcast"): 3,
    ("changelog", "jsparty"): 3,
    ("changelog", "gotime"): 3,
    ("changelog", "practicalai"): 3,
    ("changelog", "shipit"): 2,
    ("changelog", "friends"): 2,
    ("changelog", "founderstalk"): 2,
    ("changelog", "brainscience"): 3,
    ("changelog", "backstage"): 2,
    ("talkpython", "youtube_transcripts"): 7,
}

SPLIT_FRACTIONS = {"calibration": 0.1, "dev": 0.3, "test": 0.6}

# Topic area per group: descriptive metadata for stratification, not a quality label.
TOPIC_AREAS = {
    "changelog:podcast": "software engineering (general)",
    "changelog:jsparty": "web / JavaScript",
    "changelog:gotime": "Go programming",
    "changelog:practicalai": "AI / machine learning",
    "changelog:shipit": "infrastructure / operations",
    "changelog:friends": "tech culture (casual conversation)",
    "changelog:founderstalk": "business / entrepreneurship",
    "changelog:brainscience": "psychology / neuroscience",
    "changelog:backstage": "media production (behind the scenes)",
    "talkpython": "Python programming",
}


@dataclass(frozen=True)
class Eligibility:
    min_words: int = 3000
    max_words: int = 20000
    min_anchors: int = 15  # source timestamps (Changelog); VTT cues are always dense
    min_anchors_per_1k_words: float = 1.0
    language: str = "en"


@dataclass
class SelectionLog:
    examined: int = 0
    rejected: list[dict[str, str]] = field(default_factory=list)


def list_candidates(source: LicensedSource, clone_dir: Path, group: str) -> list[str]:
    """Paths at the pinned commit (requires the commit to exist in the local clone)."""
    out = subprocess.run(
        ["git", "-C", str(clone_dir), "ls-tree", "-r", "--name-only", source.commit, f"{group}/"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()  # paths may contain spaces
    suffix = ".md" if source.key == "changelog" else ".vtt"
    return sorted(
        p for p in out if p.endswith(suffix) and not any(p.startswith(x) for x in source.excluded_prefixes)
    )


def _duration_bucket(seconds: float) -> str:
    if seconds < 30 * 60:
        return "under_30m"
    if seconds < 60 * 60:
        return "30_60m"
    return "over_60m"


def _length_bucket(words: int) -> str:
    if words < 6000:
        return "3k_6k_words"
    if words < 10000:
        return "6k_10k_words"
    return "10k_plus_words"


def describe_item(source: LicensedSource, path: str, raw: bytes) -> tuple[dict[str, Any], str | None]:
    """Parse + normalise one file. Returns (manifest item, rejection reason or None)."""
    elig = Eligibility()
    text = raw.decode("utf-8")
    transcript, parse_report = parse(source, path, text)
    normalized, norm = normalize_transcript(transcript)
    report = validate_transcript(normalized)
    meta = item_metadata(source, path)
    # Source timing points: inline timestamps (Changelog) or caption cues (WebVTT).
    anchors = int(str(parse_report.get("anchors", parse_report.get("cues", report.segments))))
    density = anchors / max(1, report.words) * 1000
    item: dict[str, Any] = {
        "item_id": _item_id(source, path),
        "source": source.key,
        "path": path,
        "raw_url": source.raw_url(path),
        "provenance_url": source.blob_url(path),
        "raw_sha256": sha256_bytes(raw),
        "content_hash": normalized.content_hash(),
        **meta,
        "language": report.language.language,
        "words": report.words,
        "segments": report.segments,
        "transcript_duration_seconds": round(report.duration_seconds, 1),
        "timing": parse_report["timing"],
        "source_anchors": anchors,
        "anchors_per_1k_words": round(density, 2),
        "normalization": {
            "removed_empty": norm.removed_empty,
            "merged_duplicates": norm.merged_duplicates,
            "fixed_overlaps": norm.fixed_overlaps,
            "reordered": norm.reordered,
        },
        "validation_issues": report.issues,
    }
    reason = None
    if report.language.language != elig.language:
        reason = f"language={report.language.language}"
    elif not elig.min_words <= report.words <= elig.max_words:
        reason = f"words={report.words} outside [{elig.min_words}, {elig.max_words}]"
    elif source.key == "changelog" and anchors < elig.min_anchors:
        reason = f"source_anchors={anchors} < {elig.min_anchors}"
    elif source.key == "changelog" and density < elig.min_anchors_per_1k_words:
        reason = f"anchors_per_1k_words={density:.2f} < {elig.min_anchors_per_1k_words}"
    elif report.non_monotonic or report.overlapping:
        reason = "timestamp_anomalies"
    return item, reason


def _item_id(source: LicensedSource, path: str) -> str:
    name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if source.key == "changelog":
        group = path.split("/", 1)[0]
        return f"chl_{group}_{name.rsplit('-', 1)[-1]}"[:32]
    return f"tpy_{name.split('-', 1)[0]}"


def assign_splits(
    groups: dict[str, int], seed: int, fractions: dict[str, float] = SPLIT_FRACTIONS
) -> dict[str, str]:
    """Greedy group-to-split assignment in seeded order; returns group -> split."""
    order = sorted(groups)
    random.Random(f"{seed}:splits").shuffle(order)
    total = sum(groups.values())
    filled = dict.fromkeys(fractions, 0)
    out: dict[str, str] = {}
    for g in order:
        # Pick the split with the largest remaining deficit (ties broken by fraction order).
        split = max(fractions, key=lambda s: (fractions[s] * total - filled[s], -list(fractions).index(s)))
        out[g] = split
        filled[split] += groups[g]
    return out


def build_manifest(
    clones: dict[str, Path],
    fetcher: SourceFetcher,
    seed: int = DEFAULT_SEED,
    quotas: dict[tuple[str, str], int] = QUOTAS,
    progress: Callable[[str], None] | None = None,
    exclude_paths: frozenset[str] = frozenset(),
    **finalize: Any,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for (source_key, group), quota in quotas.items():
        source = SOURCES[source_key]
        candidates = [
            p
            for p in list_candidates(source, clones[source_key], group)
            if f"{source_key}:{p}" not in exclude_paths
        ]
        random.Random(f"{seed}:{source_key}:{group}").shuffle(candidates)
        log = SelectionLog()
        kept = 0
        for path in candidates:
            if kept >= quota:
                break
            log.examined += 1
            raw = fetcher.get(source, path)
            item, reason = describe_item(source, path, raw)
            if reason:
                log.rejected.append({"path": path, "reason": reason})
                continue
            items.append(item)
            kept += 1
            if progress:
                progress(f"{item['item_id']}: {item['words']} words, {item['segments']} segments")
        selection[f"{source_key}:{group}"] = {
            "candidates": len(candidates),
            "quota": quota,
            "selected": kept,
            "examined": log.examined,
            "rejected": log.rejected,
        }
    return finalize_manifest(items, selection, seed, quotas, **finalize)


def finalize_manifest(
    items: list[dict[str, Any]],
    selection: dict[str, Any],
    seed: int,
    quotas: dict[tuple[str, str], int] = QUOTAS,
    forced_split: str | None = None,
    name: str = DATASET_NAME,
    version: str = DATASET_VERSION,
    description: str | None = None,
) -> dict[str, Any]:
    """Assign group splits and strata, attach rights, and compute the items digest. A
    ``forced_split`` puts every item in one split (a sealed hold-out set)."""
    group_sizes: dict[str, int] = {}
    for it in items:
        group_sizes[it["group_key"]] = group_sizes.get(it["group_key"], 0) + 1
    splits = dict.fromkeys(group_sizes, forced_split) if forced_split else assign_splits(group_sizes, seed)
    for it in items:
        it["split"] = splits[it["group_key"]]
        it["stratum"] = {
            "group": it["group_key"],
            "topic_area": TOPIC_AREAS.get(it["group_key"], "unknown"),
            "expected_format": "interview / discussion podcast",
            "duration_bucket": _duration_bucket(it["transcript_duration_seconds"]),
            "length_bucket": _length_bucket(it["words"]),
            "timing_granularity": "anchor_span" if it["timing"] == "source_anchor_spans" else "cue",
        }
    items.sort(key=lambda it: it["item_id"])
    manifest: dict[str, Any] = {
        "name": name,
        "version": version,
        "description": description
        or (
            "Pilot of real, licensed long-form spoken content (podcast/interview transcripts published "
            "by their creators) for human-grounded validation of transcript-based quality assessment. "
            "Contains no quality labels; labels come from blind human annotation."
        ),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "selection": {
            "procedure": "seeded shuffle per group, first eligible items up to quota (avd.datasets.pilot)",
            "eligibility": Eligibility().__dict__,
            "quotas": {f"{s}:{g}": q for (s, g), q in quotas.items()},
            "per_group": selection,
        },
        "split": {
            "strategy": "sealed_holdout" if forced_split else "group_by_show",
            "unit": "group_key (one show / creator)",
            "fractions": SPLIT_FRACTIONS,
            "seed": seed,
            "assignment": dict(sorted(splits.items())),
        },
        "sources": {
            key: {
                "repo_url": src.repo_url,
                "commit": src.commit,
                "license": src.rights.license,
                "license_url": src.rights.license_url,
                "owner": src.rights.owner,
                "attribution": src.rights.attribution,
                "authorization_basis": src.rights.authorization_basis,
                "permitted_processing": list(src.rights.permitted_processing),
                "permitted_storage": src.rights.permitted_storage,
                "restrictions": list(src.rights.restrictions),
                "revocation_policy": src.rights.revocation_policy,
                "excluded_prefixes": list(src.excluded_prefixes),
                "legal_review_notes": src.rights.legal_review_notes,
            }
            for key, src in (("changelog", CHANGELOG), ("talkpython", TALKPYTHON))
        },
        "items": items,
    }
    manifest["items_sha256"] = items_digest(items)
    return manifest


def items_digest(items: list[dict[str, Any]]) -> str:
    """Identity of the dataset contents: item ids, checksums and split assignment."""
    canon = [
        [
            it["item_id"],
            it["source"],
            it["path"],
            it["raw_sha256"],
            it["content_hash"],
            it["split"],
            it["group_key"],
        ]
        for it in sorted(items, key=lambda x: x["item_id"])
    ]
    return hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    digest = items_digest(manifest["items"])
    if digest != manifest.get("items_sha256"):
        raise ValueError(f"manifest items digest mismatch: {digest} != {manifest.get('items_sha256')}")
    return manifest


# Sealed hold-out (Phase 3): new episodes of the shows that form real_pilot_v1's test split, so the
# show-level separation from dev holds. None of these items was inspected while genre v2, the
# injection detector or the embedding default were developed (those saw the v1 test split).
HOLDOUT_QUOTAS: dict[tuple[str, str], int] = {
    ("changelog", "podcast"): 4,
    ("changelog", "jsparty"): 4,
    ("changelog", "brainscience"): 4,
    ("changelog", "shipit"): 4,
}
HOLDOUT_SEED = 20261019


def build_holdout(
    clones: dict[str, Path],
    fetcher: SourceFetcher,
    pilot_manifest: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    exclude = frozenset(f"{it['source']}:{it['path']}" for it in pilot_manifest["items"])
    return build_manifest(
        clones,
        fetcher,
        seed=HOLDOUT_SEED,
        quotas=HOLDOUT_QUOTAS,
        progress=progress,
        exclude_paths=exclude,
        forced_split="test",
        name="real_holdout",
        version="v2",
        description=(
            "Sealed confirmatory hold-out: unseen episodes of the real_pilot_v1 test shows (Changelog, "
            "CC BY-SA 4.0). Build once; evaluate only with a locked candidate and locked gates."
        ),
    )
