"""Import a dataset manifest into the database, and revoke items.

For each manifest item the importer re-fetches (or reads from cache) the exact bytes at the pinned
commit, verifies the raw checksum, re-parses and re-normalises, and verifies the normalised content
hash — so parser drift or a changed upstream file fails closed instead of silently changing the
dataset. Rights and provenance are stored on the ``ContentSource`` row.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.content.models import SourceType
from avd.content.processing import normalize_transcript
from avd.content.store import add_transcript_source
from avd.datasets.fetch import SourceFetcher
from avd.datasets.pilot import load_manifest
from avd.datasets.sources import SOURCES, parse
from avd.db.models import ContentSource, Dataset, DatasetItem, Evaluation, Video
from avd.ingest import upsert_channel


class ManifestError(ValueError):
    pass


def import_manifest(session: Session, manifest_path: Path, fetcher: SourceFetcher) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset_id = f"{manifest['name']}@{manifest['version']}"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ds = session.get(Dataset, dataset_id)
    if ds is None:
        ds = Dataset(
            id=dataset_id,
            name=manifest["name"],
            version=manifest["version"],
            description=manifest["description"],
            manifest_sha256=manifest_sha,
            split_strategy=manifest["split"]["strategy"],
            split_seed=manifest["split"]["seed"],
        )
        session.add(ds)
    elif ds.manifest_sha256 != manifest_sha:
        raise ManifestError(
            f"{dataset_id} already imported from a different manifest ({ds.manifest_sha256[:12]}); "
            "bump the dataset version instead of mutating it"
        )
    stats = {"items": 0, "imported": 0, "already_present": 0}
    imported: list[str] = []
    for item in manifest["items"]:
        stats["items"] += 1
        source = SOURCES[item["source"]]
        rights = manifest["sources"][item["source"]]
        if not rights.get("license") or not rights.get("authorization_basis"):
            raise ManifestError(f"{item['item_id']}: missing licence or authorisation basis")
        if rights["commit"] != source.commit:
            raise ManifestError(
                f"{item['item_id']}: manifest commit {rights['commit']} != pinned {source.commit}"
            )
        raw = fetcher.get(source, item["path"], expected_sha256=item["raw_sha256"])
        transcript, parse_report = parse(source, item["path"], raw.decode("utf-8"))
        normalized, _ = normalize_transcript(transcript)
        if normalized.content_hash() != item["content_hash"]:
            raise ManifestError(
                f"{item['item_id']}: normalised content hash changed (parser drift?) "
                f"{normalized.content_hash()} != {item['content_hash']}"
            )
        channel_id = f"lic:{item['group_key']}"
        upsert_channel(session, channel_id, item["show"])
        video = session.get(Video, item["item_id"])
        if video is None:
            video = Video(id=item["item_id"], channel_id=channel_id, title=item["title"])
            session.add(video)
        video.channel_id = channel_id
        video.title = item["title"]
        video.description = ""
        video.language = item["language"]
        video.duration_seconds = int(item["transcript_duration_seconds"]) or None
        video.platform = item["platform"]
        video.title_source = item["title_source"]
        video.external_url = item["external_url"]
        video.is_synthetic = False
        video.discovery_source = f"dataset:{dataset_id}"
        video.metadata_fetched_at = datetime.now(UTC)
        if video.status == "discovered":
            video.status = "metadata_ready"
        session.flush()
        row, created = add_transcript_source(
            session,
            video_id=video.id,
            transcript=normalized,
            source_type=SourceType.LICENSED_DATASET,
            authorization_basis=rights["authorization_basis"],
            license=rights["license"],
        )
        if created:
            row.owner = rights["owner"]
            row.attribution = rights["attribution"]
            row.provenance_url = item["provenance_url"]
            row.retrieved_at = datetime.now(UTC)
            row.raw_sha256 = item["raw_sha256"]
            row.permitted_processing = rights["permitted_processing"]
            row.permitted_storage = rights["permitted_storage"][:120]
            row.restrictions = rights["restrictions"]
            row.revocation_policy = rights["revocation_policy"]
            row.preprocessing = {
                "parser": parse_report,
                "normalization": item["normalization"],
                "commit": source.commit,
                "path": item["path"],
            }
            stats["imported"] += 1
            imported.append(video.id)
        else:
            stats["already_present"] += 1
        di = session.scalar(
            select(DatasetItem).where(DatasetItem.dataset_id == dataset_id, DatasetItem.video_id == video.id)
        )
        if di is None:
            session.add(
                DatasetItem(
                    dataset_id=dataset_id,
                    video_id=video.id,
                    split=item["split"],
                    group_key=item["group_key"],
                    stratum=item["stratum"],
                    is_calibration_example=item["split"] == "calibration",
                )
            )
    session.flush()
    return {"dataset": dataset_id, **stats, "video_ids": imported}


def revoke_video_sources(session: Session, video_id: str, reason: str) -> dict[str, int]:
    """Honour a takedown/revocation: revoke the video's content sources and withdraw evaluations
    derived from them (they stop being current; history stays for audit)."""
    now = datetime.now(UTC)
    rows = session.scalars(
        select(ContentSource).where(ContentSource.video_id == video_id, ContentSource.status == "active")
    ).all()
    for r in rows:
        r.status = "revoked"
        r.revoked_at = now
        r.preprocessing = {**(r.preprocessing or {}), "revocation_reason": reason}
    current = session.scalars(
        select(Evaluation).where(Evaluation.video_id == video_id, Evaluation.is_current.is_(True))
    ).all()
    for ev in current:
        ev.is_current = False
    video = session.get(Video, video_id)
    if video is not None:
        video.status = "content_unavailable"
        video.available_modalities = []
    session.flush()
    return {"sources_revoked": len(rows), "evaluations_withdrawn": len(current)}
