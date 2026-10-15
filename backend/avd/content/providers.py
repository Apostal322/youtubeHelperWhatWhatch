"""Pluggable content-source providers.

Discovery (finding videos) is separate from content acquisition (obtaining authorized material to
evaluate). A provider answers two questions:

* ``check`` (Stage A, cheap): which modalities could this provider supply for the video?
* ``acquire`` (Stage B): fetch and persist the material as ``ContentSource`` rows.

New authorized integrations (e.g. a partner transcript feed, a licensed corpus API) implement this
protocol and are registered in ``build_providers`` — the evaluation engine is unchanged.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from avd.content.models import SourceType
from avd.content.parsing import parse_timed_text
from avd.content.store import add_transcript_source, available_modalities
from avd.db.models import Video
from avd.evaluation.schema import Modality
from avd.logging import get_logger
from avd.youtube.client import YouTubeCaptionsClient

log = get_logger(__name__)


class ContentProvider(Protocol):
    name: str

    def check(self, session: Session, video: Video) -> set[Modality]: ...

    def acquire(self, session: Session, video: Video) -> int: ...


class StoredContentProvider:
    """Content already persisted: user submissions, licensed imports, synthetic fixtures."""

    name = "stored"

    def check(self, session: Session, video: Video) -> set[Modality]:
        return available_modalities(session, video.id)

    def acquire(self, session: Session, video: Video) -> int:
        return 0


class CreatorTokenStore(Protocol):
    """Resolves a channel owner's OAuth access token, if that creator connected their channel."""

    def access_token_for_channel(self, channel_id: str) -> str | None: ...


class NoCreatorTokens:
    def access_token_for_channel(self, channel_id: str) -> str | None:
        return None


class CreatorCaptionsProvider:
    """Captions for channels whose owners granted OAuth access (requires a verified OAuth app)."""

    name = "youtube_creator_captions"

    def __init__(self, client: YouTubeCaptionsClient, tokens: CreatorTokenStore) -> None:
        self._client = client
        self._tokens = tokens

    def check(self, session: Session, video: Video) -> set[Modality]:
        token = self._tokens.access_token_for_channel(video.channel_id)
        if token is None:
            return set()
        tracks = self._client.list_tracks(video.id, token)
        return {Modality.TRANSCRIPT} if tracks else set()

    def acquire(self, session: Session, video: Video) -> int:
        token = self._tokens.access_token_for_channel(video.channel_id)
        if token is None:
            return 0
        tracks = self._client.list_tracks(video.id, token)
        if not tracks:
            return 0

        # Prefer creator-uploaded (standard) tracks in the video's language over ASR tracks.
        def rank(t: dict[str, object]) -> tuple[int, int]:
            sn = t.get("snippet", {})
            assert isinstance(sn, dict)
            return (
                0 if sn.get("trackKind") == "standard" else 1,
                0 if sn.get("language") == video.language else 1,
            )

        track = sorted(tracks, key=rank)[0]
        snippet = track.get("snippet", {})
        raw = self._client.download(str(track["id"]), token)
        transcript = parse_timed_text(raw, language=snippet.get("language"))
        _, created = add_transcript_source(
            session,
            video_id=video.id,
            transcript=transcript,
            source_type=SourceType.CREATOR_AUTHORIZED_CAPTIONS,
            authorization_basis=(
                f"Channel owner OAuth grant (youtube.force-ssl) for channel {video.channel_id}; "
                f"caption track {track['id']} ({snippet.get('trackKind', 'unknown')})"
            ),
            license="creator-authorized",
        )
        return int(created)


def build_providers(extra: list[ContentProvider] | None = None) -> list[ContentProvider]:
    providers: list[ContentProvider] = [StoredContentProvider()]
    providers.extend(extra or [])
    return providers
