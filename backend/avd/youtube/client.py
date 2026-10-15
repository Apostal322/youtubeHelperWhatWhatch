"""Minimal, quota-aware client for the official YouTube Data API v3.

Only documented endpoints are used: ``search.list`` (discovery), ``videos.list`` (metadata),
``channels.list`` (channel metadata), and — with a channel owner's OAuth token —
``captions.list`` / ``captions.download``. No scraping, no media downloading.

Quota costs follow the published quota table (search.list = 100 units, videos.list = 1,
channels.list = 1, captions.list = 50, captions.download = 200). Every call is recorded in the
cost ledger and checked against the daily allowance *before* it is made.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from avd.logging import get_logger

log = get_logger(__name__)

API_BASE = "https://www.googleapis.com/youtube/v3"

QUOTA_COST = {
    "search.list": 100,
    "videos.list": 1,
    "channels.list": 1,
    "captions.list": 50,
    "captions.download": 200,
}


class QuotaExceeded(RuntimeError):
    pass


class YouTubeAPIError(RuntimeError):
    def __init__(self, status: int, message: str, retryable: bool) -> None:
        super().__init__(f"YouTube API {status}: {message}")
        self.status = status
        self.retryable = retryable


# (operation, units) -> None; raises QuotaExceeded when the budget would be exceeded.
QuotaGuard = Callable[[str, int], None]


_DURATION = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def parse_iso_duration(value: str | None) -> int | None:
    if not value:
        return None
    m = _DURATION.fullmatch(value)
    if not m:
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


@dataclass
class VideoMetadata:
    video_id: str
    channel_id: str
    channel_title: str
    title: str
    description: str
    published_at: datetime | None
    duration_seconds: int | None
    language: str | None
    category_id: str | None
    tags: list[str]
    view_count: int | None
    like_count: int | None
    comment_count: int | None
    caption_flag: bool
    licensed_content: bool | None
    embeddable: bool | None


@dataclass
class ChannelMetadata:
    channel_id: str
    title: str
    subscriber_count: int | None
    country: str | None


class YouTubeDataClient:
    def __init__(
        self,
        api_key: str,
        quota_guard: QuotaGuard,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._key = api_key
        self._guard = quota_guard
        self._http = httpx.Client(base_url=API_BASE, timeout=timeout, transport=transport)

    def _get(self, endpoint: str, op: str, params: dict[str, Any]) -> dict[str, Any]:
        self._guard(op, QUOTA_COST[op])
        resp = self._http.get(endpoint, params={**params, "key": self._key})
        if resp.status_code != 200:
            reason = ""
            try:
                reason = resp.json()["error"]["errors"][0]["reason"]
            except Exception:
                reason = resp.text[:200]
            if resp.status_code == 403 and "quota" in reason.lower():
                raise QuotaExceeded(reason)
            raise YouTubeAPIError(
                resp.status_code, reason, retryable=resp.status_code in (429, 500, 502, 503, 504)
            )
        data: dict[str, Any] = resp.json()
        return data

    def search(
        self,
        query: str,
        *,
        language: str | None = None,
        order: str = "date",
        page_token: str | None = None,
        published_before: datetime | None = None,
        published_after: datetime | None = None,
        max_results: int = 50,
    ) -> tuple[list[str], str | None]:
        """Return (video_ids, next_page_token).

        ``order=viewCount`` is intentionally rejected: discovery must not be popularity-ranked.
        """
        if order not in ("date", "relevance", "rating", "title"):
            raise ValueError(f"order={order!r} is not allowed for popularity-blind discovery")
        params: dict[str, Any] = {
            "part": "id",
            "q": query,
            "type": "video",
            "order": order,
            "maxResults": max_results,
            "safeSearch": "moderate",
        }
        if language:
            params["relevanceLanguage"] = language
        if page_token:
            params["pageToken"] = page_token
        if published_before:
            params["publishedBefore"] = published_before.strftime("%Y-%m-%dT%H:%M:%SZ")
        if published_after:
            params["publishedAfter"] = published_after.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = self._get("/search", "search.list", params)
        ids = [it["id"]["videoId"] for it in data.get("items", []) if it.get("id", {}).get("videoId")]
        return ids, data.get("nextPageToken")

    def videos(self, video_ids: list[str]) -> list[VideoMetadata]:
        """Batch metadata lookup (up to 50 ids per request = 1 quota unit)."""
        out: list[VideoMetadata] = []
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]
            data = self._get(
                "/videos",
                "videos.list",
                {"part": "snippet,contentDetails,statistics,status", "id": ",".join(chunk)},
            )
            for it in data.get("items", []):
                sn, cd = it.get("snippet", {}), it.get("contentDetails", {})
                st, status = it.get("statistics", {}), it.get("status", {})
                published = sn.get("publishedAt")
                out.append(
                    VideoMetadata(
                        video_id=it["id"],
                        channel_id=sn.get("channelId", ""),
                        channel_title=sn.get("channelTitle", ""),
                        title=sn.get("title", ""),
                        description=sn.get("description", ""),
                        published_at=datetime.fromisoformat(published.replace("Z", "+00:00"))
                        if published
                        else None,
                        duration_seconds=parse_iso_duration(cd.get("duration")),
                        language=sn.get("defaultAudioLanguage") or sn.get("defaultLanguage"),
                        category_id=sn.get("categoryId"),
                        tags=sn.get("tags", []) or [],
                        view_count=_int(st.get("viewCount")),
                        like_count=_int(st.get("likeCount")),
                        comment_count=_int(st.get("commentCount")),
                        caption_flag=cd.get("caption") == "true",
                        licensed_content=cd.get("licensedContent"),
                        embeddable=status.get("embeddable"),
                    )
                )
        return out

    def channels(self, channel_ids: list[str]) -> list[ChannelMetadata]:
        out: list[ChannelMetadata] = []
        for i in range(0, len(channel_ids), 50):
            chunk = channel_ids[i : i + 50]
            data = self._get(
                "/channels", "channels.list", {"part": "snippet,statistics", "id": ",".join(chunk)}
            )
            for it in data.get("items", []):
                st = it.get("statistics", {})
                out.append(
                    ChannelMetadata(
                        channel_id=it["id"],
                        title=it.get("snippet", {}).get("title", ""),
                        subscriber_count=None
                        if st.get("hiddenSubscriberCount")
                        else _int(st.get("subscriberCount")),
                        country=it.get("snippet", {}).get("country"),
                    )
                )
        return out


class YouTubeCaptionsClient:
    """Captions access with the *channel owner's* OAuth token (scope youtube.force-ssl).

    The YouTube Data API only allows ``captions.download`` for videos the authenticated user can
    edit, i.e. creators who explicitly connect their channel. This client is never used for
    third-party videos.
    """

    def __init__(self, quota_guard: QuotaGuard, transport: httpx.BaseTransport | None = None) -> None:
        self._guard = quota_guard
        self._http = httpx.Client(base_url=API_BASE, timeout=30.0, transport=transport)

    def list_tracks(self, video_id: str, access_token: str) -> list[dict[str, Any]]:
        self._guard("captions.list", QUOTA_COST["captions.list"])
        resp = self._http.get(
            "/captions",
            params={"part": "snippet", "videoId": video_id},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if resp.status_code != 200:
            raise YouTubeAPIError(resp.status_code, resp.text[:200], resp.status_code >= 500)
        items: list[dict[str, Any]] = resp.json().get("items", [])
        return items

    def download(self, caption_id: str, access_token: str, fmt: str = "vtt") -> str:
        self._guard("captions.download", QUOTA_COST["captions.download"])
        resp = self._http.get(
            f"/captions/{caption_id}",
            params={"tfmt": fmt},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if resp.status_code != 200:
            raise YouTubeAPIError(resp.status_code, resp.text[:200], resp.status_code >= 500)
        return resp.text


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
