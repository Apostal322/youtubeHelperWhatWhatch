"""Fetch licensed source files at a pinned commit, with a local content-addressed cache.

Only the publisher's own repository is contacted (``raw.githubusercontent.com/<repo>/<commit>``),
so the bytes are exactly those at the pinned commit. Checksums recorded in a manifest are verified
on every read; a mismatch fails closed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import httpx

from avd.datasets.sources import LicensedSource

USER_AGENT = "ai-video-discovery-research/0.2 (+dataset builder; pinned-commit fetch)"


class ChecksumMismatch(RuntimeError):
    pass


_SAFE_PATH = re.compile(r"^[A-Za-z0-9_.\- ]+(/[A-Za-z0-9_.\- ]+)*$")


def check_relative_path(path: str) -> None:
    """Manifest paths are untrusted input: plain relative repository paths only (no absolute
    paths, no '..' segments, no backslashes or control characters)."""
    if not _SAFE_PATH.match(path) or any(part in ("", ".", "..") for part in path.split("/")):
        raise PermissionError(f"unsafe source path: {path!r}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SourceFetcher:
    def __init__(self, cache_dir: Path, client: httpx.Client | None = None, offline: bool = False) -> None:
        self.cache_dir = cache_dir
        self.offline = offline
        self._client = client

    def _cache_path(self, source: LicensedSource, path: str) -> Path:
        check_relative_path(path)
        base = (self.cache_dir / source.key / source.commit).resolve()
        target = (base / path).resolve()
        if base not in target.parents:
            raise PermissionError(f"path escapes the cache directory: {path!r}")
        return target

    def get(self, source: LicensedSource, path: str, expected_sha256: str | None = None) -> bytes:
        check_relative_path(path)
        if any(path.startswith(p) for p in source.excluded_prefixes):
            raise PermissionError(f"{source.key}:{path} is excluded from the licensed scope")
        cached = self._cache_path(source, path)
        if cached.exists():
            data = cached.read_bytes()
        else:
            if self.offline:
                raise FileNotFoundError(f"{source.key}:{path} not in cache and offline mode is on")
            if self._client is None:
                self._client = httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT})
            resp = self._client.get(source.raw_url(path))
            resp.raise_for_status()
            data = resp.content
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
        if expected_sha256 is not None and sha256_bytes(data) != expected_sha256:
            raise ChecksumMismatch(
                f"{source.key}:{path} sha256 {sha256_bytes(data)} != manifest {expected_sha256}"
            )
        return data
