"""Authorized media handling: probing, audio extraction and frame sampling with ffmpeg.

Only media the system is permitted to process is handled here: files submitted with an
attestation, creator-authorized media, or test fixtures. Nothing downloads from YouTube.

Safety:
* ffmpeg/ffprobe run with ``-protocol_whitelist file``, so a crafted playlist or container cannot
  make them fetch URLs. They also run without stdin, with a timeout, and with bounded output.
* Inputs must be regular files inside the media directory.
* Only a fixed number of downscaled grayscale frames is decoded, so memory stays bounded.
* Every ffmpeg/ffprobe process runs under OS resource limits (``prlimit``): address space, CPU
  seconds, size of any file it writes, and open files. Media longer than ``MAX_MEDIA_SECONDS`` is
  refused, and audio extraction is capped at that length.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
PRLIMIT = shutil.which("prlimit")
MAX_MEDIA_SECONDS = 6 * 3600
LIMIT_ADDRESS_SPACE = 2 * 1024**3  # bytes
LIMIT_FILE_SIZE = 1024**3  # 16 kHz mono PCM for 6 h is ~0.7 GB
LIMIT_OPEN_FILES = 64
_COMMON = ["-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file"]
SAFE = ["-nostdin", *_COMMON]  # ffmpeg
PROBE_SAFE = _COMMON  # ffprobe has no -nostdin; stdin is closed below instead
FRAME_WIDTH = 320


class MediaError(ValueError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float
    has_audio: bool
    has_video: bool
    width: int | None
    height: int | None
    fps: float | None
    format_name: str


def available() -> bool:
    return FFMPEG is not None and FFPROBE is not None


def limited(cmd: list[str], cpu_seconds: float) -> list[str]:
    """Wrap a command in OS resource limits. Without ``prlimit`` the timeout still applies, and
    ``limits_enforced()`` reports the gap (visible in /v1/status)."""
    if PRLIMIT is None:
        return cmd
    return [
        PRLIMIT,
        f"--as={LIMIT_ADDRESS_SPACE}",
        f"--cpu={max(1, int(cpu_seconds))}",
        f"--fsize={LIMIT_FILE_SIZE}",
        f"--nofile={LIMIT_OPEN_FILES}",
        "--",
        *cmd,
    ]


def limits_enforced() -> bool:
    return PRLIMIT is not None


def _check(path: Path, root: Path | None) -> Path:
    p = path.resolve()
    if not p.is_file():
        raise MediaError("media file not found")
    if root is not None and root.resolve() not in p.parents:
        raise MediaError("media must live inside the media directory")
    return p


def probe(path: Path, root: Path | None = None, timeout: float = 30.0) -> MediaInfo:
    if FFPROBE is None:
        raise MediaError("ffprobe is not installed")
    p = _check(path, root)
    proc = subprocess.run(
        limited(
            [FFPROBE, *PROBE_SAFE, "-print_format", "json", "-show_format", "-show_streams", str(p)], timeout
        ),
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError("not a readable audio/video file")
    data = json.loads(proc.stdout or b"{}")
    streams = data.get("streams") or []
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None and audio is None:
        raise MediaError("no audio or video stream")
    if float((data.get("format") or {}).get("duration") or 0.0) > MAX_MEDIA_SECONDS:
        raise MediaError(f"media longer than {MAX_MEDIA_SECONDS // 3600} h is not processed")
    fps = None
    if video is not None and video.get("avg_frame_rate", "0/0") not in ("0/0", ""):
        num, _, den = video["avg_frame_rate"].partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else None
    return MediaInfo(
        duration_seconds=float((data.get("format") or {}).get("duration") or 0.0),
        has_audio=audio is not None,
        has_video=video is not None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        fps=round(fps, 3) if fps else None,
        format_name=str((data.get("format") or {}).get("format_name", "")),
    )


def extract_audio_wav(path: Path, out: Path, root: Path | None = None, timeout: float = 600.0) -> Path:
    """Decode the audio track to 16 kHz mono 16-bit PCM WAV (the audio evaluator's input)."""
    if FFMPEG is None:
        raise MediaError("ffmpeg is not installed")
    p = _check(path, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        limited(
            [
                FFMPEG,
                *SAFE,
                "-y",
                "-i",
                str(p),
                "-t",
                str(MAX_MEDIA_SECONDS),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            timeout,
        ),
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0 or not out.exists():
        raise MediaError("audio extraction failed")
    return out


def sample_frames(
    path: Path, info: MediaInfo, n: int = 24, root: Path | None = None, timeout: float = 600.0
) -> list[np.ndarray]:
    """``n`` evenly spaced grayscale frames, downscaled to ``FRAME_WIDTH`` wide (float32, 0-255)."""
    if FFMPEG is None:
        raise MediaError("ffmpeg is not installed")
    if not info.has_video or not info.width or not info.height:
        return []
    p = _check(path, root)
    w = min(FRAME_WIDTH, info.width)
    h = max(2, round(info.height * w / info.width / 2) * 2)
    frames: list[np.ndarray] = []
    duration = max(info.duration_seconds, 0.1)
    for i in range(n):
        t = duration * (i + 0.5) / n
        proc = subprocess.run(
            limited(
                [
                    FFMPEG,
                    *SAFE,
                    "-ss",
                    f"{t:.3f}",
                    "-i",
                    str(p),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale={w}:{h}",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    "pipe:1",
                ],
                timeout / n,
            ),
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout / n,
            check=False,
        )
        if proc.returncode == 0 and len(proc.stdout) == w * h:
            frames.append(np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w).astype(np.float32))
    return frames
