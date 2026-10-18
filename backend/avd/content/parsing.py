"""Parsers for caption formats (WebVTT, SRT) into ``Transcript``."""

from __future__ import annotations

import re

from avd.content.models import Transcript, TranscriptSegment

_TS = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)
_TAG = re.compile(r"<[^>]+>")


def _secs(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_timed_text(raw: str, language: str | None = None) -> Transcript:
    """Parse WebVTT or SRT. Raises ``ValueError`` if no cues are found."""
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\r?\n\s*\r?\n", raw.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            m = _TS.search(line)
            if not m:
                continue
            g = m.groups()
            text = " ".join(_TAG.sub("", t).strip() for t in lines[i + 1 :]).strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        start=_secs(g[0], g[1], g[2], g[3]),
                        end=_secs(g[4], g[5], g[6], g[7]),
                        text=text,
                    )
                )
            break
    if not segments:
        raise ValueError("no caption cues found (expected WebVTT or SRT)")
    # Auto-generated captions often repeat rolling text; drop exact consecutive duplicates.
    deduped = [segments[0]]
    for seg in segments[1:]:
        if seg.text != deduped[-1].text:
            deduped.append(seg)
    return Transcript(language=language, segments=deduped)


def transcript_from_plain_text(
    text: str, language: str | None = None, words_per_second: float = 2.5
) -> Transcript:
    """Split untimed text into sentence-ish segments.

    Timestamps are *estimated* from an assumed speaking rate; evidence derived from such a
    transcript is labelled accordingly by the caller (``timing_estimated``).
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        raise ValueError("empty transcript text")
    segments, t, buf = [], 0.0, []
    for s in sentences:
        buf.append(s)
        if sum(len(b.split()) for b in buf) >= 20:
            words = sum(len(b.split()) for b in buf)
            segments.append(TranscriptSegment(start=t, end=t + words / words_per_second, text=" ".join(buf)))
            t += words / words_per_second
            buf = []
    if buf:
        words = sum(len(b.split()) for b in buf)
        segments.append(TranscriptSegment(start=t, end=t + words / words_per_second, text=" ".join(buf)))
    return Transcript(language=language, segments=segments)
