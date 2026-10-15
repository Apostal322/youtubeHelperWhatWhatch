"""Prompt-injection defences for untrusted video text (transcripts, titles, descriptions).

Layers:

1. **Framing** — untrusted text is rendered with ``<``/``>`` replaced by look-alike guillemets, so it
   can never open or close a delimiter tag, and delimiters carry a nonce derived from the content
   hash (not predictable from inside the content).
2. **Detection** — instruction-like content (instruction overrides, role spoofing, score
   requests, JSON/tag spoofing) is flagged per segment. A flagged evaluation is marked
   ``needs_review``; flagged segments are not accepted as evidence (an instruction to the grader
   is not evidence of quality). No numeric penalty is applied — there is no empirical basis for one.
3. **Grounding** — independent of detection, every model citation must verify against the
   transcript (``validation.py``), so obeying an injection while citing it gets invalidated.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|earlier|system|the)\b"
            r".{0,30}\b(instructions?|prompts?|rules?|rubric|guidelines)\b",
            re.I | re.S,
        ),
    ),
    (
        "instruction_override",
        re.compile(
            r"\b(ignora|olvida)\b.{0,30}\binstrucciones\b|\b(ignorier\w*|vergiss)\b.{0,30}\banweisungen\b",
            re.I | re.S,
        ),
    ),
    ("role_spoofing", re.compile(r"(^|[\s\"'(\[])(system|assistant|developer)\s*:", re.I)),
    (
        "role_spoofing",
        re.compile(
            r"\b(you are|you're|act as|pretend to be)\b.{0,25}\b(ai|assistant|evaluator|grader|language model|llm)\b",
            re.I,
        ),
    ),
    ("prompt_reference", re.compile(r"\b(system|developer)\s+(prompt|message|instructions?)\b", re.I)),
    (
        "score_manipulation",
        re.compile(
            r"\b(give|rate|score|grade|assign|award)\b.{0,40}\b(video|content|channel|it|this|me|us)\b"
            r".{0,40}\b(5|five|10|ten|100|perfect|maximum|max|highest|top|full marks)\b",
            re.I | re.S,
        ),
    ),
    (
        "tag_spoofing",
        re.compile(
            r"</?\s*(system|assistant|user|transcript[\w-]*|instructions?|rubric|metadata)\b[^>]{0,40}>", re.I
        ),
    ),
    ("json_spoofing", re.compile(r"[\"'](judgments|assessable|score|confidence)[\"']\s*:", re.I)),
]


@dataclass(frozen=True)
class InjectionFinding:
    location: str  # "segment" | "title" | "description" | "tags"
    segment_index: int | None
    kind: str
    excerpt: str


def scan_text(text: str, location: str, segment_index: int | None = None) -> list[InjectionFinding]:
    out = []
    for kind, pat in _PATTERNS:
        m = pat.search(text)
        if m:
            start = max(0, m.start() - 20)
            out.append(InjectionFinding(location, segment_index, kind, text[start : m.end() + 20][:160]))
    return out


def scan(
    segments: list[str], title: str = "", description: str = "", tags: tuple[str, ...] = ()
) -> list[InjectionFinding]:
    findings: list[InjectionFinding] = []
    for i, seg in enumerate(segments):
        findings.extend(scan_text(seg, "segment", i))
    findings.extend(scan_text(title, "title"))
    findings.extend(scan_text(description, "description"))
    findings.extend(scan_text(" ".join(tags), "tags"))
    return findings


def neutralize(text: str) -> str:
    """Make untrusted text unable to form delimiter tags. Look-alike characters keep it readable;
    evidence verification normalises punctuation, so quotes still verify."""
    return text.replace("<", "‹").replace(">", "›")


def nonce_for(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:12]
