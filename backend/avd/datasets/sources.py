"""Licensed real-world transcript sources and their parsers.

Each source is a publisher's own distribution channel with an explicit licence, fetched at a
pinned commit so the dataset is reproducible and the provenance is exact. Nothing here downloads
media, scrapes YouTube or uses third-party mirrors.

| Source | Licence | What it is |
|---|---|---|
| ``changelog`` | CC BY-SA 4.0 (README "License & Copyright"; ``bigtent/`` excluded: Grafana Labs copyright) | Human-edited transcripts of Changelog Media podcasts (interviews/discussions about software, AI, psychology, business) |
| ``talkpython`` | GPL-3.0 (LICENSE) + README: "Feel free to use the files to help you explore the conversations!" | Creator-published WebVTT transcripts of Talk Python To Me episodes, incl. the YouTube editions (``youtube_transcripts/``) |

Rights recorded per item: owner, attribution, licence, provenance URL at the pinned commit,
retrieval time, raw and normalised checksums, permitted processing/storage, restrictions and the
revocation policy (``docs/permissions.md``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from avd.content.models import Transcript, TranscriptSegment
from avd.content.parsing import parse_timed_text
from avd.content.processing import consolidate_segments


@dataclass(frozen=True)
class SourceRights:
    license: str
    license_url: str
    owner: str
    attribution: str
    authorization_basis: str
    permitted_processing: tuple[str, ...]
    permitted_storage: str
    restrictions: tuple[str, ...]
    revocation_policy: str
    legal_review_notes: str = ""


@dataclass(frozen=True)
class LicensedSource:
    key: str
    repo_url: str
    commit: str
    rights: SourceRights
    excluded_prefixes: tuple[str, ...] = ()
    group_labels: dict[str, str] = field(default_factory=dict)  # group_key -> human label

    def raw_url(self, path: str) -> str:
        owner_repo = self.repo_url.removeprefix("https://github.com/")
        return f"https://raw.githubusercontent.com/{owner_repo}/{self.commit}/{path}"

    def blob_url(self, path: str) -> str:
        return f"{self.repo_url}/blob/{self.commit}/{path}"


CHANGELOG = LicensedSource(
    key="changelog",
    repo_url="https://github.com/thechangelog/transcripts",
    commit="44a0e16c5919d332e05de481919fbfca90a2c90b",
    rights=SourceRights(
        license="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        owner="Changelog Media LLC",
        attribution="Transcript © Changelog Media LLC, CC BY-SA 4.0, via github.com/thechangelog/transcripts",
        authorization_basis=(
            "Publisher-distributed transcript under CC BY-SA 4.0 (repository README 'License & Copyright'); "
            "obtained from the publisher's own repository at a pinned commit"
        ),
        permitted_processing=("store", "analyze", "quote_excerpts_with_attribution", "derive_metrics"),
        permitted_storage="indefinite, with attribution; adaptations of the text must be CC BY-SA",
        restrictions=("attribution_required", "share_alike_for_adaptations"),
        revocation_policy=(
            "CC licences are irrevocable for obtained copies; honour publisher takedown requests by "
            "revoking the content source and withdrawing derived evaluations"
        ),
        legal_review_notes="Evaluations are treated as analysis, not adaptations of the text; confirm with counsel.",
    ),
    excluded_prefixes=(
        "bigtent/",
        "news/",
        "afk/",
        "scripts/",
    ),  # bigtent: Grafana copyright; news/afk: no timestamps
    group_labels={
        "podcast": "The Changelog",
        "jsparty": "JS Party",
        "gotime": "Go Time",
        "practicalai": "Practical AI",
        "shipit": "Ship It!",
        "friends": "Changelog & Friends",
        "founderstalk": "Founders Talk",
        "brainscience": "Brain Science",
        "backstage": "Backstage",
        "rfc": "Request For Commits",
        "spotlight": "Spotlight",
    },
)

TALKPYTHON = LicensedSource(
    key="talkpython",
    repo_url="https://github.com/mikeckennedy/talk-python-transcripts",
    commit="accd851f0d3353bf2d3792969d7f334a4c4768b3",
    rights=SourceRights(
        license="GPL-3.0",
        license_url="https://www.gnu.org/licenses/gpl-3.0.html",
        owner="Michael Kennedy / Talk Python",
        attribution="Transcript © Talk Python To Me (Michael Kennedy), GPL-3.0, via github.com/mikeckennedy/talk-python-transcripts",
        authorization_basis=(
            "Creator-published transcripts (repository LICENSE GPL-3.0; README: 'Feel free to use the files "
            "to help you explore the conversations!'); obtained from the creator's repository at a pinned commit"
        ),
        permitted_processing=("store", "analyze", "quote_excerpts_with_attribution", "derive_metrics"),
        permitted_storage="indefinite; redistribution of the text itself must remain GPL-3.0",
        restrictions=("attribution_required", "copyleft_for_redistributed_text"),
        revocation_policy="Honour creator takedown requests by revoking the content source and withdrawing evaluations",
        legal_review_notes="GPL on prose transcripts is unusual; confirm the analysis use with counsel.",
    ),
    group_labels={"talkpython": "Talk Python To Me"},
)

SOURCES = {s.key: s for s in (CHANGELOG, TALKPYTHON)}

# --------------------------------------------------------------------------------------- parsing

_TS = re.compile(r"\\\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d+))?\\\]")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPH = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_SPEAKER_ONLY = re.compile(r"^[^.?!]{1,60}:$")
# Ad-break markers ("**Break:** \[20:37\]", "**Break**: \[27:13\]"): the label is dropped, the
# timestamp after it is kept as an anchor.
_BREAK = re.compile(r"^\*\*(Break|Sponsor|Ad)\b[^*]{0,20}\*\*\s*:?\s*", re.I)


def _clean_markdown(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    for _ in range(2):
        text = _EMPH.sub(r"\2", text)
    text = text.replace("\\[", "[").replace("\\]", "]").replace("\\", "")
    return " ".join(text.split())


def _ts_seconds(m: re.Match[str]) -> float:
    h, mi, sec, frac = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(sec) + (float(f"0.{frac}") if frac else 0.0)


def parse_changelog_markdown(raw: str) -> tuple[Transcript, dict[str, object]]:
    """Parse a Changelog transcript into speaker-paragraph segments.

    The source carries inline timestamps roughly every 40-90 seconds (about 1.3-1.6 per 1,000
    words), not per paragraph. Each paragraph therefore becomes its own segment and carries the
    **source span** it falls in: from the last timestamp at or before it to the next timestamp
    after it. Times are never interpolated, so several consecutive segments can share one span;
    citations stay exact at span granularity. Text before the first timestamp starts at 0:00 (the
    episode start). The final span's end is unknown and is set equal to its start.
    """
    paragraphs = [_BREAK.sub("", p.strip()) for p in raw.split("\n")]
    paragraphs = [p for p in paragraphs if p]
    # Each item: (anchor time this piece starts at, or None, text)
    pieces: list[tuple[float | None, str]] = []
    for para in paragraphs:
        pos = 0
        for m in _TS.finditer(para):
            if para[pos : m.start()].strip():
                pieces.append((None, para[pos : m.start()]))
            pieces.append((_ts_seconds(m), ""))
            pos = m.end()
        pieces.append((None, para[pos:] + "\n"))
    # Group text into paragraphs, remembering which span (anchor) each paragraph starts in.
    anchors: list[float] = [0.0]
    paras: list[tuple[int, list[str]]] = []  # (span index, text parts)
    open_para = False
    for ts, text in pieces:
        if ts is not None:
            if ts > anchors[-1]:
                anchors.append(ts)
                if open_para and _SPEAKER_ONLY.match(" ".join(paras[-1][1])):
                    # "**Speaker:** \[ts\] text": the paragraph begins at the new anchor.
                    paras[-1] = (len(anchors) - 1, paras[-1][1])
                elif open_para:
                    open_para = False  # a mid-paragraph anchor splits the paragraph at the anchor
            continue
        ends_para = text.endswith("\n")
        cleaned = _clean_markdown(text)
        if cleaned:
            if not open_para:
                paras.append((len(anchors) - 1, []))
                open_para = True
            paras[-1][1].append(cleaned)
        if ends_para:
            open_para = False
    out: list[TranscriptSegment] = []
    for span, parts in paras:
        start = anchors[span]
        end = anchors[span + 1] if span + 1 < len(anchors) else start
        out.append(TranscriptSegment(start=start, end=end, text=" ".join(parts)))
    spans_used = len({(s.start, s.end) for s in out})
    report: dict[str, object] = {
        "timing": "source_anchor_spans",
        "anchors": len(anchors) - 1,
        "spans": spans_used,
        "segments": len(out),
        "first_segment_starts_at_episode_start": True,
        "last_span_end_unknown": True,
    }
    return Transcript(language="en", segments=out), report


def parse_vtt_consolidated(raw: str) -> tuple[Transcript, dict[str, object]]:
    cues = parse_timed_text(raw, "en")
    t = consolidate_segments(cues)
    return t, {"timing": "source_cues_consolidated", "cues": len(cues.segments), "segments": len(t.segments)}


def item_metadata(source: LicensedSource, path: str) -> dict[str, str]:
    """Publisher-derived metadata from the file path. Episode titles are not available offline for
    every source; when absent the title is the show + episode number, never invented. External
    URLs follow the publisher's URL scheme but were not fetched (``external_url_verified``)."""
    name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if source.key == "changelog":
        group = path.split("/", 1)[0]
        show = source.group_labels.get(group, group)
        ep = name.rsplit("-", 1)[-1]
        return {
            "group_key": f"changelog:{group}",
            "show": show,
            "episode": ep,
            "title": f"{show} #{ep}",
            "title_source": "derived_from_path",
            "platform": "changelog.com",
            "external_url": f"https://changelog.com/{group}/{ep}",
            "external_url_verified": "false",
        }
    num, _, slug = name.partition("-")
    slug = slug.removesuffix("-transcript-final").removesuffix("-transcript").replace("-", " ").strip()
    return {
        "group_key": "talkpython",
        "show": "Talk Python To Me",
        "episode": num.lstrip("0") or "0",
        "title": slug[:1].upper() + slug[1:],
        "title_source": "creator_file_slug",
        "platform": "talkpython.fm",
        "external_url": f"https://talkpython.fm/episodes/show/{num.lstrip('0')}",
        "external_url_verified": "false",
        "edition": "youtube" if path.startswith("youtube_transcripts/") else "audio",
    }


def parse(source: LicensedSource, path: str, raw: str) -> tuple[Transcript, dict[str, object]]:
    if source.key == "changelog":
        return parse_changelog_markdown(raw)
    return parse_vtt_consolidated(raw)
