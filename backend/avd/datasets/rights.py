"""Use-time rights review for licensed dataset sources.

A licence file in a repository does not by itself establish that it covers prose transcripts, or
every use we make of them. Each source therefore carries a *review*: which uses rest on an explicit
statement by the publisher, and which are still pending confirmation. Pending uses are enforced
(fail closed) wherever they would happen:

* ``external_annotation`` — showing the text to human raters outside the project;
* ``public_quotes`` — returning verbatim excerpts through the public API / extension.

Internal analysis (storing, evaluating, deriving metrics) is allowed only when the review says so.

A pending use becomes confirmed only through a committed, reviewable record in
``benchmarks/datasets/rights_confirmations.json`` (who confirmed, what scope, evidence reference,
date). This is not legal advice; it makes the assumptions explicit and auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from avd.config import REPO_ROOT

CONFIRMATIONS_FILE = REPO_ROOT / "benchmarks" / "datasets" / "rights_confirmations.json"

USES = ("internal_analysis", "external_annotation", "public_quotes", "redistribution")


@dataclass(frozen=True)
class RightsReview:
    source: str
    status: str  # explicit_content_licence | licence_scope_unconfirmed
    confirmed_by_source: tuple[str, ...]  # uses resting on the publisher's own statements
    pending: tuple[str, ...]
    facts: tuple[str, ...]  # verified against the upstream repository at the pinned commit
    assumptions: tuple[str, ...]  # unresolved; not established by the source documentation
    remediation: str


REVIEWS: dict[str, RightsReview] = {
    "changelog": RightsReview(
        source="changelog",
        status="explicit_content_licence",
        confirmed_by_source=("internal_analysis", "external_annotation", "public_quotes", "redistribution"),
        pending=(),
        facts=(
            "Repository README section 'License & Copyright' states: Creative Commons Attribution Share "
            "Alike 4.0, for the transcripts repository.",
            "The same section states the bigtent/ folder is copyright Grafana Labs; bigtent/ is excluded.",
            "LICENSE file at the pinned commit is the CC BY-SA 4.0 legal code.",
        ),
        assumptions=(
            "Changelog Media holds (or was granted) the rights to license guests' spoken contributions.",
            "Rubric evaluations quoting short attributed excerpts are analysis, not adaptations; if they "
            "were adaptations, ShareAlike would apply to them when published.",
        ),
        remediation="None required for current uses; honour takedown requests via `avd dataset revoke`.",
    ),
    "talkpython": RightsReview(
        source="talkpython",
        status="licence_scope_unconfirmed",
        confirmed_by_source=("internal_analysis",),
        pending=("external_annotation", "public_quotes", "redistribution"),
        facts=(
            "Repository root LICENSE at the pinned commit is the GNU GPL v3 (a software licence).",
            "README: 'This repository holds the transcripts for the podcast Talk Python To Me ... Feel "
            "free to use the files to help you explore the conversations!'",
            "Neither file states that the GPL is intended to cover the transcript text.",
        ),
        assumptions=(
            "GPL-3.0 applies to the transcript text (plausible as the repository licence, but not stated).",
            "The README invitation covers automated analysis for research (relied on for internal analysis only).",
            "The publisher holds rights to guests' spoken contributions.",
        ),
        remediation=(
            "Ask the publisher (Michael Kennedy, Talk Python) to confirm in writing that the transcripts "
            "may be shown to external raters and quoted with attribution, then record it in "
            "benchmarks/datasets/rights_confirmations.json. Until then these items are excluded from "
            "annotation and their quotes are withheld from public responses."
        ),
    ),
}


@lru_cache(maxsize=1)
def _confirmations() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not CONFIRMATIONS_FILE.exists():
        return out
    data: list[dict[str, Any]] = json.loads(CONFIRMATIONS_FILE.read_text()).get("confirmations", [])
    for rec in data:
        # A confirmation must say who, what, and on what evidence; incomplete records are ignored.
        if all(rec.get(k) for k in ("source", "uses", "confirmed_by", "evidence", "date")):
            out.setdefault(rec["source"], set()).update(u for u in rec["uses"] if u in USES)
    return out


def pending_uses(source_key: str | None) -> set[str]:
    """Uses still pending for a dataset source; unknown sources have nothing pending here (their
    rights come from their own authorization basis, e.g. a user attestation)."""
    review = REVIEWS.get(source_key or "")
    if review is None:
        return set()
    return set(review.pending) - _confirmations().get(review.source, set())


def source_key_for(provenance_url: str | None) -> str | None:
    if not provenance_url:
        return None
    from avd.datasets.sources import SOURCES

    for key, src in SOURCES.items():
        if provenance_url.startswith(src.repo_url):
            return key
    return None


def allowed(provenance_url: str | None, use: str) -> bool:
    return use not in pending_uses(source_key_for(provenance_url))


def report() -> dict[str, Any]:
    return {
        key: {
            "status": r.status,
            "confirmed_by_source": list(r.confirmed_by_source),
            "confirmed_by_record": sorted(_confirmations().get(key, set())),
            "pending": sorted(pending_uses(key)),
            "facts": list(r.facts),
            "assumptions": list(r.assumptions),
            "remediation": r.remediation,
        }
        for key, r in REVIEWS.items()
    }


WITHHELD = "[quote withheld: publisher has not yet confirmed public quoting]"


def video_pending_uses(session: Any, video_id: str) -> set[str]:
    """Pending uses across the video's active content sources (any pending source blocks the use)."""
    from sqlalchemy import select

    from avd.db.models import ContentSource

    urls = session.scalars(
        select(ContentSource.provenance_url).where(
            ContentSource.video_id == video_id, ContentSource.status == "active"
        )
    ).all()
    out: set[str] = set()
    for url in urls:
        out |= pending_uses(source_key_for(url))
    return out


def withhold_quotes(payload: dict[str, Any]) -> dict[str, Any]:
    """Blank verbatim excerpts in a serialised evaluation; positions and verification status stay."""
    for d in payload.get("dimensions", []):
        for e in d.get("evidence", []):
            if e.get("quote"):
                e["quote"] = WITHHELD
    payload["quotes_withheld"] = True
    return payload
