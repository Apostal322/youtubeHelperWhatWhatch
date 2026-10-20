"""The usability pilot: a labelled excerpt that tests the packet, and can never become a rating.

The "answers" are test inputs exercising the workflow, not ratings of any item.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.annotation import offline, usability
from avd.annotation import service as svc
from avd.db.models import Annotation, AnnotationPacket
from tests.conftest import DB
from tests.test_annotation import DS, seed

CONSENT = "Test consent text for the usability pilot."


@pytest.fixture
def sess(db: DB) -> Iterator[Session]:
    with db.sf() as s:
        seed(s)
        yield s


def _issue(s: Session, words: int = 300) -> dict[str, Any]:
    return usability.issue_usability_packet(s, DS, "pilot-1", consent_text=CONSENT, max_words=words)[1]


def _return(data: dict[str, Any], scores: list[dict[str, Any]], consent: str = CONSENT) -> dict[str, Any]:
    item = data["items"][0]
    return {
        "format": offline.RETURN_FORMAT,
        "format_version": "1.0",
        "scope": data["scope"],
        "packet_id": data["packet_id"],
        "secret": "",
        "rater": "pilot-1",
        "consent": {"agreed": True, "text_sha256": offline.consent_digest(consent)},
        "results": [
            {
                "annotation_id": item["annotation_id"],
                "scores": scores,
                "notes": "The 'ambiguous' status was unclear.",
                "client_duration_seconds": 960,
                "flags": ["ambiguous_rubric"],
            }
        ],
    }


def test_the_pilot_is_a_labelled_excerpt_and_stores_nothing(sess: Session) -> None:
    before = sess.scalar(select(func.count()).select_from(Annotation))
    data = _issue(sess)
    item = data["items"][0]
    assert data["scope"] == item["scope"] == usability.SCOPE and item["is_calibration"]
    assert 300 <= item["excerpt_words"] < item["episode_words"] == 485  # whole segments, cut early
    assert item["relevance_queries"] == []  # relevance needs the whole episode
    assert sess.scalar(select(func.count()).select_from(Annotation)) == before
    assert sess.scalar(select(func.count()).select_from(AnnotationPacket)) == 0
    html = usability.render_usability_packet(data)
    assert "default-src 'none'" in html and "never used as a rating of the episode" in html


def test_a_draft_consent_text_cannot_be_used(sess: Session) -> None:
    with pytest.raises(svc.AnnotationError, match="draft"):
        usability.issue_usability_packet(sess, DS, "p", consent_text=offline.CONSENT_DRAFT_MARKER + "\ntext")


def test_a_pilot_return_is_refused_as_a_rating_and_read_as_usability(sess: Session) -> None:
    data = _issue(sess)
    seg = data["items"][0]["segments"][2]
    quote = " ".join(seg["text"].split()[:4])
    scores = [
        {
            "dimension": "depth",
            "status": "scored",
            "score": 3,
            "evidence": [{"segment_index": 2, "quote": quote}],
        },
        # a quote from outside the excerpt does not count as found
        {
            "dimension": "clarity",
            "status": "scored",
            "score": 4,
            "evidence": [{"segment_index": 999, "quote": "x"}],
        },
        {"dimension": "structure", "status": "ambiguous", "note": "Unclear what counts as structure."},
        {"dimension": "pacing", "status": ""},
    ]
    ret = _return(data, scores)
    with pytest.raises(svc.AnnotationError, match="never a rating"):
        offline.import_return(sess, ret)
    report = usability.usability_report(
        sess, json.loads(json.dumps(ret)), consent_text=CONSENT, max_words=300
    )
    assert report["is_rating"] is False and "Nothing was stored" in report["notice"]
    it = report["items"][0]
    assert it["minutes_reported"] == 16.0 and it["dimensions_answered"] == 3
    assert it["citations"] == {"made": 2, "found_in_excerpt": 1, "not_found": ["clarity[999]"]}
    assert it["dimension_notes"] == {"structure": "Unclear what counts as structure."}
    assert sess.scalar(select(func.count()).select_from(Annotation)) == 0


def test_the_report_needs_agreement_to_the_same_consent_text(sess: Session) -> None:
    ret = _return(_issue(sess), [], consent="a different text")
    with pytest.raises(svc.AnnotationError, match="consent"):
        usability.usability_report(sess, ret, consent_text=CONSENT)


def test_full_packet_returns_are_not_usability_returns(sess: Session) -> None:
    with pytest.raises(svc.AnnotationError, match="not a usability-pilot return"):
        usability.usability_report(
            sess, {"format": offline.RETURN_FORMAT, "packet_id": "abc", "results": []}, consent_text=CONSENT
        )
    assert re.fullmatch(r"usability-[0-9a-f]{16}", _issue(sess)["packet_id"])
