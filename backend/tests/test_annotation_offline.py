"""Offline annotation packets: issue, rate without a server, return, import.

The "ratings" are test inputs exercising the workflow; they are never exported as labels for any
real item.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.annotation import offline
from avd.annotation import service as svc
from avd.datasets.importer import revoke_video_sources
from avd.db.models import Annotation, AnnotationPacket, Rater
from tests.conftest import DB, _make_db
from tests.test_annotation import DS, SPLITS, full_scores, rater, seed

CONSENT = "Test consent text: ratings are pseudonymous and can be withdrawn."


def _issue(*args: Any, **kw: Any) -> tuple[AnnotationPacket, str, list[dict[str, Any]]]:
    return offline.issue_packet(*args, consent_text=CONSENT, **kw)


@pytest.fixture
def sess(db: DB) -> Iterator[Session]:
    with db.sf() as s:
        seed(s)
        yield s


def _result(ann_id: str, seconds: float = 3600, **kw: Any) -> dict[str, Any]:
    scores = kw.pop("scores", None) or full_scores()
    return {
        "annotation_id": ann_id,
        "scores": [
            {
                "dimension": s.dimension,
                "status": s.status,
                "score": s.score,
                "confidence": s.confidence,
                "evidence": s.evidence,
                "note": s.note,
            }
            for s in scores
        ],
        "confidence": 0.7,
        "flags": kw.pop("flags", []),
        "notes": None,
        "enjoyment": kw.pop("enjoyment", 4),
        "client_duration_seconds": seconds,
    }


def _return(packet: AnnotationPacket, secret: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format": offline.RETURN_FORMAT,
        "format_version": "1.0",
        "packet_id": packet.id,
        "secret": secret,
        "consent": {"agreed": True, "text_sha256": packet.consent_sha256},
        "results": results,
    }


def test_onboarding_packet_is_calibration_only_then_dev(sess: Session) -> None:
    r = rater(sess, "r1", active=False)
    packet, secret, payloads = _issue(sess, r, DS, max_items=3)
    assert [p["video_id"] for p in payloads] == ["cal_1"] and payloads[0]["is_calibration"]
    with pytest.raises(svc.AnnotationError, match="calibration packet first"):
        _issue(sess, r, DS)
    out = offline.import_return(sess, _return(packet, secret, [_result(packet.annotation_ids[0])]))
    assert out["accepted"] == packet.annotation_ids and out["rater_status"] == "active"
    _, _, dev = _issue(sess, r, DS, max_items=3)
    assert len(dev) == 3 and {SPLITS[p["video_id"]] for p in dev} == {"dev"}


def test_items_are_never_assigned_to_more_raters_than_needed(sess: Session) -> None:
    raters = [rater(sess, f"r{i}") for i in range(4)]
    assigned: Counter[str] = Counter()
    for r in raters[:3]:
        _, _, payloads = _issue(sess, r, DS, max_items=5)
        assigned.update(p["video_id"] for p in payloads)
    assert assigned == Counter({"dev_1": 3, "dev_2": 3, "dev_3": 3})  # dev only, 3 raters each
    with pytest.raises(svc.AnnotationError, match="nothing to assign"):
        _issue(sess, raters[3], DS)
    _, _, both = _issue(sess, raters[3], DS, splits=("dev", "test"))
    assert {SPLITS[p["video_id"]] for p in both} == {"test"}


def test_packet_file_is_blind_offline_and_escaped(sess: Session) -> None:
    r = rater(sess, "r1")
    packet, secret, payloads = _issue(sess, r, DS, max_items=1)
    payloads[0]["segments"][0]["text"] = "</script><script>alert(1)</script>"
    html = offline.render_packet(packet, r, payloads, secret, CONSENT)
    assert "default-src 'none'" in html  # no network access from the file
    assert html.count("</script>") == 2  # the data cannot close the script block early
    data = json.loads(
        re.search(r'<script id="packet" type="application/json">(.*?)</script>', html, re.S).group(1)  # type: ignore[union-attr]
    )
    assert data["items"][0]["segments"][0]["text"].startswith("</script>")
    forbidden = {"evaluation", "overall_score", "view_count", "like_count", "subscriber_count", "annotations"}
    for item in data["items"]:
        assert not forbidden & set(item)
    assert not re.search(r"\b(view_count|views|like_count|subscriber_count|subscribers)\b", html, re.I)


def test_items_without_rights_or_content_are_never_packed(
    sess: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = svc.video_pending_uses
    monkeypatch.setattr(
        svc, "video_pending_uses", lambda s, vid: {"external_annotation"} if vid == "dev_1" else real(s, vid)
    )
    revoke_video_sources(sess, "dev_2", "test revocation")
    _, _, payloads = _issue(sess, rater(sess, "r1"), DS, max_items=5)
    assert [p["video_id"] for p in payloads] == ["dev_3"]


class TestImport:
    def test_valid_return_is_recorded_with_offline_and_client_timing_flags(self, sess: Session) -> None:
        r = rater(sess, "r1")
        packet, secret, _ = _issue(sess, r, DS, max_items=2)
        out = offline.import_return(
            sess, _return(packet, secret, [_result(a) for a in packet.annotation_ids])
        )
        assert sorted(out["accepted"]) == sorted(packet.annotation_ids) and not out["rejected"]
        for a in packet.annotation_ids:
            ann = sess.get(Annotation, a)
            assert ann is not None and ann.status == "submitted" and ann.duration_seconds == 3600
            assert {"offline_packet", "client_timing"} <= set(ann.flags) and ann.enjoyment == 4
        assert packet.status == "returned"
        exported = svc.export_labels(sess, DS)
        assert len(exported["labels"]) == 2 * len(svc.ANNOTATION_DIMENSIONS)
        assert all(row["label_source"] == "human_annotation" for row in exported["labels"])
        assert len(exported["enjoyment"]) == 2  # kept apart from quality labels

    def test_wrong_secret_or_unknown_packet_is_refused(self, sess: Session) -> None:
        r = rater(sess, "r1")
        packet, secret, _ = _issue(sess, r, DS, max_items=1)
        res = [_result(packet.annotation_ids[0])]
        with pytest.raises(svc.AnnotationError, match="secret"):
            offline.import_return(sess, _return(packet, "pkt_guess", res))
        with pytest.raises(svc.AnnotationError, match="unknown packet"):
            offline.import_return(sess, {**_return(packet, secret, res), "packet_id": "nope"})
        with pytest.raises(svc.AnnotationError, match="not an annotation return"):
            offline.import_return(sess, {"format": "something-else"})

    def test_each_item_is_validated_on_its_own(self, sess: Session) -> None:
        r = rater(sess, "r1")
        other = rater(sess, "r2")
        packet, secret, _ = _issue(sess, r, DS, max_items=2)
        theirs, _, _ = _issue(sess, other, DS, max_items=1)
        good, bad = packet.annotation_ids
        bad_scores = full_scores()
        bad_scores[0].evidence = [
            {"segment_index": 4, "quote": "words that are not in the transcript at all"}
        ]
        out = offline.import_return(
            sess,
            _return(
                packet,
                secret,
                [_result(good), _result(bad, scores=bad_scores), _result(theirs.annotation_ids[0])],
            ),
        )
        assert out["accepted"] == [good]
        reasons = {x["annotation_id"]: x["reason"] for x in out["rejected"]}
        assert (
            "quote not found" in reasons[bad]
            and "not part of this packet" in reasons[theirs.annotation_ids[0]]
        )
        assert out["still_open"] == [bad] and packet.status == "issued"
        # The rater fixes the item and sends the file again: the accepted one is not recorded twice.
        again = offline.import_return(sess, _return(packet, secret, [_result(good), _result(bad)]))
        assert again["accepted"] == [bad] and again["already_imported"] == [good]
        assert packet.status == "returned"

    def test_qc_still_applies_to_offline_work(self, sess: Session) -> None:
        r = rater(sess, "r1")
        packet, secret, _ = _issue(sess, r, DS, max_items=2)
        a, b = packet.annotation_ids
        offline.import_return(
            sess,
            _return(packet, secret, [_result(a, seconds=5), _result(b, scores=full_scores(3))]),
        )
        assert "too_fast" in sess.get(Annotation, a).flags  # type: ignore[union-attr]
        assert "straight_lining" in sess.get(Annotation, b).flags  # type: ignore[union-attr]

    def test_release_frees_unreturned_items_and_blocks_late_returns(self, sess: Session) -> None:
        r = rater(sess, "r1")
        packet, secret, _ = _issue(sess, r, DS, max_items=3)
        offline.import_return(sess, _return(packet, secret, [_result(packet.annotation_ids[0])]))
        assert offline.release_packet(sess, packet.id) == 2
        assert sess.scalars(select(Annotation).where(Annotation.status == "in_progress")).all() == []
        with pytest.raises(svc.AnnotationError, match="released"):
            offline.import_return(sess, _return(packet, secret, [_result(packet.annotation_ids[1])]))
        assert sess.get(Annotation, packet.annotation_ids[0]).status == "submitted"  # type: ignore[union-attr]

    def test_suspended_raters_cannot_return_work(self, sess: Session) -> None:
        r = rater(sess, "r1")
        packet, secret, _ = _issue(sess, r, DS, max_items=1)
        sess.get(Rater, r.id).status = "suspended"  # type: ignore[union-attr]
        with pytest.raises(svc.AnnotationError, match="suspended"):
            offline.import_return(sess, _return(packet, secret, [_result(packet.annotation_ids[0])]))


class TestRelevance:
    def test_dev_items_collect_relevance_and_export_it_in_benchmark_format(self, sess: Session) -> None:
        from avd.bench.relevance import Judgment

        r = rater(sess, "r1")
        packet, secret, payloads = _issue(sess, r, DS, max_items=1)
        assert len(payloads[0]["relevance_queries"]) == 10
        res = _result(packet.annotation_ids[0])
        res["relevance"] = [
            {"query_id": "q01", "grade": 2},
            {"query_id": "q02", "grade": 0},
            {"query_id": "q03", "grade": None},
        ]
        out = offline.import_return(sess, _return(packet, secret, [res]))
        assert out["relevance_judgments"] == 2  # "can't tell" is not a judgment
        rows = offline.export_relevance(sess)
        assert [(x["query_id"], x["grade"], x["rater"]) for x in rows] == [("q01", 2, "r1"), ("q02", 0, "r1")]
        for row in rows:
            Judgment.model_validate(row)

    def test_bad_relevance_rejects_the_item_and_test_items_get_no_queries(self, sess: Session) -> None:
        r = rater(sess, "r1")
        packet, secret, _ = _issue(sess, r, DS, max_items=1)
        res = _result(packet.annotation_ids[0])
        res["relevance"] = [{"query_id": "q01", "grade": 3}]
        out = offline.import_return(sess, _return(packet, secret, [res]))
        assert not out["accepted"] and "0, 1 or 2" in out["rejected"][0]["reason"]
        assert sess.get(Annotation, packet.annotation_ids[0]).status == "in_progress"  # type: ignore[union-attr]
        assert offline.export_relevance(sess) == []
        for other in [rater(sess, f"x{i}") for i in range(3)]:
            _issue(sess, other, DS, max_items=3)
        _, _, test_items = _issue(sess, rater(sess, "late"), DS, splits=("dev", "test"))
        assert test_items and all(p["relevance_queries"] == [] for p in test_items)


def test_consent_is_required_shown_and_checked(sess: Session) -> None:
    r = rater(sess, "r1")
    with pytest.raises(svc.AnnotationError, match="consent text is required"):
        offline.issue_packet(sess, r, DS, max_items=1, consent_text="  ")
    packet, secret, payloads = _issue(sess, r, DS, max_items=1)
    html = offline.render_packet(packet, r, payloads, secret, CONSENT)
    assert CONSENT in html and "consent-box" in html
    with pytest.raises(svc.AnnotationError, match="not the consent text"):
        offline.render_packet(packet, r, payloads, secret, "some other text")
    res = [_result(packet.annotation_ids[0])]
    for consent in (
        None,
        {"agreed": False, "text_sha256": packet.consent_sha256},
        {"agreed": True, "text_sha256": "0" * 64},
    ):
        data = _return(packet, secret, res)
        data["consent"] = consent
        with pytest.raises(svc.AnnotationError, match="has not agreed"):
            offline.import_return(sess, data)
    assert sess.get(Annotation, packet.annotation_ids[0]).status == "in_progress"  # type: ignore[union-attr]
    offline.import_return(sess, _return(packet, secret, res))
    ann = sess.get(Annotation, packet.annotation_ids[0])
    assert ann is not None and f"consent:{packet.consent_sha256[:12]}" in ann.flags


class TestRobustness:
    @pytest.mark.parametrize(
        "mutate,expect",
        [
            (lambda d: d.update(results="x"), "malformed ratings file"),
            (lambda d: d.update(results=["x"]), "malformed result"),
            (lambda d: d["results"][0].update(scores="x"), "malformed scores"),
            (lambda d: d["results"][0].update(scores=["x"]), "malformed scores"),
            (lambda d: d["results"][0]["scores"][0].update(evidence="x"), "malformed evidence"),
            (lambda d: d["results"][0].update(relevance="x"), "malformed relevance"),
            (lambda d: d.update(consent="yes"), "has not agreed"),
        ],
    )
    def test_malformed_returns_are_refused_not_crashing(
        self, sess: Session, mutate: Any, expect: str
    ) -> None:
        r = rater(sess, "r1")
        packet, secret, _ = _issue(sess, r, DS, max_items=1)
        data = _return(packet, secret, [_result(packet.annotation_ids[0])])
        mutate(data)
        try:
            out = offline.import_return(sess, data)
        except svc.AnnotationError as exc:
            assert expect in str(exc)
        else:
            assert not out["accepted"] and expect in out["rejected"][0]["reason"]
        assert sess.get(Annotation, packet.annotation_ids[0]).status == "in_progress"  # type: ignore[union-attr]
        with pytest.raises(svc.AnnotationError, match="malformed"):
            offline.import_return(sess, [])  # type: ignore[arg-type]

    def test_a_return_sent_to_another_coordinator_database_is_refused(
        self, sess: Session, tmp_path: Path
    ) -> None:
        packet, secret, _ = _issue(sess, rater(sess, "r1"), DS, max_items=1)
        data = _return(packet, secret, [_result(packet.annotation_ids[0])])
        (tmp_path / "other").mkdir()
        other = _make_db(tmp_path / "other", None)
        with other.sf() as s2:
            seed(s2)
            with pytest.raises(svc.AnnotationError, match="unknown packet"):
                offline.import_return(s2, data)

    def test_missing_evidence_is_flagged_and_disagreement_reaches_adjudication(self, sess: Session) -> None:
        returned = []
        for name, depth in (("r1", 1), ("r2", 5), ("r3", 3)):
            packet, secret, _ = _issue(sess, rater(sess, name), DS, max_items=3)
            ann = next(a for a in packet.annotation_ids if sess.get(Annotation, a).video_id == "dev_1")  # type: ignore[union-attr]
            scores = full_scores(depth=depth)
            if name == "r3":  # scored without citing anything
                scores = [
                    x if x.dimension != "depth" else svc.ScoreInput("depth", "scored", 3, 0.5, [], None)
                    for x in scores
                ]
            offline.import_return(sess, _return(packet, secret, [_result(ann, scores=scores)]))
            returned.append(ann)
        assert "missing_evidence:depth" in sess.get(Annotation, returned[2]).flags  # type: ignore[union-attr]
        disputed = svc.disagreements(sess, DS)
        assert any(d["video_id"] == "dev_1" and d["dimension"] == "depth" for d in disputed), disputed


def test_calibration_can_start_with_one_episode_shortest_first(sess: Session) -> None:
    from avd.content.models import SourceType
    from avd.content.store import add_transcript_source
    from avd.db.models import DatasetItem, Video
    from tests.conftest import LECTURE, make_transcript

    sess.add(
        Video(
            id="cal_long",
            channel_id="lic:show",
            title="Long practice",
            platform="example",
            is_synthetic=False,
        )
    )
    sess.flush()
    add_transcript_source(
        sess,
        video_id="cal_long",
        transcript=make_transcript(LECTURE * 3),
        source_type=SourceType.LICENSED_DATASET,
        authorization_basis="test licence",
        license="CC-BY-4.0",
    )
    sess.add(
        DatasetItem(
            dataset_id=DS,
            video_id="cal_long",
            split="calibration",
            group_key="g_cal",
            is_calibration_example=True,
        )
    )
    sess.flush()
    r = rater(sess, "r1", active=False)
    first, secret, payloads = _issue(sess, r, DS, max_items=1)
    assert [p["video_id"] for p in payloads] == ["cal_1"]  # the shorter practice episode
    offline.import_return(sess, _return(first, secret, [_result(first.annotation_ids[0])]))
    assert r.status == "onboarding"  # one practice episode is not the whole calibration
    _, _, second = _issue(sess, r, DS, max_items=1)
    assert [p["video_id"] for p in second] == ["cal_long"]


def test_a_consent_text_still_marked_draft_cannot_be_issued(sess: Session) -> None:
    template = (Path(__file__).resolve().parents[2] / "docs" / "templates" / "rating-consent.txt").read_text()
    assert offline.CONSENT_DRAFT_MARKER in template  # the repository template awaits approval
    with pytest.raises(svc.AnnotationError, match="still marked as a draft"):
        offline.issue_packet(sess, rater(sess, "r1"), DS, max_items=1, consent_text=template)
    approved = "\n".join(line for line in template.splitlines() if offline.CONSENT_DRAFT_MARKER not in line)
    packet, _, _ = offline.issue_packet(sess, rater(sess, "r2"), DS, max_items=1, consent_text=approved)
    assert packet.consent_sha256 == offline.consent_digest(approved)
