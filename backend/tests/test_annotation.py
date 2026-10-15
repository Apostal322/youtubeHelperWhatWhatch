"""Phase 2 P2: human annotation workflow (blind assignment, QC, adjudication, export).

All "ratings" here are test inputs exercising the workflow; they are not, and are never exported
as, labels for any real item.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from avd.annotation import service as svc
from avd.content.models import SourceType
from avd.content.store import add_transcript_source
from avd.datasets.importer import revoke_video_sources
from avd.db.models import Annotation, Dataset, DatasetItem, Rater, Video
from avd.ingest import upsert_channel
from tests.conftest import DB, LECTURE, make_transcript

DS = "pilot_test@v1"
SPLITS = {"cal_1": "calibration", "dev_1": "dev", "dev_2": "dev", "dev_3": "dev", "tst_1": "test", "tst_2": "test"}


def seed(s: Session) -> None:
    upsert_channel(s, "lic:show", "Show")
    s.add(Dataset(id=DS, name="pilot_test", version="v1", manifest_sha256="0" * 64, split_strategy="group_by_show", split_seed=1))
    for vid, split in SPLITS.items():
        s.add(Video(id=vid, channel_id="lic:show", title=f"Episode {vid}", platform="example", is_synthetic=False))
        s.flush()
        add_transcript_source(
            s, video_id=vid, transcript=make_transcript(LECTURE), source_type=SourceType.LICENSED_DATASET,
            authorization_basis="test licence", license="CC-BY-4.0",
        )
        s.add(DatasetItem(dataset_id=DS, video_id=vid, split=split, group_key=f"g_{split}", is_calibration_example=split == "calibration"))
    s.flush()


def rater(s: Session, name: str, role: str = "rater", active: bool = True) -> Rater:
    r, _ = svc.create_rater(s, name, ["biology"], role)
    if active:
        r.status = "active"
    return r


VARIED = (3, 4, 2, 3, 4, 3, 2, 4, 3, 3)


def full_scores(score: int | None = 0, quote: bool = True, **overrides: int | None) -> list[svc.ScoreInput]:
    """score=0: a varied, non-straight-lined pattern; None: abstain everywhere; 1-5: constant."""
    out = []
    for i, d in enumerate(svc.ANNOTATION_DIMENSIONS):
        sc = overrides.get(d, VARIED[i % len(VARIED)] if score == 0 else score)
        out.append(
            svc.ScoreInput(
                dimension=d, status="scored" if sc is not None else "abstain", score=sc, confidence=0.7,
                evidence=[{"segment_index": 4, "quote": "the key enzyme is RuBisCO"}] if (quote and sc is not None) else [],
            )
        )
    return out


def submit(s: Session, r: Rater, ann: Annotation, slow: bool = True, **kw: object) -> Annotation:
    if slow:  # pretend the rater spent an hour (the server measures time; tests move the start back)
        ann.started_at = datetime.now(UTC) - timedelta(hours=1)
    return svc.submit_annotation(s, r, ann.id, kw.pop("scores", None) or full_scores(), 0.8, [], None)  # type: ignore[arg-type]


@pytest.fixture
def sess(db: DB) -> Iterator[Session]:
    with db.sf() as s:
        seed(s)
        yield s


def test_onboarding_rater_gets_calibration_first_then_becomes_active(sess: Session) -> None:
    r = rater(sess, "r1", active=False)
    a = svc.next_assignment(sess, r, DS)
    assert a is not None and a.video_id == "cal_1" and r.status == "onboarding"
    assert svc.next_assignment(sess, r, DS) is a  # in-progress work is returned, not duplicated
    submit(sess, r, a)
    b = svc.next_assignment(sess, r, DS)
    assert r.status == "active" and b is not None and SPLITS[b.video_id] in ("dev", "test")


def test_payload_is_blind(sess: Session) -> None:
    r = rater(sess, "r1")
    a = svc.next_assignment(sess, r, DS)
    assert a is not None
    p = svc.item_payload(sess, a)
    forbidden = {"evaluation", "scores", "overall_score", "view_count", "like_count", "subscriber_count", "annotations"}
    assert not forbidden & set(p)
    assert p["segments"][0]["text"] and len(p["rubric"]) == len(svc.ANNOTATION_DIMENSIONS)
    assert all(set(r_["anchors"]) == {"1", "2", "3", "4", "5"} for r_ in p["rubric"])


class TestSubmission:
    def test_valid_submission_verifies_evidence_and_records_server_time(self, sess: Session) -> None:
        r = rater(sess, "r1")
        a = svc.next_assignment(sess, r, DS)
        assert a is not None
        done = submit(sess, r, a)
        assert done.status == "submitted" and done.duration_seconds and done.duration_seconds >= 3500
        ev = done.scores[0].evidence[0]
        assert ev["segment_index"] == 4 and ev["start"] == 32.0 and ev["end"] == 40.0
        assert done.flags == []

    @pytest.mark.parametrize(
        ("scores", "msg"),
        [
            (full_scores()[:-1], "missing"),
            ([svc.ScoreInput(d, "abstain", score=4) for d in svc.ANNOTATION_DIMENSIONS], "iff"),
            ([svc.ScoreInput(d, "scored", score=7) for d in svc.ANNOTATION_DIMENSIONS], "1-5"),
            (
                [svc.ScoreInput(d, "scored", score=3, evidence=[{"segment_index": 0, "quote": "an invented quote never said"}]) for d in svc.ANNOTATION_DIMENSIONS],
                "quote not found",
            ),
            ([svc.ScoreInput(d, "maybe") for d in svc.ANNOTATION_DIMENSIONS], "invalid status"),
        ],
    )
    def test_invalid_submissions_are_rejected(self, sess: Session, scores: list[svc.ScoreInput], msg: str) -> None:
        r = rater(sess, "r1")
        a = svc.next_assignment(sess, r, DS)
        assert a is not None
        with pytest.raises(svc.AnnotationError, match=msg):
            svc.submit_annotation(sess, r, a.id, scores, None, [], None)

    def test_cannot_submit_twice_or_for_someone_else(self, sess: Session) -> None:
        r1, r2 = rater(sess, "r1"), rater(sess, "r2")
        a = svc.next_assignment(sess, r1, DS)
        assert a is not None
        with pytest.raises(svc.AnnotationError, match="not found"):
            svc.submit_annotation(sess, r2, a.id, full_scores(), None, [], None)
        submit(sess, r1, a)
        with pytest.raises(svc.AnnotationError, match="already"):
            svc.submit_annotation(sess, r1, a.id, full_scores(), None, [], None)

    def test_qc_flags(self, sess: Session) -> None:
        r = rater(sess, "r1")
        a = svc.next_assignment(sess, r, DS)
        assert a is not None
        done = svc.submit_annotation(sess, r, a.id, full_scores(score=5, quote=False), None, [], None)
        assert "too_fast" in done.flags and "straight_lining" in done.flags
        assert any(f.startswith("missing_evidence:") for f in done.flags)


def test_assignment_balances_items_and_never_repeats_unflagged(sess: Session) -> None:
    raters = [rater(sess, f"r{i}") for i in range(4)]
    seen: dict[str, list[str]] = {}
    for _ in range(5):
        for r in raters:
            a = svc.next_assignment(sess, r, DS)
            if a is None:
                continue
            submit(sess, r, a)
            seen.setdefault(r.id, []).append(a.video_id)
    counts = svc._submitted_counts(sess, DS)
    assert max(counts.values()) <= svc.RATERS_PER_ITEM
    for vids in seen.values():
        firsts = [v for v in vids]
        assert len(firsts) == len(set(firsts))  # 5 dev/test items, no repeats before REPEAT_AFTER


def test_repeat_assignment_is_blind_and_kept_out_of_labels(sess: Session) -> None:
    r = rater(sess, "r1")
    first = []
    for _ in range(svc.REPEAT_AFTER):
        a = svc.next_assignment(sess, r, DS)
        assert a is not None and not a.is_repeat
        submit(sess, r, a)
        first.append(a.video_id)
    rep = svc.next_assignment(sess, r, DS)
    assert rep is not None and rep.is_repeat and rep.video_id in first
    submit(sess, r, rep, scores=full_scores(score=4))
    exp = svc.export_labels(sess, DS)
    assert {x["annotation_id"] for x in exp["repeats"]} == {rep.id}
    assert rep.id not in {x.get("annotation_id") for x in exp["labels"]}
    assert svc.qc_report(sess, DS)["repeat_pairs"] == 1


def test_disagreement_goes_to_an_independent_adjudicator(sess: Session) -> None:
    r1, r2 = rater(sess, "r1"), rater(sess, "r2")
    adj_rater, adj = rater(sess, "adj_who_rated", "adjudicator"), rater(sess, "adj", "adjudicator")
    a1 = svc.next_assignment(sess, r1, DS)
    assert a1 is not None
    submit(sess, r1, a1, scores=full_scores(depth=1))
    a2 = svc._start(sess, r2, DS, a1.video_id, is_repeat=False)
    submit(sess, r2, a2, scores=full_scores(depth=5))
    a3 = svc._start(sess, adj_rater, DS, a1.video_id, is_repeat=False)
    submit(sess, adj_rater, a3)
    open_ = svc.disagreements(sess, DS)
    assert [(d["video_id"], d["dimension"]) for d in open_] == [(a1.video_id, "depth")]
    with pytest.raises(svc.AnnotationError, match="only adjudicators"):
        svc.adjudicate(sess, r1, DS, a1.video_id, "depth", "score", 3, "middle")
    with pytest.raises(svc.AnnotationError, match="they rated"):
        svc.adjudicate(sess, adj_rater, DS, a1.video_id, "depth", "score", 3, "middle")
    with pytest.raises(svc.AnnotationError, match="rationale"):
        svc.adjudicate(sess, adj, DS, a1.video_id, "depth", "score", 3, "  ")
    svc.adjudicate(sess, adj, DS, a1.video_id, "depth", "score", 3, "Covers mechanisms but not edge cases.")
    assert svc.disagreements(sess, DS) == []
    labels = svc.export_labels(sess, DS)["labels"]
    assert any(x["label_source"] == "human_adjudication" and x["score"] == 3 for x in labels)


def test_pairwise_uses_items_the_rater_annotated_in_one_split(sess: Session) -> None:
    r = rater(sess, "r1")
    for _ in range(4):
        a = svc.next_assignment(sess, r, DS)
        assert a is not None
        submit(sess, r, a)
    pair = svc.next_pair(sess, r, DS, "overall")
    assert pair is not None and SPLITS[pair[0]] == SPLITS[pair[1]] != "calibration"
    svc.submit_pair(sess, r, DS, pair[0], pair[1], "overall", "a", 0.6, "clearer", 30.0)
    nxt = svc.next_pair(sess, r, DS, "overall")
    assert nxt is None or frozenset(nxt) != frozenset(pair)
    with pytest.raises(svc.AnnotationError):
        svc.submit_pair(sess, r, DS, pair[0], pair[0], "overall", "a", None, None, None)


def test_empty_dataset_exports_nothing(sess: Session) -> None:
    exp = svc.export_labels(sess, DS)
    assert exp["labels"] == [] and exp["pairs"] == [] and len(exp["items"]) == len(SPLITS)
    assert svc.qc_report(sess, DS)["submitted_annotations"] == 0


def test_revoked_items_are_not_assigned(sess: Session) -> None:
    for vid in SPLITS:
        if vid != "dev_2":
            revoke_video_sources(sess, vid, "test")
    r = rater(sess, "r1")
    a = svc.next_assignment(sess, r, DS)
    assert a is not None and a.video_id == "dev_2"
    submit(sess, r, a)
    assert svc.next_assignment(sess, r, DS) is None


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from avd.config import get_settings
    from avd.db.models import Base
    from avd.db.session import get_engine, get_sessionmaker

    monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{tmp_path / 'ann.db'}")
    monkeypatch.setenv("AVD_ADMIN_TOKEN", "test-admin")
    for f in (get_settings, get_engine, get_sessionmaker):
        f.cache_clear()
    Base.metadata.create_all(get_engine())
    with get_sessionmaker()() as s:
        seed(s)
        s.commit()
    from avd.api.app import create_app

    with TestClient(create_app()) as c:
        yield c
    get_engine().dispose()
    for f in (get_settings, get_engine, get_sessionmaker):
        f.cache_clear()


def test_api_flow(api: TestClient) -> None:
    admin = {"X-Admin-Token": "test-admin"}
    assert api.post("/v1/annotation/raters", json={"pseudonym": "r1"}).status_code == 403
    r = api.post("/v1/annotation/raters", json={"pseudonym": "r1", "expertise": ["bio"]}, headers=admin)
    assert r.status_code == 201
    tok = {"Authorization": f"Bearer {r.json()['token']}"}
    assert api.get("/v1/annotation/next", params={"dataset": DS}).status_code == 401
    item = api.get("/v1/annotation/next", params={"dataset": DS}, headers=tok).json()
    assert item["video_id"] == "cal_1" and item["is_calibration"]
    body = {
        "scores": [{"dimension": d, "status": "insufficient_evidence"} for d in svc.ANNOTATION_DIMENSIONS],
        "confidence": 0.5,
    }
    res = api.post(f"/v1/annotation/annotations/{item['annotation_id']}/submit", json=body, headers=tok)
    assert res.status_code == 200 and res.json()["status"] == "submitted"
    bad = api.post(f"/v1/annotation/annotations/{item['annotation_id']}/submit", json=body, headers=tok)
    assert bad.status_code == 422
    assert api.get("/v1/annotation/qc", params={"dataset": DS}, headers=admin).json()["submitted_annotations"] == 1
    exp = api.get("/v1/annotation/export", params={"dataset": DS}, headers=admin).json()
    assert exp["labels"] == []  # calibration annotations are never exported as labels
    page = api.get("/annotate")
    assert page.status_code == 200 and "textContent" in page.text and "innerHTML" not in page.text
