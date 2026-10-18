"""Phase 2 P5: genre v2 (expanded taxonomy, distributions, mixed/ambiguous/unknown, overrides) and
aggregation v2 (blended and neutral profiles)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from avd.evaluation.aggregation import aggregate
from avd.evaluation.genre import KeywordGenreClassifier, apply_override, speaker_turn_cue
from avd.evaluation.profiles import PROFILES, blend, get_profile, profile_for, profile_from_dict
from avd.evaluation.schema import (
    AGGREGATION_VERSION,
    GENRE_PARENT,
    Dimension,
    DimensionScore,
    DimensionStatus,
    Genre,
    GenreAssessment,
    GenreStatus,
)

C = KeywordGenreClassifier()


def test_every_genre_has_a_profile_and_v2_genres_have_parents() -> None:
    assert set(PROFILES) == set(Genre)
    for g in (
        Genre.LECTURE,
        Genre.PROGRAMMING,
        Genre.PRODUCT_REVIEW,
        Genre.INTERVIEW_PODCAST,
        Genre.COMEDY,
        Genre.PERSONAL_STORYTELLING,
    ):
        assert g in GENRE_PARENT


@pytest.mark.parametrize(
    ("title", "tags", "expected"),
    [
        ("Lecture 7: eigenvalues and the spectral theorem", ["math"], Genre.LECTURE),
        ("Refactoring a Python API: debugging async code", ["python", "programming"], Genre.PROGRAMMING),
        ("Phone X review: battery life, specs and first impressions", ["review"], Genre.PRODUCT_REVIEW),
        ("Stand-up comedy: jokes about airports", ["comedy"], Genre.COMEDY),
        ("My story: what happened to me when I was 20", ["storytime"], Genre.PERSONAL_STORYTELLING),
    ],
)
def test_v2_genres_are_recognised(title: str, tags: list[str], expected: Genre) -> None:
    assert C.classify(title, "", tags, None).genre == expected


def test_speaker_turns_indicate_a_conversation() -> None:
    convo = "\n".join(["Ana Ruiz: So what got you started?", "Ben Okafor: Honestly, a broken build."] * 8)
    assert speaker_turn_cue(convo) == (2, 16)
    g = C.classify("Episode 12", "", [], None, transcript_text=convo)
    assert g.genre == Genre.INTERVIEW_PODCAST and g.status == GenreStatus.CONFIDENT
    assert speaker_turn_cue("Note: this is a lecture. Note: it continues.") == (0, 0)


def test_statuses_unknown_ambiguous_mixed() -> None:
    unknown = C.classify("Untitled", "", [], None)
    assert unknown.status == GenreStatus.UNKNOWN and unknown.genre == Genre.OTHER
    weak = C.classify("A quick guide", "", [], None)  # one weak keyword: too little evidence
    assert weak.status == GenreStatus.AMBIGUOUS
    mixed = C.classify("Python programming interview podcast", "", ["podcast", "python"], None)
    assert mixed.status == GenreStatus.MIXED and mixed.secondary is not None
    assert {mixed.genre, mixed.secondary} == {Genre.PROGRAMMING, Genre.INTERVIEW_PODCAST}
    assert abs(sum(mixed.distribution.values()) - 1) < 0.01


def _dims(**scores: float) -> list[DimensionScore]:
    return [
        DimensionScore(
            dimension=Dimension(k),
            status=DimensionStatus.SCORED,
            score=v,
            confidence=0.8,
            justification="t",
            method="t",
        )
        for k, v in scores.items()
    ]


def test_blended_profile_is_share_weighted_and_reported() -> None:
    p = blend({Genre.COMEDY: 0.6, Genre.LECTURE: 0.4})
    assert p.genre == Genre.COMEDY
    assert p.weight(Dimension.STORYTELLING) == pytest.approx(0.6 * 1.4 + 0.4 * 0.3, abs=1e-4)
    # factual accuracy is not applicable to comedy (0) but is to lectures: the mixture keeps it
    assert p.weight(Dimension.FACTUAL_ACCURACY) == pytest.approx(0.4 * 1.6, abs=1e-4)
    agg = aggregate(
        _dims(depth=4, factual_accuracy=4, information_density=4, informational_value=4, storytelling=2), p
    )
    assert agg.method == AGGREGATION_VERSION and agg.profile_detail["description"].startswith("Mixture")
    assert any("blended profile" in n for n in agg.notes)
    assert profile_from_dict(agg.profile_detail).dimension_weights == p.dimension_weights


def test_profile_for_each_status() -> None:
    a = GenreAssessment(genre=Genre.SCIENCE, confidence=0.3, method="t", status=GenreStatus.AMBIGUOUS)
    assert profile_for(a) is PROFILES[Genre.OTHER]
    assert profile_for(a.model_copy(update={"status": GenreStatus.CONFIDENT})) is PROFILES[Genre.SCIENCE]
    m = a.model_copy(
        update={
            "status": GenreStatus.MIXED,
            "secondary": Genre.COMEDY,
            "distribution": {"science": 0.5, "comedy": 0.4},
        }
    )
    assert profile_for(m).description.startswith("Mixture: science 56% + comedy 44%")


def test_override_replaces_genre_but_keeps_distribution() -> None:
    g = C.classify("Python programming interview podcast", "", [], None)
    o = apply_override(g, "lecture", "editor_1")
    assert o.genre == Genre.LECTURE and o.status == GenreStatus.OVERRIDDEN and o.confidence == 1.0
    assert o.method == "manual_override:editor_1" and o.distribution == g.distribution
    assert apply_override(g, None, None) is g


def test_profiles_never_let_production_quality_change_content_weights() -> None:
    for p in PROFILES.values():
        assert p.group_weights[next(iter(p.group_weights))] >= 0  # sanity
        for d in Dimension:
            if d.group.value == "technical":
                assert d.value in {"audio_quality", "visual_quality", "editing", "visual_readability"}
    assert (
        get_profile(Genre.LECTURE).weight(Dimension.FACTUAL_ACCURACY)
        > get_profile(Genre.COMEDY).weight(Dimension.FACTUAL_ACCURACY)
        == 0
    )


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from avd.config import get_settings
    from avd.db.models import Base
    from avd.db.session import get_engine, get_sessionmaker
    from avd.embeddings.base import get_embedder

    monkeypatch.setenv("AVD_DATABASE_URL", f"sqlite:///{tmp_path / 'g.db'}")
    monkeypatch.setenv("AVD_ADMIN_TOKEN", "test-admin")
    monkeypatch.setenv("AVD_MEDIA_CACHE_DIR", str(tmp_path / "media"))
    monkeypatch.delenv("AVD_REDIS_URL", raising=False)
    for f in (get_settings, get_engine, get_sessionmaker, get_embedder):
        f.cache_clear()
    Base.metadata.create_all(get_engine())
    from avd.api.app import create_app

    with TestClient(create_app()) as c:
        yield c
    get_engine().dispose()
    for f in (get_settings, get_engine, get_sessionmaker, get_embedder):
        f.cache_clear()


def test_genre_override_reevaluates_with_the_corrected_profile(api: TestClient) -> None:
    admin = {"X-Admin-Token": "test-admin"}
    assert api.post("/v1/admin/seed-demo", headers=admin).status_code == 200
    api.post("/v1/admin/worker/run", headers=admin)
    before = api.get("/v1/videos/syn_sci_001/evaluation").json()
    r = api.put(
        "/v1/admin/videos/syn_sci_001/genre", json={"genre": "lecture", "by": "editor_1"}, headers=admin
    )
    assert r.status_code == 200 and r.json()["requeued"]
    out = api.post("/v1/admin/worker/run", headers=admin).json()["outcomes"]
    assert out.get("succeeded", 0) >= 2  # stage A and the forced stage B
    after = api.get("/v1/videos/syn_sci_001/evaluation").json()
    assert after["genre"] == "lecture" and after["evaluation_id"] != before["evaluation_id"]
    assert after["overall"]["method"] == AGGREGATION_VERSION
    assert (
        api.put(
            "/v1/admin/videos/nope/genre", json={"genre": "lecture", "by": "x"}, headers=admin
        ).status_code
        == 404
    )
    assert (
        api.put("/v1/admin/videos/syn_sci_001/genre", json={"genre": "lecture", "by": "x"}).status_code == 403
    )
