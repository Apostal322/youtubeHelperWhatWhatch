from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from avd.config import REPO_ROOT, LLMProviderName, Settings
from avd.content.models import Transcript, TranscriptSegment
from avd.db.models import Base
from avd.db.session import make_engine
from avd.evaluation.inputs import EvaluationInput
from avd.evaluation.schema import Genre, GenreAssessment, Modality
from avd.llm.base import LLMRequest, LLMResponse

FIXTURES = REPO_ROOT / "data" / "fixtures"
PG_URL = os.environ.get("AVD_TEST_DATABASE_URL")
REDIS_URL = os.environ.get("AVD_TEST_REDIS_URL")


@dataclass
class DB:
    settings: Settings
    sf: sessionmaker[Session]
    dialect: str


def make_settings(tmp_path: Path, url: str | None = None, **kw: Any) -> Settings:
    base: dict[str, Any] = {
        "database_url": url or f"sqlite:///{tmp_path / 'test.db'}",
        "media_cache_dir": tmp_path / "media",
        "fixtures_dir": FIXTURES,
        "llm_provider": LLMProviderName.HEURISTIC,
        "redis_url": None,
        "admin_token": "test-admin",
        "log_json": False,
    }
    base.update(kw)
    return Settings(**base)


def _make_db(tmp_path: Path, url: str | None) -> DB:
    settings = make_settings(tmp_path, url)
    engine = make_engine(settings.database_url)
    if engine.dialect.name == "postgresql":
        with engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return DB(
        settings=settings, sf=sessionmaker(bind=engine, expire_on_commit=False), dialect=engine.dialect.name
    )


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[DB]:
    """A fresh database: SQLite always; PostgreSQL+pgvector when AVD_TEST_DATABASE_URL is set."""
    if request.param == "postgres":
        if not PG_URL:
            pytest.skip("AVD_TEST_DATABASE_URL not set")
        d = _make_db(tmp_path, PG_URL)
    else:
        d = _make_db(tmp_path, None)
    yield d
    d.sf.kw["bind"].dispose()


@pytest.fixture
def sqlite_db(tmp_path: Path) -> DB:
    return _make_db(tmp_path, None)


def make_transcript(texts: list[str], seg_seconds: float = 8.0, language: str = "en") -> Transcript:
    return Transcript(
        language=language,
        segments=[
            TranscriptSegment(start=i * seg_seconds, end=(i + 1) * seg_seconds, text=t)
            for i, t in enumerate(texts)
        ],
    )


LECTURE = [
    "Today we will learn how photosynthesis works. First, let's start with light absorption.",
    "Chlorophyll absorbs red and blue light because of its electronic structure, which means green light is reflected.",
    "For example, imagine a leaf in sunlight. The energy excites electrons in photosystem two.",
    "However, this is not always efficient; efficiency depends on temperature and roughly one percent of light is stored.",
    "Next, the Calvin cycle fixes carbon dioxide. This is called carbon fixation, and the key enzyme is RuBisCO.",
    "In summary, light reactions make ATP and NADPH, and the Calvin cycle uses them to build sugars.",
] * 5


def make_input(
    transcript: Transcript | None = None,
    *,
    title: str = "How photosynthesis works: a lecture",
    genre: Genre = Genre.SCIENCE,
    audio_path: str | None = None,
) -> EvaluationInput:
    t = transcript if transcript is not None else make_transcript(LECTURE)
    mods = {Modality.TRANSCRIPT} if t is not None else set()
    if audio_path:
        mods.add(Modality.AUDIO)
    return EvaluationInput(
        video_id="vid_test_01",
        title=title,
        description="A biology lecture.",
        duration_seconds=240,
        language="en",
        tags=("biology",),
        genre=GenreAssessment(genre=genre, confidence=0.9, method="test"),
        transcript=t,
        transcript_source_id="src-1" if t else None,
        audio_path=audio_path,
        audio_sha256="x" if audio_path else None,
        audio_source_id="src-2" if audio_path else None,
        modalities=frozenset(mods),
        content_source_ids=("src-1",),
    )


class ScriptedProvider:
    """Test double for an LLM provider: ``respond(request, call_index) -> dict | Exception``."""

    name = "scripted"

    def __init__(self, respond: Callable[[LLMRequest, int], Any], model: str = "claude-opus-5-5") -> None:
        self.respond = respond
        self.calls: list[LLMRequest] = []
        self.model = model

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        out = self.respond(request, len(self.calls) - 1)
        if isinstance(out, Exception):
            raise out
        return LLMResponse(
            data=out,
            provider=self.name,
            requested_model=request.model,
            served_model=self.model,
            input_tokens=1000,
            output_tokens=500,
            stop_reason="end_turn",
        )


def load_fixture(video_id: str) -> dict[str, Any]:
    return json.loads((FIXTURES / "videos" / f"{video_id}.json").read_text())
