"""Long-lived dependencies for job execution (built once per worker process)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from avd.config import Settings, get_settings
from avd.content.providers import ContentProvider, build_providers
from avd.db.session import get_sessionmaker
from avd.embeddings.base import Embedder, get_embedder
from avd.evaluation.evaluators.factual import ReferenceRetriever
from avd.evaluation.genre import GenreClassifier, KeywordGenreClassifier
from avd.evaluation.pipeline import EvaluationPipeline, build_pipeline, load_reference_corpus
from avd.llm.base import LLMProvider
from avd.llm.providers import build_llm_provider
from avd.workers.budget import BudgetManager, make_quota_guard
from avd.youtube.client import YouTubeDataClient


@dataclass
class WorkerContext:
    settings: Settings
    session_factory: Callable[[], Session]
    llm: LLMProvider | None
    providers: list[ContentProvider]
    embedder: Embedder
    classifier: GenreClassifier
    budget: BudgetManager
    retriever: ReferenceRetriever | None = None
    youtube_factory: Callable[[str | None], YouTubeDataClient] | None = None
    _pipelines: dict[str, EvaluationPipeline] = field(default_factory=dict)

    def pipeline(self, stage: str) -> EvaluationPipeline:
        if stage not in self._pipelines:
            self._pipelines[stage] = build_pipeline(stage, self.settings, self.llm, self.retriever)
        return self._pipelines[stage]

    def youtube(self, job_id: str | None) -> YouTubeDataClient:
        if self.youtube_factory is None:
            raise RuntimeError("YouTube Data API is not configured (set AVD_YOUTUBE_API_KEY)")
        return self.youtube_factory(job_id)


def build_context(
    settings: Settings | None = None,
    session_factory: Callable[[], Session] | None = None,
    llm: LLMProvider | None = None,
    providers: list[ContentProvider] | None = None,
) -> WorkerContext:
    settings = settings or get_settings()
    sf: Callable[[], Session] = session_factory or get_sessionmaker()
    youtube_factory = None
    if settings.youtube_api_key is not None:
        key = settings.youtube_api_key.get_secret_value()

        def youtube_factory(job_id: str | None) -> YouTubeDataClient:
            return YouTubeDataClient(key, make_quota_guard(sf, settings, job_id))

    return WorkerContext(
        settings=settings,
        session_factory=sf,
        llm=llm if llm is not None else build_llm_provider(settings, sf),
        providers=providers if providers is not None else build_providers(),
        embedder=get_embedder(),
        classifier=KeywordGenreClassifier(),
        budget=BudgetManager(settings),
        retriever=load_reference_corpus(settings.reference_corpus_dir),
        youtube_factory=youtube_factory,
    )
