"""Environment-based configuration.

All settings are read from environment variables prefixed with ``AVD_`` (or a ``.env`` file).
Credentials are never hardcoded; when a credential is missing, the corresponding integration is
disabled and the system falls back to explicitly-labelled demo behaviour.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_repo_root() -> Path:
    """Directory containing ``data/fixtures`` (fixtures, benchmarks). Works for editable installs,
    site-packages installs (Docker: WORKDIR inside the repo) and an explicit ``AVD_REPO_ROOT``."""
    if env := os.environ.get("AVD_REPO_ROOT"):
        return Path(env)
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for p in [*here.parents, cwd, *cwd.parents]:
        if (p / "data" / "fixtures").is_dir():
            return p
    return cwd


REPO_ROOT = _find_repo_root()


class LLMProviderName(StrEnum):
    # Deterministic heuristic baseline. Produces *synthetic* evaluations, never presented as AI output.
    HEURISTIC = "heuristic"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


class TaskBackend(StrEnum):
    INLINE = "inline"  # DB-backed queue drained by `avd worker` (no Redis needed)
    CELERY = "celery"  # Celery workers + beat dispatcher, Redis broker


class EmbeddingProviderName(StrEnum):
    HASHING = "hashing"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AVD_", env_file=".env", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = True

    database_url: str = f"sqlite:///{REPO_ROOT / 'backend' / 'avd_demo.db'}"
    redis_url: str | None = None
    task_backend: TaskBackend = TaskBackend.INLINE

    # --- Evaluation -------------------------------------------------------------------------
    llm_provider: LLMProviderName = LLMProviderName.HEURISTIC
    # Stage B (deep semantic analysis) and Stage C (selective expensive analysis) models.
    llm_model_stage_b: str = "claude-opus-5-5"
    llm_model_stage_c: str = "claude-opus-5-5"
    llm_effort_stage_b: str = "medium"
    llm_effort_stage_c: str = "high"
    llm_max_output_tokens: int = 16000
    llm_requests_per_minute: int = 30
    llm_self_consistency_samples_stage_c: int = 3
    # Optional sampling temperature. Leave unset for models that reject sampling parameters
    # (e.g. claude-opus-5-5); OpenAI-compatible servers usually accept it.
    llm_temperature: float | None = None
    llm_timeout_seconds: float = 180.0
    llm_max_retries: int = 3  # bounded retries for transient errors, exponential backoff
    llm_retry_base_seconds: float = 2.0
    llm_max_concurrency: int = 4  # concurrent in-flight requests per worker process
    llm_model_revision: str | None = None  # recorded with every run (e.g. a local model build)
    # Transcripts whose estimated prompt exceeds this many tokens are evaluated hierarchically.
    llm_single_pass_max_tokens: int = 60_000
    llm_chunk_target_tokens: int = 6_000
    anthropic_api_key: SecretStr | None = None
    # Explicit so the SDK never silently inherits an unrelated ANTHROPIC_BASE_URL from the env.
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_refusal_fallback: bool = True
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    # Optional per-model price overrides "model=in_per_mtok:out_per_mtok,..." for cost tracking.
    llm_price_overrides: str = ""

    # --- Budget ------------------------------------------------------------------------------
    daily_llm_budget_usd: float = 5.0
    max_job_cost_usd: float = 0.50  # hard per-job cap checked before every LLM call
    stage_c_min_remaining_budget_fraction: float = 0.3
    # Spend cap for `avd llm-smoke` (explicitly authorized live check).
    smoke_test_max_cost_usd: float = 0.25

    # --- YouTube Data API --------------------------------------------------------------------
    youtube_api_key: SecretStr | None = None
    youtube_daily_quota_units: int = 10_000
    youtube_quota_reserve_units: int = 500

    # --- Embeddings --------------------------------------------------------------------------
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.HASHING
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Worker ------------------------------------------------------------------------------
    worker_lease_seconds: int = 600
    worker_batch_size: int = 8
    job_max_attempts: int = 4
    job_retry_base_seconds: float = 30.0

    # --- API ---------------------------------------------------------------------------------
    admin_token: SecretStr | None = None
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    cors_allow_origin_regex: str = r"chrome-extension://.*"

    fixtures_dir: Path = REPO_ROOT / "data" / "fixtures"
    media_cache_dir: Path = REPO_ROOT / "backend" / ".media_cache"
    # Directory of licensed reference documents enabling Stage-C retrieval fact checking.
    reference_corpus_dir: Path | None = None

    @property
    def demo_mode(self) -> bool:
        return self.llm_provider == LLMProviderName.HEURISTIC

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


@lru_cache
def get_settings() -> Settings:
    return Settings()
