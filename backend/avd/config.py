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

from pydantic import Field, SecretStr, model_validator
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
    # Local open-weight models served by Ollama on this machine (native API, $0, no cloud).
    OLLAMA = "ollama"


class TaskBackend(StrEnum):
    INLINE = "inline"  # DB-backed queue drained by `avd worker` (no Redis needed)
    CELERY = "celery"  # Celery workers + beat dispatcher, Redis broker


class DeploymentMode(StrEnum):
    # One person's computer: loopback clients only, Host/Origin checks against DNS rebinding and
    # cross-site requests, embedded worker, no account needed.
    LOCAL = "local"
    # A shared or public server: explicit allowed hosts, an admin token and controlled user
    # registration are required before the API will start.
    SERVER = "server"


class Registration(StrEnum):
    OPEN = "open"  # anyone who can reach the API gets an anonymous account
    INVITE = "invite"  # account creation needs an invite code (or the admin token)
    CLOSED = "closed"  # only the admin token can create accounts


class EmbeddingProviderName(StrEnum):
    HASHING = "hashing"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    ONNX_MINILM = "onnx_minilm"  # all-MiniLM-L6-v2 on ONNX Runtime; 'avd embeddings fetch-model'


# Tokens (at 3 chars/token) reserved in a local context window for everything except the
# transcript: system prompt, metadata, genre guidance and rubric text (measured at ~7.4k chars
# plus up to 2k chars of description: ~3.1k tokens, plus chat-template tokens).
LOCAL_PROMPT_OVERHEAD_TOKENS = 3_500


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
    # Registered prompt key (avd.evaluation.prompts); variants exist for the sensitivity experiment.
    llm_prompt: str = "rubric_prompt_v2"
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
    # --- Local inference (Ollama) -----------------------------------------------------------
    # Native API root (a trailing /v1 is accepted and stripped). Must be loopback/private unless
    # ollama_allow_remote: transcripts never leave hosts you control.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"  # pick with `avd local doctor`; tag is recorded with its digest
    ollama_num_ctx: int = 16_384  # context window requested per call (KV cache grows with it)
    ollama_keep_alive: str = "10m"  # how long the model stays resident after the last request
    ollama_think: bool = False  # reasoning traces off by default: slower, and not needed for JSON
    ollama_allow_remote: bool = False
    ollama_seed: int = 0  # sampling seed (+ sample index); benchmarks vary it across repeats
    # One model instance serves requests serially unless OLLAMA_NUM_PARALLEL is raised; more
    # in-flight requests only queue (and multiply KV-cache memory when parallel slots exist).
    ollama_max_concurrency: int = 1
    # Output tokens reserved inside num_ctx for each answer; prompts are sized to fit the rest.
    # Default: a quarter of the window, between 1536 and 4096 (a 10-dimension rubric answer is
    # ~1.5-2k tokens; a longer one fails as "cut off", never as a result).
    ollama_output_reserve_tokens: int | None = None
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
    embedding_model_dir: Path = REPO_ROOT / "backend" / ".model_cache"

    # --- Worker ------------------------------------------------------------------------------
    worker_lease_seconds: int = 600
    worker_batch_size: int = 8
    job_max_attempts: int = 4
    job_retry_base_seconds: float = 30.0

    # Experimental evaluators ("package.module:Class"), lowest precedence, recorded in every
    # evaluation's versions and config hash (avd.plugins). Empty = the official protocol only.
    experimental_evaluators: list[str] = Field(default_factory=list)

    # --- Deployment ---------------------------------------------------------------------------
    deployment_mode: DeploymentMode = DeploymentMode.LOCAL
    # Host headers the API answers (DNS-rebinding protection). Local default: loopback names only.
    # Server mode must set them explicitly (e.g. ["avd.example.org"]).
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "[::1]", "::1"])
    # Local mode: extra peer networks treated as "this computer", e.g. the Docker bridge when the
    # API runs in a container whose port is published on 127.0.0.1 only. Never a LAN or the internet.
    local_peer_networks: list[str] = Field(default_factory=list)
    # Who may create accounts. Default: open locally (loopback only), invite on a server.
    user_registration: Registration | None = None
    invite_codes: SecretStr | None = None  # comma-separated invite codes for registration=invite
    # Server mode only: require a user token for catalog/evaluation reads as well (private instance).
    require_auth_for_read: bool = False
    # Run queued jobs in a background thread inside the API process (`avd start`); no Redis/Celery.
    embedded_worker: bool = False
    embedded_worker_poll_seconds: float = 3.0
    # New submissions are refused (HTTP 503) while this many jobs are queued: bounded backlog.
    max_queued_jobs: int = 500
    # Outbound network calls the operator has not configured are never made; this only governs the
    # optional, explicit update check, which is off.
    telemetry: bool = False

    # --- API ---------------------------------------------------------------------------------
    admin_token: SecretStr | None = None
    # API protection: per-caller token bucket (0 disables); /metrics needs the admin token unless public.
    max_media_upload_mb: int = 500
    api_rate_limit_per_minute: int = 300
    api_rate_limit_burst: int = 60
    # Every request also draws from a per-client-IP bucket, so rotating (even invalid) bearer
    # tokens cannot multiply a caller's allowance.
    api_rate_limit_per_ip_per_minute: int = 600
    api_rate_limit_ip_burst: int = 120
    metrics_public: bool = False
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    # The extension does not need CORS: its pages fetch with host permissions the user granted.
    # Set a regex (e.g. "chrome-extension://<your-extension-id>") only for other browser clients.
    cors_allow_origin_regex: str | None = None

    fixtures_dir: Path = REPO_ROOT / "data" / "fixtures"
    media_cache_dir: Path = REPO_ROOT / "backend" / ".media_cache"
    # Pinned-commit cache of licensed dataset source files (not committed; re-fetchable).
    dataset_cache_dir: Path = REPO_ROOT / "backend" / ".dataset_cache"
    # Directory of licensed reference documents enabling Stage-C retrieval fact checking.
    reference_corpus_dir: Path | None = None

    @property
    def registration(self) -> Registration:
        if self.user_registration is not None:
            return self.user_registration
        return Registration.OPEN if self.deployment_mode == DeploymentMode.LOCAL else Registration.INVITE

    def server_mode_problems(self) -> list[str]:
        """Why this configuration must not serve a shared instance (empty = fine)."""
        if self.deployment_mode != DeploymentMode.SERVER:
            return []
        out = []
        if self.admin_token is None or len(self.admin_token.get_secret_value()) < 24:
            out.append("AVD_ADMIN_TOKEN must be set (at least 24 characters)")
        elif self.admin_token.get_secret_value().startswith("change-me"):
            out.append("AVD_ADMIN_TOKEN is still the placeholder value")
        if not self.allowed_hosts or set(self.allowed_hosts) <= {"localhost", "127.0.0.1", "[::1]", "::1"}:
            out.append('AVD_ALLOWED_HOSTS must list the public host names (e.g. ["avd.example.org"])')
        if self.registration == Registration.INVITE and self.invite_codes is None:
            out.append("AVD_USER_REGISTRATION=invite needs AVD_INVITE_CODES (or choose open/closed)")
        if self.metrics_public:
            out.append("AVD_METRICS_PUBLIC must be false on a shared server")
        if self.is_postgres and ":avd@" in self.database_url:
            out.append("the database password is the default 'avd': set POSTGRES_PASSWORD")
        return out

    @model_validator(mode="after")
    def _empty_secrets_are_unset(self) -> Settings:
        # An empty env var (e.g. AVD_ADMIN_TOKEN= in Compose) must disable a secret, never make
        # the empty string a valid credential.
        for f in ("admin_token", "invite_codes", "anthropic_api_key", "openai_api_key", "youtube_api_key"):
            v = getattr(self, f)
            if v is not None and not v.get_secret_value().strip():
                setattr(self, f, None)
        return self

    @model_validator(mode="after")
    def _local_peers_are_private(self) -> Settings:
        import ipaddress

        for n in self.local_peer_networks:
            net = ipaddress.ip_network(n, strict=False)
            if not (net.is_private or net.is_loopback) or net.prefixlen < 8:
                raise ValueError(f"local_peer_networks may only contain private networks (got {n})")
        return self

    @model_validator(mode="after")
    def _local_model_defaults(self) -> Settings:
        """With ``llm_provider=ollama``: use ``ollama_model`` for any stage model left at its cloud
        default, and size prompts and outputs to fit the local context window, so long
        transcripts are chunked rather than truncated. Explicit settings are kept, but are
        clamped to what the window can hold."""
        if self.llm_provider != LLMProviderName.OLLAMA:
            return self
        explicit = self.model_fields_set
        for f in ("llm_model_stage_b", "llm_model_stage_c"):
            if f not in explicit:
                setattr(self, f, self.ollama_model)
        if "llm_max_concurrency" not in explicit:
            self.llm_max_concurrency = self.ollama_max_concurrency
        if self.ollama_output_reserve_tokens is None:
            self.ollama_output_reserve_tokens = min(4096, max(1536, self.ollama_num_ctx // 4))
        self.llm_max_output_tokens = min(self.llm_max_output_tokens, self.ollama_output_reserve_tokens)
        # Pipeline estimates use 4 chars/token, the provider's pre-flight check 3: convert, and keep
        # room for the system prompt, metadata and rubric.
        room = self.ollama_num_ctx - self.ollama_output_reserve_tokens - LOCAL_PROMPT_OVERHEAD_TOKENS
        if room < 1024:
            raise ValueError(
                f"ollama_num_ctx={self.ollama_num_ctx} leaves {room} tokens for transcript text; "
                "raise the context window or lower ollama_output_reserve_tokens"
            )
        fit = room * 3 // 4
        self.llm_single_pass_max_tokens = min(self.llm_single_pass_max_tokens, fit)
        self.llm_chunk_target_tokens = min(self.llm_chunk_target_tokens, fit)
        return self

    @property
    def demo_mode(self) -> bool:
        return self.llm_provider == LLMProviderName.HEURISTIC

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


@lru_cache
def get_settings() -> Settings:
    return Settings()
