"""Concrete LLM providers and the wrapper chain used in production.

Chain (outermost first), built by ``build_llm_provider``::

    LoggingProvider      every attempt -> cost_ledger row (tokens, USD, latency, status, retries)
    CachedProvider       content-addressed response cache (re-runs are free and reproducible)
    SpendGuardProvider   refuses calls that would exceed the daily or per-job budget
    RateLimitedProvider  token bucket (in-process or Redis) + bounded concurrency
    RetryingProvider     bounded retries with exponential backoff + jitter for transient errors
    <raw provider>       AnthropicProvider | OpenAICompatibleProvider

Credentials are read from settings/environment only. No provider is constructed in demo mode.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
from sqlalchemy.orm import Session

from avd.config import LLMProviderName, Settings
from avd.db.models import CostLedger, LLMCache
from avd.llm.base import (
    LLMBudgetExceeded,
    LLMConfigurationError,
    LLMError,
    LLMOutputError,
    LLMProvider,
    LLMRefusalError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
)
from avd.llm.pricing import cost_usd, estimate_cost
from avd.llm.ratelimit import InMemoryTokenBucket, RateLimiter, RedisTokenBucket
from avd.logging import get_logger
from avd.observability import METRICS, add_job_spend, correlation_id, job_id, job_spend

log = get_logger(__name__)


# ----------------------------------------------------------------------------------- raw providers


class AnthropicProvider:
    """Claude via the official Anthropic SDK using JSON-schema structured outputs."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        *,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 180.0,
        refusal_fallback: bool = True,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMConfigurationError(
                "the 'anthropic' package is not installed; pip install 'ai-video-discovery[anthropic]'"
            ) from exc
        self._anthropic = anthropic
        try:
            # Retries are handled by RetryingProvider so every attempt is accounted for.
            self._client = anthropic.Anthropic(
                api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
            )
        except anthropic.AnthropicError as exc:
            raise LLMConfigurationError(f"cannot construct Anthropic client: {exc}") from exc
        self._fallback = refusal_fallback

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        a = self._anthropic
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": request.json_schema}}
        if request.effort:
            output_config["effort"] = request.effort
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            "output_config": output_config,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        started = time.monotonic()
        try:
            if self._fallback:
                # Server-side refusal fallback: on a policy decline the API re-runs the request on
                # a fallback model. The served model is recorded from the response.
                resp = self._client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = self._client.messages.create(**kwargs)
        except (a.AuthenticationError, a.PermissionDeniedError) as exc:
            raise LLMConfigurationError(f"anthropic credentials rejected: {exc}") from exc
        except a.NotFoundError as exc:
            raise LLMConfigurationError(f"unknown model or endpoint: {exc}") from exc
        except (a.RateLimitError, a.InternalServerError, a.APIConnectionError) as exc:
            raise LLMTransientError(str(exc)) from exc
        except a.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in (408, 409, 429):
                raise LLMTransientError(str(exc)) from exc
            raise LLMError(f"anthropic API error {exc.status_code}: {exc.message}") from exc
        except a.AnthropicError as exc:  # e.g. no resolvable credentials
            raise LLMConfigurationError(str(exc)) from exc

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefusalError(f"model declined: {getattr(details, 'category', None)}")
        if resp.stop_reason == "max_tokens":
            raise LLMOutputError("output truncated at max_tokens")
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LLMOutputError("no text block in response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"invalid JSON: {exc}") from exc
        return LLMResponse(
            data=data,
            provider=self.name,
            requested_model=request.model,
            served_model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
            request_id=resp.id,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


class OpenAICompatibleProvider:
    """OpenAI's API or any server implementing chat completions with ``json_schema`` responses
    (vLLM, llama.cpp server, Ollama's OpenAI endpoint, hosted gateways)."""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 180.0,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": request.schema_name, "schema": request.json_schema, "strict": True},
            },
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        started = time.monotonic()
        try:
            resp = self._http.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise LLMTransientError(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code in (408, 409, 429, 500, 502, 503, 504):
            raise LLMTransientError(f"HTTP {resp.status_code}")
        if resp.status_code in (401, 403, 404):
            raise LLMConfigurationError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            payload = resp.json()
            choice = payload["choices"][0]
            msg = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMOutputError(f"malformed completion envelope: {exc}") from exc
        if choice.get("finish_reason") == "length":
            raise LLMOutputError("output truncated")
        if msg.get("refusal"):
            raise LLMRefusalError(str(msg["refusal"]))
        try:
            data = json.loads(msg["content"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMOutputError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMOutputError("JSON output is not an object")
        usage = payload.get("usage", {})
        return LLMResponse(
            data=data,
            provider=self.name,
            requested_model=request.model,
            served_model=payload.get("model", request.model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            stop_reason=choice.get("finish_reason"),
            request_id=payload.get("id"),
            latency_ms=int((time.monotonic() - started) * 1000),
        )


# ------------------------------------------------------------------------------------------ wrappers


class RetryingProvider:
    """Bounded retries for transient errors only (never for refusals, invalid output, auth or
    budget errors). Exponential backoff with full jitter."""

    def __init__(
        self,
        inner: LLMProvider,
        max_retries: int = 3,
        base_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self.name = inner.name
        self.max_retries = max(0, max_retries)
        self.base_seconds = base_seconds
        self._sleep = sleep

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._inner.complete_json(request)
                resp.attempts = attempt
                return resp
            except LLMTransientError as exc:
                METRICS.inc("avd_llm_retries_total", provider=self.name)
                if attempt > self.max_retries:
                    exc.attempts = attempt  # type: ignore[attr-defined]
                    raise
                delay = random.uniform(0, self.base_seconds * (2 ** (attempt - 1)))
                log.warning(
                    "transient LLM error, retrying",
                    extra={"attempt": attempt, "delay_s": round(delay, 2), "error": str(exc)},
                )
                self._sleep(delay)


class RateLimitedProvider:
    def __init__(self, inner: LLMProvider, limiter: RateLimiter, max_concurrency: int = 4) -> None:
        self._inner = inner
        self._limiter = limiter
        self._sem = threading.BoundedSemaphore(max(1, max_concurrency))
        self.name = inner.name

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if not self._limiter.acquire(f"{self.name}:{request.model}", timeout=120.0):
            raise LLMTransientError("rate limit wait exceeded")
        with self._sem:
            return self._inner.complete_json(request)


class SpendGuardProvider:
    """Refuses a call when its *estimated* cost would exceed today's remaining LLM budget or the
    per-job cap. Actual costs are added to the job's running total after each call."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        remaining_budget: Callable[[], float],
        max_job_cost_usd: float,
    ) -> None:
        self._inner = inner
        self.name = inner.name
        self._remaining = remaining_budget
        self.max_job_cost_usd = max_job_cost_usd

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        est = estimate_cost(
            request.model, len(request.system) + len(request.user), min(request.max_tokens, 4000)
        )
        if job_spend() + est > self.max_job_cost_usd:
            raise LLMBudgetExceeded(
                f"per-job cap: spent ${job_spend():.4f} + estimated ${est:.4f} > ${self.max_job_cost_usd:.2f}",
                scope="job",
            )
        remaining = self._remaining()
        if est > remaining:
            raise LLMBudgetExceeded(
                f"daily budget: estimated ${est:.4f} > remaining ${remaining:.4f}", scope="daily"
            )
        resp = self._inner.complete_json(request)
        add_job_spend(cost_usd(resp.served_model, resp.input_tokens, resp.output_tokens))
        return resp


def model_revision(provider: Any, model: str) -> str | None:
    """The innermost provider's runtime revision for ``model`` (local models only; hosted APIs
    return ``None``). Raises an ``LLMError`` when a local server cannot be asked."""
    p = provider
    while p is not None:
        rev = getattr(p, "revision", None)
        if callable(rev):
            return str(rev(model))
        p = getattr(p, "_inner", None)
    return None


class CachedProvider:
    """Content-addressed response cache. Identical requests (same prompt, schema, model, sampling
    settings, sample index) are served from the database: re-running the pipeline is free and
    reproducible. Only syntactically valid JSON responses reach the cache."""

    def __init__(self, inner: LLMProvider, session_factory: Callable[[], Session]) -> None:
        self._inner = inner
        self._session_factory = session_factory
        self.name = inner.name

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        try:
            rev = model_revision(self._inner, request.model)
        except LLMError:
            # A local model whose revision cannot be read: never serve or store a cached answer
            # that might belong to other weights. The call itself will report the problem.
            return self._inner.complete_json(request)
        key = request.cache_key(self.name if rev is None else f"{self.name}|{rev}")
        with self._session_factory() as s:
            hit = s.get(LLMCache, key)
            if hit is not None:
                r = hit.response
                return LLMResponse(
                    data=r["data"],
                    provider=self.name,
                    requested_model=request.model,
                    served_model=r["served_model"],
                    input_tokens=r["input_tokens"],
                    output_tokens=r["output_tokens"],
                    stop_reason=r.get("stop_reason"),
                    cached=True,
                    request_id=r.get("request_id"),
                    attempts=0,
                    latency_ms=0,
                )
        resp = self._inner.complete_json(request)
        with self._session_factory() as s:
            if s.get(LLMCache, key) is None:
                s.add(
                    LLMCache(
                        key=key,
                        provider=self.name,
                        model=resp.served_model,
                        response={
                            "data": resp.data,
                            "served_model": resp.served_model,
                            "input_tokens": resp.input_tokens,
                            "output_tokens": resp.output_tokens,
                            "stop_reason": resp.stop_reason,
                            "request_id": resp.request_id,
                        },
                    )
                )
                s.commit()
        return resp


class LoggingProvider:
    """Records every model call (success, cache hit or failure) in ``cost_ledger`` so retries,
    refusals and failed attempts are visible in cost and reliability reports."""

    def __init__(
        self, inner: LLMProvider, session_factory: Callable[[], Session], model_revision: str | None = None
    ) -> None:
        self._inner = inner
        self._session_factory = session_factory
        self.name = inner.name
        self.model_revision = model_revision

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        started = time.monotonic()
        try:
            resp = self._inner.complete_json(request)
        except LLMError as exc:
            self._record(
                request,
                status="error",
                error_type=type(exc).__name__,
                latency_ms=int((time.monotonic() - started) * 1000),
                attempts=getattr(exc, "attempts", 1),
            )
            METRICS.inc("avd_llm_calls_total", provider=self.name, status="error", error=type(exc).__name__)
            raise
        status = "cached" if resp.cached else "ok"
        latency = resp.latency_ms if resp.latency_ms is not None else int((time.monotonic() - started) * 1000)
        self._record(request, status=status, latency_ms=latency, attempts=resp.attempts, resp=resp)
        METRICS.inc("avd_llm_calls_total", provider=self.name, status=status, error="")
        if not resp.cached:
            METRICS.observe("avd_llm_latency_seconds", latency / 1000.0, provider=self.name)
            METRICS.inc("avd_llm_tokens_total", resp.input_tokens, direction="input")
            METRICS.inc("avd_llm_tokens_total", resp.output_tokens, direction="output")
        return resp

    def _record(
        self,
        request: LLMRequest,
        *,
        status: str,
        latency_ms: int,
        attempts: int,
        resp: LLMResponse | None = None,
        error_type: str | None = None,
    ) -> None:
        try:
            with self._session_factory() as s:
                s.add(
                    CostLedger(
                        job_id=job_id.get(),
                        video_id=request.metadata.get("video_id"),
                        provider=self.name,
                        model=resp.served_model if resp else request.model,
                        model_revision=self.model_revision,
                        operation=request.schema_name,
                        prompt_version=request.metadata.get("prompt_version"),
                        input_tokens=resp.input_tokens if resp else 0,
                        output_tokens=resp.output_tokens if resp else 0,
                        cost_usd=0.0
                        if resp is None or resp.cached
                        else cost_usd(resp.served_model, resp.input_tokens, resp.output_tokens),
                        cached=bool(resp and resp.cached),
                        status=status,
                        error_type=error_type,
                        latency_ms=latency_ms,
                        attempts=attempts,
                        request_id=resp.request_id if resp else None,
                        correlation_id=correlation_id.get(),
                    )
                )
                s.commit()
        except Exception:
            log.exception("failed to record model run")


class RunBudget:
    """A hard spend ceiling for one experiment run, safe under concurrency and across resumes.

    Each call *reserves* its worst-case cost before it is made (prompt at ~3 characters per token,
    the full ``max_tokens`` as output, billed at the model's price) and settles to the actual cost
    afterwards. A call whose worst case would cross the ceiling is refused, so the ceiling cannot
    be exceeded by in-flight calls or retries."""

    CHARS_PER_TOKEN_WORST = 3.0

    def __init__(self, ceiling_usd: float, already_spent_usd: float = 0.0) -> None:
        self.ceiling_usd = ceiling_usd
        self.spent_usd = already_spent_usd
        self.reserved_usd = 0.0
        self.refused = 0
        self._lock = threading.Lock()

    def worst_case(self, request: LLMRequest) -> float:
        tokens = int((len(request.system) + len(request.user)) / self.CHARS_PER_TOKEN_WORST) + 1
        return cost_usd(request.model, tokens, request.max_tokens)

    def reserve(self, amount: float) -> bool:
        with self._lock:
            if self.spent_usd + self.reserved_usd + amount > self.ceiling_usd:
                self.refused += 1
                return False
            self.reserved_usd += amount
            return True

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved_usd -= reserved
            self.spent_usd += actual


class RunBudgetGuard:
    def __init__(self, inner: LLMProvider, budget: RunBudget) -> None:
        self._inner = inner
        self.name = inner.name
        self.budget = budget

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        worst = self.budget.worst_case(request)
        if not self.budget.reserve(worst):
            raise LLMBudgetExceeded(
                f"run ceiling: spent ${self.budget.spent_usd:.4f}, worst case of this call "
                f"${worst:.4f}, ceiling ${self.budget.ceiling_usd:.2f}",
                scope="job",
            )
        try:
            resp = self._inner.complete_json(request)
        except BaseException:
            self.budget.settle(worst, 0.0)  # a failed call is not billed by the providers we support
            raise
        self.budget.settle(worst, cost_usd(resp.served_model, resp.input_tokens, resp.output_tokens))
        return resp


def build_llm_provider(
    settings: Settings,
    session_factory: Callable[[], Session] | None = None,
    raw: LLMProvider | None = None,
    run_budget: RunBudget | None = None,
) -> LLMProvider | None:
    """Returns ``None`` in demo mode (heuristic baseline, no LLM). ``raw`` injects a provider for
    tests; the production wrapper chain is still applied."""
    if settings.llm_provider == LLMProviderName.HEURISTIC and raw is None:
        return None
    inner: LLMProvider
    if raw is not None:
        inner = raw
    elif settings.llm_provider == LLMProviderName.ANTHROPIC:
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        inner = AnthropicProvider(
            key,
            base_url=settings.anthropic_base_url,
            timeout=settings.llm_timeout_seconds,
            refusal_fallback=settings.anthropic_refusal_fallback,
        )
    elif settings.llm_provider == LLMProviderName.OPENAI_COMPATIBLE:
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        inner = OpenAICompatibleProvider(settings.openai_base_url, key, timeout=settings.llm_timeout_seconds)
    elif settings.llm_provider == LLMProviderName.OLLAMA:
        from avd.llm.ollama import OllamaProvider
        from avd.llm.pricing import register_local_model

        for m in (settings.llm_model_stage_b, settings.llm_model_stage_c):
            register_local_model(m)
        inner = OllamaProvider(
            settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
            keep_alive=settings.ollama_keep_alive,
            think=settings.ollama_think,
            allow_remote=settings.ollama_allow_remote,
            timeout=settings.llm_timeout_seconds,
            seed=settings.ollama_seed,
        )
    else:  # pragma: no cover
        raise LLMConfigurationError(f"unknown provider {settings.llm_provider}")
    if run_budget is not None:  # innermost: every attempt, including retries, is checked
        inner = RunBudgetGuard(inner, run_budget)
    provider: LLMProvider = RetryingProvider(inner, settings.llm_max_retries, settings.llm_retry_base_seconds)
    limiter: RateLimiter = (
        RedisTokenBucket(settings.redis_url, settings.llm_requests_per_minute)
        if settings.redis_url
        else InMemoryTokenBucket(settings.llm_requests_per_minute)
    )
    provider = RateLimitedProvider(provider, limiter, settings.llm_max_concurrency)
    if session_factory is not None:
        from avd.workers.budget import BudgetManager

        budget = BudgetManager(settings)

        def remaining() -> float:
            with session_factory() as s:
                return budget.remaining(s)

        provider = SpendGuardProvider(
            provider, remaining_budget=remaining, max_job_cost_usd=settings.max_job_cost_usd
        )
        provider = CachedProvider(provider, session_factory)
        provider = LoggingProvider(provider, session_factory, settings.llm_model_revision)
    return provider
