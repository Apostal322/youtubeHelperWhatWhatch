"""Concrete LLM providers and wrappers (rate limiting, response caching)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
from sqlalchemy.orm import Session

from avd.config import LLMProviderName, Settings
from avd.db.models import LLMCache
from avd.llm.base import (
    LLMConfigurationError,
    LLMError,
    LLMOutputError,
    LLMProvider,
    LLMRefusalError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
)
from avd.llm.ratelimit import InMemoryTokenBucket, RateLimiter, RedisTokenBucket
from avd.logging import get_logger

log = get_logger(__name__)


class AnthropicProvider:
    """Claude via the official Anthropic SDK using JSON-schema structured outputs."""

    name = "anthropic"

    def __init__(self, api_key: str | None, refusal_fallback: bool = True) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMConfigurationError(
                "the 'anthropic' package is not installed; pip install 'ai-video-discovery[anthropic]'"
            ) from exc
        self._anthropic = anthropic
        # api_key=None lets the SDK resolve credentials from its standard environment chain.
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=2)
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
        try:
            if self._fallback:
                # Server-side refusal fallback: on a policy decline the API re-runs the request on
                # a fallback model. The served model is recorded from the response.
                resp = self._client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = self._client.messages.create(**kwargs)
        except (a.RateLimitError, a.InternalServerError, a.APIConnectionError) as exc:
            raise LLMTransientError(str(exc)) from exc
        except a.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code == 429:
                raise LLMTransientError(str(exc)) from exc
            raise LLMError(f"anthropic API error {exc.status_code}: {exc.message}") from exc

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
        )


class OpenAICompatibleProvider:
    """Any server implementing the OpenAI chat-completions API with ``json_schema`` responses
    (hosted APIs or self-hosted vLLM / llama.cpp / Ollama gateways)."""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=180.0, transport=transport
        )

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        body = {
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
        try:
            resp = self._http.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise LLMTransientError(str(exc)) from exc
        if resp.status_code in (429, 500, 502, 503, 504):
            raise LLMTransientError(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        choice = payload["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMOutputError("output truncated")
        msg = choice["message"]
        if msg.get("refusal"):
            raise LLMRefusalError(str(msg["refusal"]))
        try:
            data = json.loads(msg["content"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMOutputError(f"invalid JSON: {exc}") from exc
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
        )


class RateLimitedProvider:
    def __init__(self, inner: LLMProvider, limiter: RateLimiter) -> None:
        self._inner = inner
        self._limiter = limiter
        self.name = inner.name

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        if not self._limiter.acquire(f"{self.name}:{request.model}", timeout=120.0):
            raise LLMTransientError("rate limit wait exceeded")
        return self._inner.complete_json(request)


class CachedProvider:
    """Content-addressed response cache. Identical requests (same prompt, schema, model, sample
    index) are served from the database: re-running the pipeline is free and reproducible."""

    def __init__(self, inner: LLMProvider, session_factory: Callable[[], Session]) -> None:
        self._inner = inner
        self._session_factory = session_factory
        self.name = inner.name

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key(self.name)
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


def build_llm_provider(
    settings: Settings, session_factory: Callable[[], Session] | None = None
) -> LLMProvider | None:
    """Returns ``None`` in demo mode (heuristic baseline, no LLM)."""
    if settings.llm_provider == LLMProviderName.HEURISTIC:
        return None
    inner: LLMProvider
    if settings.llm_provider == LLMProviderName.ANTHROPIC:
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        inner = AnthropicProvider(key, refusal_fallback=settings.anthropic_refusal_fallback)
    elif settings.llm_provider == LLMProviderName.OPENAI_COMPATIBLE:
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        inner = OpenAICompatibleProvider(settings.openai_base_url, key)
    else:  # pragma: no cover
        raise LLMConfigurationError(f"unknown provider {settings.llm_provider}")
    limiter: RateLimiter = (
        RedisTokenBucket(settings.redis_url, settings.llm_requests_per_minute)
        if settings.redis_url
        else InMemoryTokenBucket(settings.llm_requests_per_minute)
    )
    provider: LLMProvider = RateLimitedProvider(inner, limiter)
    if session_factory is not None:
        provider = CachedProvider(provider, session_factory)
    return provider
