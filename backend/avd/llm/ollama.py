"""Ollama as a first-class, *local-only* inference backend.

The native API is used rather than Ollama's OpenAI-compatible layer, because only it exposes
what this pipeline needs: JSON-schema constrained output (``format``), the context window
(``options.num_ctx``), thinking control (``think``), model residency (``keep_alive``) and exact
token counts and durations.

Safety properties (each covered by tests):

* **Local only.** The base URL must be loopback or a private address unless
  ``AVD_OLLAMA_ALLOW_REMOTE=true``. Ollama "cloud" models (names ending in ``-cloud``/``:cloud``,
  or ``/api/show`` reporting a ``remote_host``) are refused: they would send transcripts to a
  third party.
* **No silent truncation.** Ollama truncates prompts that exceed ``num_ctx`` without an error.
  The provider refuses a request whose conservative token estimate cannot fit
  (``LLMContextOverflow``), and treats a response whose reported prompt + output tokens reach
  ``num_ctx`` as possibly truncated (an output error, never a result).
* **Failure semantics.** Server unreachable → transient (jobs are deferred and retried, the API
  stays up); unknown model → configuration error naming ``ollama pull <model>``; output cut off
  (``done_reason == "length"``) or invalid JSON → output error.
* **Cost.** Local inference is recorded at $0; tokens and durations are still recorded.
"""

from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from avd.llm.base import (
    LLMConfigurationError,
    LLMError,
    LLMOutputError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
)

# Conservative prompt-size estimate for the pre-flight context check (no tokenizer available
# for every local model). Real counts come back as prompt_eval_count.
CHARS_PER_TOKEN_CONSERVATIVE = 3.0


class LLMContextOverflow(LLMOutputError):
    """The prompt plus requested output cannot fit the model's context window."""


@dataclass(frozen=True)
class OllamaModelInfo:
    name: str
    digest: str | None
    parameter_size: str | None
    quantization: str | None
    family: str | None
    context_length: int | None
    capabilities: tuple[str, ...]
    remote: bool

    @property
    def fingerprint(self) -> str:
        return f"{self.name}@{(self.digest or 'unknown')[:12]}:{self.quantization or '?'}"


def is_cloud_model(name: str) -> bool:
    return name.endswith("-cloud") or name.endswith(":cloud")


def check_local_url(base_url: str, allow_remote: bool) -> None:
    host = urlparse(base_url).hostname or ""
    if allow_remote:
        return
    if host in ("localhost", "host.docker.internal") or host.endswith(".svc.cluster.local"):
        return
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (ip.is_loopback or ip.is_private):
        return
    raise LLMConfigurationError(
        f"Ollama endpoint {base_url} is not local; set AVD_OLLAMA_ALLOW_REMOTE=true only for a host you control"
    )


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        num_ctx: int = 16384,
        keep_alive: str = "10m",
        think: bool = False,
        allow_remote: bool = False,
        timeout: float = 600.0,
        seed: int | None = 0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        base_url = base_url.rstrip("/").removesuffix("/v1")
        check_local_url(base_url, allow_remote)
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self.seed = seed
        # No proxy for a local server: a corporate/egress proxy must never see transcripts.
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport, trust_env=False)
        self._info: dict[str, OllamaModelInfo] = {}

    # ------------------------------------------------------------------------------ inspection

    def _get(self, path: str) -> Any:
        try:
            r = self._http.get(path)
        except httpx.HTTPError as exc:
            raise LLMTransientError(f"Ollama unreachable at {self.base_url}: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise LLMError(f"Ollama {path}: HTTP {r.status_code}")
        return r.json()

    def version(self) -> str:
        return str(self._get("/api/version").get("version", "unknown"))

    def installed(self) -> list[dict[str, Any]]:
        return list(self._get("/api/tags").get("models", []))

    def running(self) -> list[dict[str, Any]]:
        return list(self._get("/api/ps").get("models", []))

    def pull(self, model: str) -> dict[str, Any]:
        """Download a model into the local server. Callers must have size approval (avd local pull)."""
        if is_cloud_model(model):
            raise LLMConfigurationError(f"{model} is an Ollama cloud model; only local models are allowed")
        try:
            r = self._http.post("/api/pull", json={"model": model, "stream": False}, timeout=None)
        except httpx.HTTPError as exc:
            raise LLMTransientError(f"Ollama unreachable at {self.base_url}: {type(exc).__name__}") from exc
        if r.status_code != 200:
            raise LLMError(f"pull {model}: HTTP {r.status_code}: {r.text[:300]}")
        self._info.pop(model, None)
        return dict(r.json())

    def model_info(self, model: str) -> OllamaModelInfo:
        if model in self._info:
            return self._info[model]
        if is_cloud_model(model):
            raise LLMConfigurationError(f"{model} is an Ollama cloud model; only local models are allowed")
        try:
            r = self._http.post("/api/show", json={"model": model})
        except httpx.HTTPError as exc:
            raise LLMTransientError(f"Ollama unreachable at {self.base_url}: {type(exc).__name__}") from exc
        if r.status_code == 404:
            raise LLMConfigurationError(f"model {model!r} is not installed: run `ollama pull {model}`")
        if r.status_code != 200:
            raise LLMError(f"Ollama /api/show: HTTP {r.status_code}")
        data = r.json()
        if data.get("remote_host") or data.get("remote_model"):
            raise LLMConfigurationError(f"{model} is served remotely ({data.get('remote_host')}); refused")
        details = data.get("details") or {}
        ctx = next(
            (int(v) for k, v in (data.get("model_info") or {}).items() if k.endswith(".context_length")), None
        )
        digest = next(
            (m.get("digest") for m in self.installed() if m.get("name") == model or m.get("model") == model),
            None,
        )
        info = OllamaModelInfo(
            name=model,
            digest=digest,
            parameter_size=details.get("parameter_size"),
            quantization=details.get("quantization_level"),
            family=details.get("family"),
            context_length=ctx,
            capabilities=tuple(data.get("capabilities") or ()),
            remote=False,
        )
        self._info[model] = info
        return info

    def effective_context(self, model: str) -> int:
        native = self.model_info(model).context_length
        return min(self.num_ctx, native) if native else self.num_ctx

    # ------------------------------------------------------------------------------ inference

    def complete_json(self, request: LLMRequest) -> LLMResponse:
        info = self.model_info(request.model)
        ctx = min(self.num_ctx, info.context_length) if info.context_length else self.num_ctx
        prompt_tokens_est = int((len(request.system) + len(request.user)) / CHARS_PER_TOKEN_CONSERVATIVE) + 64
        num_predict = min(request.max_tokens, max(256, ctx - prompt_tokens_est))
        if prompt_tokens_est + 256 > ctx:
            raise LLMContextOverflow(
                f"prompt ~{prompt_tokens_est} tokens does not fit num_ctx={ctx} for {request.model}; "
                "the pipeline must chunk (never truncate)"
            )
        options: dict[str, Any] = {"num_ctx": ctx, "num_predict": num_predict}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if self.seed is not None:
            options["seed"] = self.seed + request.sample_index
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "format": request.json_schema,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if "thinking" in info.capabilities:
            body["think"] = self.think
        started = time.monotonic()
        try:
            r = self._http.post("/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"Ollama timed out after {self._http.timeout.read}s") from exc
        except httpx.HTTPError as exc:
            raise LLMTransientError(f"Ollama unreachable at {self.base_url}: {type(exc).__name__}") from exc
        if r.status_code == 404:
            self._info.pop(request.model, None)
            raise LLMConfigurationError(
                f"model {request.model!r} is not installed: run `ollama pull {request.model}`"
            )
        if r.status_code in (429, 500, 502, 503):
            raise LLMTransientError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        try:
            payload = r.json()
            content = payload["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMOutputError(f"malformed Ollama response: {exc}") from exc
        prompt_n = int(payload.get("prompt_eval_count") or 0)
        out_n = int(payload.get("eval_count") or 0)
        if payload.get("done_reason") == "length":
            raise LLMOutputError(f"output cut off at num_predict={num_predict}")
        if prompt_n and prompt_n + out_n >= ctx:
            raise LLMContextOverflow(
                f"prompt {prompt_n} + output {out_n} tokens reached num_ctx={ctx}: possibly truncated"
            )
        if not content.strip():
            raise LLMOutputError("empty content (thinking-only answer?)")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMOutputError("JSON output is not an object")
        latency = int((time.monotonic() - started) * 1000)
        eval_s = (payload.get("eval_duration") or 0) / 1e9
        prompt_s = (payload.get("prompt_eval_duration") or 0) / 1e9
        return LLMResponse(
            data=data,
            provider=self.name,
            requested_model=request.model,
            served_model=f"ollama:{info.fingerprint}",
            input_tokens=prompt_n,
            output_tokens=out_n,
            stop_reason=payload.get("done_reason"),
            request_id=None,
            latency_ms=latency,
            runtime={
                "load_s": round((payload.get("load_duration") or 0) / 1e9, 3),
                "prompt_tokens_per_s": round(prompt_n / prompt_s, 2) if prompt_s else None,
                "output_tokens_per_s": round(out_n / eval_s, 2) if eval_s else None,
                "num_ctx": ctx,
            },
        )
