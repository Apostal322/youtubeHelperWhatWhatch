"""Provider-agnostic structured-output LLM interface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    json_schema: dict[str, Any]
    schema_name: str
    model: str
    max_tokens: int = 16000
    effort: str | None = None
    # Distinguishes independent samples for self-consistency (part of the cache key).
    sample_index: int = 0
    metadata: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def cache_key(self, provider: str) -> str:
        payload = json.dumps(
            {
                "provider": provider,
                "model": self.model,
                "system": self.system,
                "user": self.user,
                "schema": self.json_schema,
                "effort": self.effort,
                "max_tokens": self.max_tokens,
                "sample": self.sample_index,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class LLMResponse:
    data: dict[str, Any]
    provider: str
    requested_model: str
    served_model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    cached: bool = False
    request_id: str | None = None


class LLMError(RuntimeError):
    retryable = False


class LLMTransientError(LLMError):
    retryable = True


class LLMRefusalError(LLMError):
    """The model declined. Recorded as a failed dimension set — never replaced with a guess."""


class LLMOutputError(LLMError):
    """Truncated / non-JSON / schema-violating output."""


class LLMConfigurationError(LLMError):
    pass


class LLMProvider(Protocol):
    name: str

    def complete_json(self, request: LLMRequest) -> LLMResponse: ...
