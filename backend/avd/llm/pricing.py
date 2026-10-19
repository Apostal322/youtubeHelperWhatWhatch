"""Token pricing for cost tracking and budget estimates (USD per million tokens).

Prices are first-party list prices (verified 2026-10-16 for Opus 5.5, Sonnet 5.5, Haiku 5.5 and
Fable 5.1). Cache-read discounts are not modelled, so costs err high. Prices are and can be overridden with
``AVD_LLM_PRICE_OVERRIDES="model=in:out,..."``. Unknown models are priced conservatively high so
budget enforcement errs on the safe side.
"""

from __future__ import annotations

from avd.config import get_settings

DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "heuristic-baseline": (0.0, 0.0),
}
UNKNOWN_MODEL_PRICE = (15.0, 75.0)
# Long-prompt tiers: (input-token threshold, (in, out) price above it). Verified 2026-10-16 against
# the Claude API reference (Haiku 5.5 prices prompts over 100K tokens at $0.50 / $2.50).
LONG_PROMPT_TIERS: dict[str, tuple[int, tuple[float, float]]] = {
    "claude-haiku-5-5": (100_000, (0.50, 2.50)),
}


def _overrides() -> dict[str, tuple[float, float]]:
    raw = get_settings().llm_price_overrides.strip()
    out: dict[str, tuple[float, float]] = {}
    if not raw:
        return out
    for item in raw.split(","):
        model, _, prices = item.partition("=")
        i, _, o = prices.partition(":")
        out[model.strip()] = (float(i), float(o))
    return out


# Models served on this machine (Ollama). Their marginal cost is $0: tokens are still recorded,
# and electricity/hardware are reported as durations, not dollars. Registered by the provider
# factory and the local runners; served models carry an ``ollama:`` prefix.
LOCAL_PREFIX = "ollama:"
_LOCAL_MODELS: set[str] = set()


def register_local_model(model: str) -> None:
    _LOCAL_MODELS.add(model)


def is_local_model(model: str) -> bool:
    return model.startswith(LOCAL_PREFIX) or model in _LOCAL_MODELS


def price_for(model: str) -> tuple[float, float]:
    if is_local_model(model):
        return (0.0, 0.0)
    return _overrides().get(model) or DEFAULT_PRICES.get(model) or UNKNOWN_MODEL_PRICE


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    tier = LONG_PROMPT_TIERS.get(model)
    if tier is not None and input_tokens > tier[0] and model not in _overrides():
        pin, pout = tier[1]
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)


def estimate_cost(model: str, prompt_chars: int, expected_output_tokens: int = 2500) -> float:
    """Rough pre-call estimate (~4 chars/token) used by the budget-aware scheduler."""
    return cost_usd(model, prompt_chars // 4 + 1, expected_output_tokens)
