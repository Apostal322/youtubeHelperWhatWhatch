"""Token pricing for cost tracking and budget estimates (USD per million tokens).

Prices are first-party list prices at the time of writing and can be overridden with
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


def price_for(model: str) -> tuple[float, float]:
    return _overrides().get(model) or DEFAULT_PRICES.get(model) or UNKNOWN_MODEL_PRICE


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)


def estimate_cost(model: str, prompt_chars: int, expected_output_tokens: int = 2500) -> float:
    """Rough pre-call estimate (~4 chars/token) used by the budget-aware scheduler."""
    return cost_usd(model, prompt_chars // 4 + 1, expected_output_tokens)
