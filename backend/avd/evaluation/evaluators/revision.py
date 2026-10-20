"""Local model revisions in evaluator descriptors (and so in the evaluation config hash)."""

from __future__ import annotations

from typing import Any

from avd.llm.base import LLMError


def with_model_revision(descriptor: dict[str, Any], provider: Any, model: str) -> dict[str, Any]:
    """Adds ``model_revision`` for local models (weights digest, quantization, context, seed), so
    new weights under the same tag make earlier evaluations stale. Hosted models have no
    revision and their descriptors, hence config hashes, are unchanged. When a local server is
    down the revision is ``"unresolved"``; evaluations are only persisted after successful calls,
    so a stored config hash always carries the real revision."""
    from avd.llm.providers import model_revision

    try:
        rev = model_revision(provider, model)
    except LLMError:
        rev = "unresolved"
    if rev is not None:
        descriptor["model_revision"] = rev
    return descriptor
