"""Versioned description of how recommendations are selected.

``ranker_version`` alone does not say which videos a user could have been shown: retrieval
sizes, the relevance floor, the per-channel cap, diversification and the exploration quota all
change the result without touching the ranker's weights. Every impression therefore records a
*selection-policy version*, a digest of the complete static configuration, and every distinct
configuration is stored once in ``recommendation_policies`` so a version stays interpretable
after the code changes.

Per-request settings that come from the user (diversity, minimum quality, depth, the
small-creator boost, the requested size) are not part of the version; they are stored with each
impression as its ``selection_context``.

Bump ``ALGORITHM_REVISION`` whenever the selection *code* changes behaviour in a way the
configuration below does not capture. ``tests/test_selection_policy.py`` pins the current digest,
so a configuration change cannot ship without the test (and this note) being updated.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from avd.embeddings.base import Embedder
from avd.recsys.ranker import FEATURES, Ranker

ALGORITHM = "two_stage_mmr"
# 1: hard per-channel cap (Phase 2 to 3.2).
# 2: soft per-channel cap, relaxed only when every remaining candidate is from a capped channel
#    (Phase 3.3).
ALGORITHM_REVISION = 2

RETRIEVAL_PER_INTEREST = 60
QUALITY_CANDIDATES = 40
MAX_PER_CHANNEL = 2
CHANNEL_CAP = "soft"
EXPLORATION_SHARE = 5  # exploration quota = max(1, limit // EXPLORATION_SHARE)


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def policy_config(ranker: Ranker, embedder: Embedder) -> dict[str, Any]:
    """Everything static that decides which candidates can appear, and in what order."""
    weights = getattr(ranker, "weights", None)
    return {
        "algorithm": ALGORITHM,
        "algorithm_revision": ALGORITHM_REVISION,
        "retrieval": {
            "per_interest": RETRIEVAL_PER_INTEREST,
            "quality_candidates": QUALITY_CANDIDATES,
            "embedder": embedder.model_id,
        },
        "relevance_threshold": {
            "floor": embedder.relevance_floor,
            "ceiling": embedder.relevance_ceiling,
            "below_floor": "exploration_only",
        },
        "eligibility": {"excluded_states": ["failed", "insufficient_evidence"]},
        "ranker": {
            "version": ranker.version,
            "features": list(FEATURES),
            # A ranker that reuses a version name with other weights still changes the digest.
            "weights_digest": _digest(weights) if weights is not None else None,
        },
        "diversification": {"method": "mmr_embedding_cosine", "channel_cap": CHANNEL_CAP},
        "channel_cap": {"mode": CHANNEL_CAP, "max_per_channel": MAX_PER_CHANNEL},
        "exploration": {
            "strategy": "below_relevance_floor_quota",
            "quota": f"max(1, limit // {EXPLORATION_SHARE})",
            "applies": "only when the user has stated interests",
        },
    }


def policy_version(config: dict[str, Any]) -> str:
    return f"{config['algorithm']}_r{config['algorithm_revision']}:{_digest(config)}"
