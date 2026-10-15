"""Text embeddings.

* ``HashingEmbedder`` (default): deterministic signed feature hashing of unigrams + bigrams with
  sublinear TF, L2-normalised, 384 dims. No downloads, no GPU, reproducible — but purely lexical:
  it does not know that "car" and "automobile" are related, and does not align across languages.
* ``OnnxMiniLMEmbedder`` (``pip install '.[semantic]'``; ``avd embeddings fetch-model``):
  all-MiniLM-L6-v2 on ONNX Runtime, semantic but English-only (``onnx_minilm.py``).
* ``SentenceTransformerEmbedder`` (``pip install '.[ml]'``): semantic embeddings, e.g.
  all-MiniLM-L6-v2 (384 dims, same column size). Switching models requires re-embedding; every
  stored vector records the model that produced it and queries only compare like with like.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from itertools import pairwise
from typing import Protocol

from avd.config import EmbeddingProviderName, get_settings
from avd.db.types import EMBEDDING_DIM

_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "it",
        "this",
        "that",
        "i",
        "you",
        "we",
        "they",
        "my",
        "your",
        "our",
        "so",
        "as",
        "by",
        "from",
        "not",
        "do",
        "have",
        "has",
        "can",
        "will",
        "just",
        "about",
        "what",
        "how",
        "why",
        "who",
        "el",
        "la",
        "los",
        "las",
        "y",
        "de",
        "en",
        "que",
        "un",
        "una",
        "der",
        "die",
        "das",
        "und",
        "ist",
        "zu",
    ]
)


class Embedder(Protocol):
    model_id: str
    dim: int
    preprocessing: str
    relevance_floor: float
    relevance_ceiling: float

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    model_id = "hashing-uni-bigram-v1"
    dim = EMBEDDING_DIM
    preprocessing = "meta_keywords_v1"
    # Empirical similarity range: unrelated texts mostly fall below the floor, strong topical
    # matches approach the ceiling. Used to map raw cosine to a [0, 1] relevance feature.
    relevance_floor = 0.06
    relevance_ceiling = 0.33  # calibrated on synthetic_v1 personas (small sample; see docs)

    def _features(self, text: str) -> dict[str, float]:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]
        counts: dict[str, float] = {}
        for t in toks:
            counts[t] = counts.get(t, 0.0) + 1.0
        for a, b in pairwise(toks):
            key = f"{a}_{b}"
            counts[key] = counts.get(key, 0.0) + 0.5
        return {k: 1.0 + math.log(v) if v >= 1 else v for k, v in counts.items()}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat, w in self._features(text).items():
                h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                vec[idx] += sign * w
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:  # pragma: no cover - optional heavy dependency
    preprocessing = "chunk_mean_v1"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_id = model_name
        self.relevance_floor = 0.15
        self.relevance_ceiling = 0.65
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        if self.dim != EMBEDDING_DIM:
            raise ValueError(f"{model_name} has dim {self.dim}; the schema stores {EMBEDDING_DIM}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == EmbeddingProviderName.SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbedder(s.embedding_model)
    if s.embedding_provider == EmbeddingProviderName.ONNX_MINILM:
        from avd.embeddings.onnx_minilm import OnnxMiniLMEmbedder

        return OnnxMiniLMEmbedder(s.embedding_model_dir)
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


def calibrated_relevance(embedder: Embedder, similarity: float) -> float:
    lo, hi = embedder.relevance_floor, embedder.relevance_ceiling
    return max(0.0, min(1.0, (similarity - lo) / (hi - lo)))


def video_embedding_text(
    title: str, description: str, tags: list[str], topics: list[str], transcript_keywords: list[str] | None
) -> str:
    """Compact text for video embeddings: title (weighted), tags, topics, a description excerpt and
    the transcript's most frequent content words. Long raw text makes the 384-bucket hashing
    embedder collision-dominated (a short query almost always shares buckets with unrelated words);
    keywords keep the topical signal with far fewer features (see docs/evaluation.md)."""
    parts = [title, title, " ".join(tags), " ".join(topics), description[:400]]
    if transcript_keywords:
        parts.append(" ".join(transcript_keywords))
    return ". ".join(p for p in parts if p)
