"""How a video becomes one vector, with the metadata needed for audits and safe re-indexing.

Two document recipes (``Embedder.preprocessing``):

* ``meta_keywords_v1`` (hashing embedder): title (x2), tags, topics, a description excerpt and
  the transcript's most frequent content words, as one compact text. Long raw text makes the
  384-bucket hashing space collision-dominated.
* ``chunk_mean_v1`` (sentence encoders, 256-token window): one metadata text plus up to 24
  evenly spaced ~180-word transcript windows. Each part is embedded; the window vectors are
  averaged, added to the metadata vector, and the sum is L2-normalised. This covers long
  transcripts that would otherwise be truncated to their first ~200 words.

Every stored vector records model id, dimension, recipe, a content hash of the exact inputs, and
the time. Re-indexing re-embeds a row only when one of these is stale.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime

from avd.content.models import Transcript
from avd.db.models import Video
from avd.embeddings.base import Embedder, video_embedding_text
from avd.evaluation.topics import transcript_keywords

WINDOW_WORDS = 180
MAX_WINDOWS = 24


@dataclass(frozen=True)
class EmbeddingRecord:
    vector: list[float]
    model_id: str
    dim: int
    preprocessing: str
    input_kind: str  # "metadata" | "transcript"
    content_hash: str


def _windows(text: str, size: int = WINDOW_WORDS, limit: int = MAX_WINDOWS) -> list[str]:
    words = text.split()
    if not words:
        return []
    starts = list(range(0, len(words), size))
    if len(starts) > limit:  # evenly spaced, deterministic
        step = len(starts) / limit
        starts = [starts[int(i * step)] for i in range(limit)]
    return [" ".join(words[s : s + size]) for s in starts]


def document_inputs(embedder: Embedder, video: Video, transcript: Transcript | None) -> tuple[str, list[str]]:
    from avd.evaluation.sanitize import sanitize_description, sanitize_tags, sanitize_title

    recipe = getattr(embedder, "preprocessing", "meta_keywords_v1")
    # The same popularity stripping as the evaluator's input: "1M views", "trending" and the like
    # must not shape retrieval either. Changed text changes the content hash, so `avd embeddings
    # reindex` re-embeds affected videos.
    title = sanitize_title(video.title)[0]
    description = sanitize_description(video.description or "")[0]
    tags, topics = list(sanitize_tags(list(video.tags or []))[0]), list(video.topics or [])
    if recipe == "chunk_mean_v1":
        meta = ". ".join(p for p in (title, " ".join(tags), description[:600]) if p)
        return recipe, [meta, *(_windows(transcript.text) if transcript is not None else [])]
    keywords = transcript_keywords(transcript.text) if transcript is not None else None
    return recipe, [video_embedding_text(title, description, tags, topics, keywords)]


def embed_video(embedder: Embedder, video: Video, transcript: Transcript | None) -> EmbeddingRecord:
    recipe, texts = document_inputs(embedder, video, transcript)
    content_hash = hashlib.sha256("\x1e".join([recipe, *texts]).encode()).hexdigest()
    if recipe == "chunk_mean_v1" and len(texts) > 1:
        has_meta = bool(texts[0].strip())  # an empty metadata text would only add noise
        vecs = embedder.embed(texts if has_meta else texts[1:])
        windows = vecs[1:] if has_meta else vecs
        mean = [sum(col) / len(windows) for col in zip(*windows, strict=True)]
        summed = [a + b for a, b in zip(vecs[0], mean, strict=True)] if has_meta else mean
        norm = math.sqrt(sum(x * x for x in summed)) or 1.0
        vector = [x / norm for x in summed]
    else:
        vector = embedder.embed(texts[:1])[0]
    return EmbeddingRecord(
        vector=vector,
        model_id=embedder.model_id,
        dim=len(vector),
        preprocessing=recipe,
        input_kind="transcript" if transcript is not None else "metadata",
        content_hash=content_hash,
    )


def apply(video: Video, rec: EmbeddingRecord) -> None:
    video.embedding = rec.vector
    video.embedding_model = rec.model_id
    video.embedding_dim = rec.dim
    video.embedding_preprocessing = rec.preprocessing
    video.embedding_input = rec.input_kind
    video.embedding_content_hash = rec.content_hash
    video.embedded_at = datetime.now(UTC)


def is_stale(embedder: Embedder, video: Video, transcript: Transcript | None) -> bool:
    if video.embedding is None or video.embedding_model != embedder.model_id:
        return True
    recipe, texts = document_inputs(embedder, video, transcript)
    content_hash = hashlib.sha256("\x1e".join([recipe, *texts]).encode()).hexdigest()
    return video.embedding_preprocessing != recipe or video.embedding_content_hash != content_hash
