"""Phase 2 P4: embedding metadata, document recipes, safe re-indexing and the ONNX MiniLM model."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from avd.config import get_settings
from avd.content.models import SourceType
from avd.content.store import add_transcript_source
from avd.db.models import Video
from avd.embeddings import documents
from avd.embeddings.base import HashingEmbedder, cosine
from avd.embeddings.reindex import reindex
from avd.ingest import upsert_channel
from avd.recsys.index import vector_search
from tests.conftest import DB, LECTURE, make_transcript


class FakeSentenceEmbedder:
    """Deterministic stand-in with the chunk recipe (records what it was asked to embed)."""

    model_id = "fake-sentence-v1"
    dim = 4
    preprocessing = "chunk_mean_v1"
    relevance_floor = 0.1
    relevance_ceiling = 0.5

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, float(len(t) % 7), 0.5, 0.0] for t in texts]


def test_record_carries_full_provenance() -> None:
    emb = HashingEmbedder()
    v = Video(id="v1", title="Photosynthesis explained", description="Biology", tags=["plants"], topics=[])
    a = documents.embed_video(emb, v, make_transcript(LECTURE))
    b = documents.embed_video(emb, v, make_transcript(LECTURE))
    assert a.model_id == emb.model_id and a.dim == 384 and a.preprocessing == "meta_keywords_v1"
    assert a.input_kind == "transcript" and a.content_hash == b.content_hash and len(a.content_hash) == 64
    assert documents.embed_video(emb, v, None).input_kind == "metadata"


def test_chunk_recipe_caps_windows_and_skips_empty_metadata() -> None:
    emb = FakeSentenceEmbedder()
    long = make_transcript(["word " * 100] * 100)  # 10,000 words -> 56 windows -> capped
    v = Video(id="v1", title="", description="", tags=[], topics=[])
    rec = documents.embed_video(emb, v, long)
    assert len(emb.calls[0]) == documents.MAX_WINDOWS  # no empty metadata text embedded
    assert rec.preprocessing == "chunk_mean_v1" and abs(sum(x * x for x in rec.vector) - 1) < 1e-9
    v.title = "A title"
    documents.embed_video(emb, v, long)
    assert len(emb.calls[1]) == documents.MAX_WINDOWS + 1 and emb.calls[1][0].startswith("A title")


def test_staleness_tracks_model_recipe_and_content() -> None:
    emb = HashingEmbedder()
    v = Video(id="v1", title="Photosynthesis", description="", tags=[], topics=[])
    t = make_transcript(LECTURE)
    assert documents.is_stale(emb, v, t)
    documents.apply(v, documents.embed_video(emb, v, t))
    assert not documents.is_stale(emb, v, t) and v.embedded_at is not None and v.embedding_dim == 384
    assert documents.is_stale(emb, v, make_transcript(["something else entirely"] * 30))
    v.embedding_model = "old-model"
    assert documents.is_stale(emb, v, t)


def _seed(db: DB, n: int = 6) -> None:
    with db.sf() as s:
        upsert_channel(s, "c1", "Channel")
        for i in range(n):
            s.add(
                Video(id=f"v{i}", channel_id="c1", title=f"Episode {i} about topic {i}", platform="youtube")
            )
            s.flush()
            add_transcript_source(
                s,
                video_id=f"v{i}",
                transcript=make_transcript([f"Topic {i} " + LECTURE[i % len(LECTURE)]] * 10),
                source_type=SourceType.USER_PROVIDED_TRANSCRIPT,
                authorization_basis="test",
            )
        s.commit()


def test_reindex_is_resumable_and_never_mixes_vector_spaces(db: DB) -> None:
    _seed(db)
    emb = HashingEmbedder()
    with db.sf() as s:
        for i, v in enumerate(s.scalars(select(Video).order_by(Video.id)).all()):
            v.embedding = [1.0] + [0.0] * 383
            v.embedding_model = "old-model" if i % 2 else emb.model_id
        s.commit()
        assert len(vector_search(s, [1.0] + [0.0] * 383, 10, emb.model_id, None)) <= 3
    dry = reindex(db.sf, emb, batch_size=2, dry_run=True)
    assert dry["would_reembed"] == 6 and dry["rows_with_other_model_vectors"] == 3
    r = reindex(db.sf, emb, batch_size=2)
    assert r["reembedded"] == 6 and r["rows_with_other_model_vectors"] == 0
    assert reindex(db.sf, emb, batch_size=4)["reembedded"] == 0  # idempotent
    with db.sf() as s:
        v = s.get(Video, "v3")
        assert v is not None and v.embedding_preprocessing == "meta_keywords_v1" and v.embedding_content_hash


def test_hashing_collision_regression() -> None:
    """Short queries must not look related to unrelated documents through bucket collisions; the
    compact keyword recipe keeps unrelated similarity below the relevance floor."""
    emb = HashingEmbedder()
    q = emb.embed(["photosynthesis chlorophyll light reactions"])[0]
    unrelated = Video(
        id="u", title="Medieval castle architecture", description="", tags=["history"], topics=[]
    )
    t = make_transcript(["The keep and the curtain wall defended the bailey against siege engines."] * 40)
    d = documents.embed_video(emb, unrelated, t).vector
    assert cosine(q, d) < emb.relevance_floor


MODEL_DIR = get_settings().embedding_model_dir


def _onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401

        from avd.embeddings.onnx_minilm import verify_model
    except ImportError:
        return False
    return verify_model(MODEL_DIR / "all-MiniLM-L6-v2-onnx")


def test_onnx_embedder_fails_closed_without_verified_model(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from avd.embeddings.onnx_minilm import ModelNotAvailable, OnnxMiniLMEmbedder

    (tmp_path / "all-MiniLM-L6-v2-onnx").mkdir()
    (tmp_path / "all-MiniLM-L6-v2-onnx" / "model.onnx").write_bytes(b"tampered")
    (tmp_path / "all-MiniLM-L6-v2-onnx" / "tokenizer.json").write_text("{}")
    with pytest.raises(ModelNotAvailable):
        OnnxMiniLMEmbedder(tmp_path)


@pytest.mark.skipif(
    not _onnx_available(), reason="MiniLM ONNX model not fetched (avd embeddings fetch-model)"
)
def test_onnx_minilm_is_semantic() -> None:
    from avd.embeddings.onnx_minilm import OnnxMiniLMEmbedder

    e = OnnxMiniLMEmbedder(MODEL_DIR)
    a, b, c = e.embed(
        [
            "How do I fix a flat bicycle tire?",
            "Repairing a punctured bike tyre step by step",
            "The history of the Roman senate",
        ]
    )
    assert len(a) == 384 and abs(sum(x * x for x in a) - 1) < 1e-5
    assert cosine(a, b) > 0.4 > 0.15 > cosine(a, c)
    # Batching does not change results.
    assert (
        max(
            abs(x - y)
            for x, y in zip(e.embed(["Repairing a punctured bike tyre step by step"])[0], b, strict=True)
        )
        < 1e-5
    )
    assert e.count_tokens("word " * 1000) == 256  # truncated to the model's window
