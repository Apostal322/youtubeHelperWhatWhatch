"""Semantic sentence embeddings with all-MiniLM-L6-v2 on ONNX Runtime (CPU, no PyTorch).

Model: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), 384 dimensions, max 256 word pieces.
The ONNX export is Chroma's (``chroma-onnx-models`` bucket), exported from Hugging Face snapshot
``7dbbc90392e2f80f3d3c277d6e90027e55de9125``. The archive and the model file are pinned by
SHA-256 and verified before use. Inference matches the sentence-transformers recipe: WordPiece
tokenisation (lower-cased), truncation to 256 tokens, mean pooling over the attention mask,
L2 normalisation.

Not verified here: numerical equivalence to the PyTorch sentence-transformers implementation.
Hugging Face is unreachable from the build environment, so the reference model could not be
loaded for comparison. The behavioural tests (paraphrase vs unrelated similarity) pass.

The model is English-only. Text in other languages embeds poorly; there is no multilingual
model in this repository.

Fetching is explicit (``avd embeddings fetch-model``). The embedder never downloads silently and
fails closed when the verified model is missing.

**ONNX Runtime telemetry is turned off.** Release 1.31 ships a telemetry client that, on Linux too,
connects to ``mobile.events.data.microsoft.com`` after a session is created, and tries to persist
a device identifier (observed with ``strace`` in Phase 3.4). This project promises no telemetry,
so ``disable_onnxruntime_telemetry`` sets ``ORT_DISABLE_TELEMETRY=1`` before onnxruntime is
imported and calls ``disable_telemetry_events()``. ``tests/test_no_telemetry.py`` checks, at the
system-call level, that loading and running the embedder opens no network connection.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np


def disable_onnxruntime_telemetry() -> Any:
    """Import onnxruntime with its telemetry off; returns the module."""
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    import onnxruntime as ort

    ort.disable_telemetry_events()
    return ort


ARCHIVE_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
ARCHIVE_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
MODEL_SHA256 = "4f148ba8ae9c2c7fbee4af2b132db8d06c6a6545b47fc83bbb98c3d22b8393e6"
TOKENIZER_SHA256 = "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"
HF_SNAPSHOT = "7dbbc90392e2f80f3d3c277d6e90027e55de9125"
MAX_TOKENS = 256
DIM = 384


class ModelNotAvailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(model_dir: Path) -> Path:
    """Download and verify the pinned archive into ``model_dir``. Returns the model directory."""
    import httpx

    target = model_dir / "all-MiniLM-L6-v2-onnx"
    if verify_model(target):
        return target
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir / "onnx.tar.gz.part"
    with httpx.stream("GET", ARCHIVE_URL, timeout=120.0, follow_redirects=True) as r:
        r.raise_for_status()
        with archive.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    if _sha256(archive) != ARCHIVE_SHA256:
        archive.unlink()
        raise ModelNotAvailable("downloaded archive checksum mismatch; refusing to use it")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name in ("model.onnx", "tokenizer.json", "config.json"):
                src = tar.extractfile(member)
                assert src is not None
                (target / name).write_bytes(src.read())
    archive.unlink()
    if not verify_model(target):
        raise ModelNotAvailable("extracted model files failed checksum verification")
    return target


def verify_model(target: Path) -> bool:
    m, t = target / "model.onnx", target / "tokenizer.json"
    return m.exists() and t.exists() and _sha256(m) == MODEL_SHA256 and _sha256(t) == TOKENIZER_SHA256


class OnnxMiniLMEmbedder:
    model_id = f"all-MiniLM-L6-v2-onnx@{MODEL_SHA256[:12]}"
    dim = DIM
    preprocessing = "chunk_mean_v1"
    # Raw cosine -> [0, 1] relevance mapping for the recommender, set from the retrieval experiment
    # (benchmarks/reports/retrieval-talkpython-n100-s7.md: title->transcript cosine, non-relevant
    # median 0.20, relevant median 0.48). One corpus, 100 queries: a starting point, not calibration.
    relevance_floor = 0.20
    relevance_ceiling = 0.50

    def __init__(self, model_dir: Path, threads: int = 2) -> None:
        ort = disable_onnxruntime_telemetry()
        from tokenizers import Tokenizer

        target = model_dir / "all-MiniLM-L6-v2-onnx"
        if not verify_model(target):
            raise ModelNotAvailable(
                f"verified MiniLM ONNX model not found in {target}; run 'avd embeddings fetch-model'"
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(target / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        tok = Tokenizer.from_file(str(target / "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_TOKENS)
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tok = tok

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            enc = self._tok.encode_batch([t or " " for t in texts[i : i + batch_size]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = self._session.run(None, feeds)[0]  # (batch, seq, 384)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.extend(pooled.astype(float).tolist())
        return out

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text).ids)
