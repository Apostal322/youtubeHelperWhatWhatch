"""No telemetry from third-party native code.

ONNX Runtime 1.31 contains a telemetry client that, after a session is created, opens a
connection to ``mobile.events.data.microsoft.com`` (seen through the egress proxy in Phase 3.4)
and keeps a device identifier and an offline event store under
``~/.cache/Microsoft/DeveloperTools``. Python's audit hooks cannot see native sockets, so this
test runs the embedder in a child process with its own ``HOME`` and with ``HTTPS_PROXY``
pointing at a listener held by this test, and checks that neither is touched.
"""

from __future__ import annotations

import os
import select
import socket
import subprocess
import sys
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / ".model_cache"

CHILD = """
import sys
from pathlib import Path
from avd.embeddings.onnx_minilm import OnnxMiniLMEmbedder
OnnxMiniLMEmbedder(Path(sys.argv[1])).embed(["a short sentence to embed", "another one"])
"""


@pytest.mark.skipif(
    not (MODEL_DIR / "all-MiniLM-L6-v2-onnx" / "model.onnx").exists(), reason="MiniLM model not fetched"
)
def test_embedder_sends_no_telemetry(tmp_path: Path) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    proxy = f"http://127.0.0.1:{listener.getsockname()[1]}"
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("no_proxy", "ort_disable_telemetry")}
    env.update(HOME=str(tmp_path), HTTPS_PROXY=proxy, HTTP_PROXY=proxy, https_proxy=proxy, http_proxy=proxy)
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(MODEL_DIR)],
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    ready, _, _ = select.select([listener], [], [], 0.5)
    listener.close()
    assert not ready, "the embedder process opened a network connection (telemetry?)"
    assert not (tmp_path / ".cache" / "Microsoft").exists(), "ONNX Runtime created its telemetry store"
