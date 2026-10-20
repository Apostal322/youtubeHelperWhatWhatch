"""PROTOCOL MOCK of the Ollama HTTP API, for tests only.

This is not a model and produces no judgement worth anything: ``/api/chat`` answers with a
scripted, schema-shaped JSON object that cites the first transcript segment it was shown,
verbatim. It exists so the client, error handling, context sizing and the evaluation path can be
tested without a GPU, a model download or a running server. Any report produced through it says
the transport was injected.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx

from avd.evaluation.evaluators.llm_rubric import TEXT_DIMENSIONS

_SEG = re.compile(r"^\[(\d+)\] \(\d[^)]*\) (.+)$", re.M)


def scripted_answer(body: dict[str, Any]) -> dict[str, Any]:
    user = body["messages"][-1]["content"]
    props = (body.get("format") or {}).get("properties", {})
    if "judgments" not in props:  # chunk notes / other schemas: empty but valid
        m = _SEG.search(user)
        out: dict[str, Any] = {k: ([] if v.get("type") == "array" else "") for k, v in props.items()}
        if m and "points" in props:
            out["summary"] = "part summary"
            out["points"] = [
                {
                    "segment_index": int(m.group(1)),
                    "quote": " ".join(m.group(2).split()[:8]),
                    "kind": "explanation",
                    "note": "n",
                }
            ]
        return out
    m = _SEG.search(user)
    assert m, "prompt must contain numbered segments"
    seg, text = int(m.group(1)), " ".join(m.group(2).split()[:8])
    return {
        "content_summary": "summary",
        "detected_issues": [],
        "judgments": [
            {
                "dimension": d.value,
                "assessable": True,
                "score": 3,
                "confidence": 0.6,
                "justification": "Cited segment supports a middling judgement.",
                "evidence": [{"segment_index": seg, "quote": text}],
                "counter_evidence": [],
                "unassessable_reason": "",
            }
            for d in TEXT_DIMENSIONS
        ],
    }


class FakeOllama:
    def __init__(
        self,
        models: dict[str, dict[str, Any]] | None = None,
        answer: Callable[[dict[str, Any]], Any] = scripted_answer,
        version: str = "0.0.0-mock",
    ) -> None:
        self.models = (
            models
            if models is not None
            else {
                "qwen3.5:4b": {
                    "digest": "sha256:" + "ab" * 32,
                    "parameter_size": "4.7B",
                    "quantization_level": "Q4_K_M",
                    "family": "qwen35",
                    "context_length": 262144,
                    "capabilities": ["completion", "vision", "thinking"],
                }
            }
        )
        self.answer = answer
        self.version = version
        self.down = False
        self.timeout = False
        self.chat_status: int | None = None
        self.chat_override: dict[str, Any] = {}  # merged into the /api/chat payload
        self.chat_bodies: list[dict[str, Any]] = []
        self.pulled: list[str] = []
        self.show_extra: dict[str, Any] = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [{"name": n, "model": n, "digest": m["digest"]} for n, m in self.models.items()]
                },
            )
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n, "size_vram": 0} for n in self.models]})
        body = json.loads(request.content or b"{}")
        if path == "/api/show":
            m = self.models.get(body["model"])
            if m is None:
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            return httpx.Response(
                200,
                json={
                    "details": {
                        "parameter_size": m["parameter_size"],
                        "quantization_level": m["quantization_level"],
                        "family": m["family"],
                    },
                    "model_info": {f"{m['family']}.context_length": m["context_length"]},
                    "capabilities": m["capabilities"],
                    **self.show_extra,
                },
            )
        if path == "/api/pull":
            self.pulled.append(body["model"])
            return httpx.Response(200, json={"status": "success"})
        if path == "/api/chat":
            if self.timeout:
                raise httpx.ReadTimeout("timed out", request=request)
            if body["model"] not in self.models:
                return httpx.Response(404, json={"error": "model not found"})
            if self.chat_status:
                return httpx.Response(self.chat_status, text="server error")
            self.chat_bodies.append(body)
            data = self.answer(body)
            content = data if isinstance(data, str) else json.dumps(data)
            prompt_chars = sum(len(m["content"]) for m in body["messages"])
            payload = {
                "model": body["model"],
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": prompt_chars // 4,
                "eval_count": max(1, len(content) // 4),
                "load_duration": 1_000_000,
                "prompt_eval_duration": 50_000_000,
                "eval_duration": 100_000_000,
                **self.chat_override,
            }
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "unknown path"})
