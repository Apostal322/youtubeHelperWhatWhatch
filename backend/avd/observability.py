"""Correlation context and in-process metrics.

* ``correlation_id`` ties an API request, the jobs it enqueues, the evaluation run and every model
  call together in logs (``X-Request-ID`` header in, ``correlation_id`` field out).
* ``job_id`` / ``evaluation_run_id`` identify background work.
* A per-job spend accumulator lets the spend guard enforce ``AVD_MAX_JOB_COST_USD`` across every
  model call a job makes (including retries and map-reduce chunk calls).
* ``METRICS`` is a minimal Prometheus-compatible registry (counters + latency summaries) with no
  external dependency; DB-derived gauges are added at scrape time by the API.
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)
job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)
evaluation_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "evaluation_run_id", default=None
)
_job_spend: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar("job_spend", default=None)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def job_context(job: str, correlation: str | None = None) -> Iterator[None]:
    t_job = job_id.set(job)
    t_corr = correlation_id.set(correlation or correlation_id.get() or new_id())
    t_spend = _job_spend.set([0.0])
    try:
        yield
    finally:
        _job_spend.reset(t_spend)
        correlation_id.reset(t_corr)
        job_id.reset(t_job)


@contextmanager
def evaluation_run_context() -> Iterator[str]:
    run = new_id()
    token = evaluation_run_id.set(run)
    try:
        yield run
    finally:
        evaluation_run_id.reset(token)


def add_job_spend(usd: float) -> None:
    acc = _job_spend.get()
    if acc is not None:
        acc[0] += usd


def job_spend() -> float:
    acc = _job_spend.get()
    return acc[0] if acc is not None else 0.0


def context_fields() -> dict[str, str]:
    out = {}
    for name, var in (
        ("correlation_id", correlation_id),
        ("job_id", job_id),
        ("evaluation_run_id", evaluation_run_id),
    ):
        value = var.get()
        if value:
            out[name] = value
    return out


@dataclass
class _Summary:
    count: int = 0
    total: float = 0.0
    samples: list[float] = field(default_factory=list)

    def observe(self, v: float) -> None:
        self.count += 1
        self.total += v
        self.samples.append(v)
        if len(self.samples) > 2048:  # bounded reservoir (most recent)
            del self.samples[:1024]


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self.summaries: dict[tuple[str, tuple[tuple[str, str], ...]], _Summary] = {}

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self.summaries.setdefault(key, _Summary()).observe(value)

    def render(self, extra_gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] | None = None) -> str:
        def fmt(name: str, labels: tuple[tuple[str, str], ...]) -> str:
            if not labels:
                return name
            inner = ",".join(f'{k}="{_esc(v)}"' for k, v in labels)
            return f"{name}{{{inner}}}"

        lines: list[str] = []
        with self._lock:
            for (name, labels), v in sorted(self.counters.items()):
                lines.append(f"{fmt(name, labels)} {v}")
            for (name, labels), s in sorted(self.summaries.items()):
                ordered = sorted(s.samples)
                for q in (0.5, 0.95):
                    if ordered:
                        val = ordered[min(len(ordered) - 1, int(q * len(ordered)))]
                        lines.append(f"{fmt(name, (*labels, ('quantile', str(q))))} {val}")
                lines.append(f"{fmt(name + '_count', labels)} {s.count}")
                lines.append(f"{fmt(name + '_sum', labels)} {s.total}")
        for (name, labels), v in sorted((extra_gauges or {}).items()):
            lines.append(f"{fmt(name, labels)} {v}")
        return "\n".join(lines) + "\n"


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


METRICS = Metrics()
