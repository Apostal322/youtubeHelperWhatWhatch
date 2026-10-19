"""API operations: request correlation, structured errors, rate limiting, metrics and status.

* **Correlation.** Each request gets an ``X-Request-ID`` (a safe inbound value is reused). It is
  set as the correlation id for logs, enqueued jobs and model calls, and returned in the response.
* **Errors.** All errors share one shape, ``{"detail": ..., "error": {"code", "message",
  "request_id"}}``. Unhandled exceptions return a generic 500 without internals.
* **Rate limiting.** A token bucket per caller: the hash of the bearer or admin token, else the
  client IP, plus a per-IP bucket for every request (rotating tokens cannot multiply the
  allowance). Shared through Redis when ``AVD_REDIS_URL`` is set (atomic Lua on the Redis
  clock), per process otherwise or when Redis fails. ``/health`` and ``/metrics`` are exempt.
* **Metrics.** ``/metrics`` serves Prometheus text: in-process counters plus queue, dead-letter
  and spend gauges read from the database. It requires the admin token unless
  ``AVD_METRICS_PUBLIC=true``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from sqlalchemy import func, select, text
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from avd.config import Settings, get_settings
from avd.db.models import PipelineJob
from avd.db.session import get_sessionmaker
from avd.logging import get_logger
from avd.observability import METRICS, correlation_id, new_id

log = get_logger("avd.api")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
EXEMPT = ("/health", "/metrics")
ERROR_CODES = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    503: "unavailable",
}


def error_body(status: int, message: Any, code: str | None = None) -> dict[str, Any]:
    return {
        "detail": message,
        "error": {
            "code": code or ERROR_CODES.get(status, "internal_error" if status >= 500 else "error"),
            "message": message if isinstance(message, str) else "request validation failed",
            "request_id": correlation_id.get(),
        },
    }


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        inbound = request.headers.get("x-request-id", "")
        rid = inbound if _SAFE_ID.match(inbound) else new_id()
        token = correlation_id.set(rid)
        started = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            correlation_id.reset(token)
        elapsed = time.monotonic() - started
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")  # templated path: no ids or tokens in metrics
        METRICS.inc(
            "avd_http_requests_total", method=request.method, route=path, status=str(response.status_code)
        )
        METRICS.observe("avd_http_request_seconds", elapsed, route=path)
        response.headers["X-Request-ID"] = rid
        log.info(
            "request",
            extra={
                "method": request.method,
                "route": path,
                "status": response.status_code,
                "ms": round(elapsed * 1000, 1),
                "correlation_id": rid,
            },
        )
        return response


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
# The annotation UI is one self-contained page: inline script/style, same-origin API calls only.
ANNOTATE_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
    "img-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path == "/annotate":
            response.headers["Content-Security-Policy"] = ANNOTATE_CSP
        return response


class TokenBucket:
    """Per-process bucket (single API worker, or the fallback when Redis is unreachable)."""

    def __init__(self, rate_per_minute: int, burst: int) -> None:
        self.rate = rate_per_minute / 60.0
        self.burst = float(burst)
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str, now: float | None = None) -> float:
        """Consume one token; returns 0 when allowed, else the seconds until a token is available."""
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._state.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return 0.0
            self._state[key] = (tokens, now)
            return (1.0 - tokens) / self.rate if self.rate > 0 else 60.0


# Atomic refill-and-take on the Redis server's clock (no dependence on API hosts' clocks).
_REDIS_TAKE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local s = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(s[1]) or burst
local ts = tonumber(s[2]) or now
tokens = math.min(burst, tokens + math.max(0, now - ts) * rate)
local wait = 0
if tokens >= 1 then tokens = tokens - 1 else wait = (1 - tokens) / rate end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], math.ceil(burst / rate) + 60)
return tostring(wait)
"""


class RedisTokenBucket:
    """Shared across API processes and hosts. On a Redis error the caller falls back to the
    per-process bucket (degraded but still limiting) and the fallback is counted in /metrics."""

    def __init__(self, redis_url: str, rate_per_minute: int, burst: int, prefix: str) -> None:
        import redis

        self._r = redis.Redis.from_url(redis_url, socket_timeout=0.25, socket_connect_timeout=0.25)
        self._script = self._r.register_script(_REDIS_TAKE)
        self.rate = rate_per_minute / 60.0
        self.burst = burst
        self.prefix = prefix

    def take(self, key: str, now: float | None = None) -> float:
        return float(self._script(keys=[f"avd:api-rl:{self.prefix}:{key}"], args=[self.rate, self.burst]))


class RateLimiterPolicy:
    """Two buckets per request: the caller's (credential hash, else client IP) and the client IP's.
    Requests are refused when either is empty. The IP bucket bounds token rotation; the caller
    bucket keeps one heavy user behind a shared NAT from starving others."""

    def __init__(self, settings: Settings) -> None:
        self.caller_local = TokenBucket(
            max(1, settings.api_rate_limit_per_minute), settings.api_rate_limit_burst
        )
        self.ip_local = TokenBucket(
            max(1, settings.api_rate_limit_per_ip_per_minute), settings.api_rate_limit_ip_burst
        )
        self.caller_shared: RedisTokenBucket | None = None
        self.ip_shared: RedisTokenBucket | None = None
        if settings.redis_url:
            self.caller_shared = RedisTokenBucket(
                settings.redis_url,
                max(1, settings.api_rate_limit_per_minute),
                settings.api_rate_limit_burst,
                "caller",
            )
            self.ip_shared = RedisTokenBucket(
                settings.redis_url,
                max(1, settings.api_rate_limit_per_ip_per_minute),
                settings.api_rate_limit_ip_burst,
                "ip",
            )

    @property
    def backend(self) -> str:
        return "redis" if self.caller_shared else "process"

    def _take(self, shared: RedisTokenBucket | None, local: TokenBucket, key: str) -> float:
        if shared is not None:
            try:
                return shared.take(key)
            except Exception as exc:  # Redis down or slow: keep limiting locally, visibly
                METRICS.inc("avd_rate_limiter_fallback_total")
                log.warning(
                    "shared rate limiter unavailable; using per-process limits",
                    extra={"error": type(exc).__name__},
                )
        return local.take(key)

    def check(self, caller_key: str, ip_key: str) -> float:
        wait_ip = self._take(self.ip_shared, self.ip_local, ip_key)
        if wait_ip > 0:
            return wait_ip
        return self._take(self.caller_shared, self.caller_local, caller_key)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, settings: Settings) -> None:
        super().__init__(app)
        self.enabled = settings.api_rate_limit_per_minute > 0
        self.policy = RateLimiterPolicy(settings)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self.enabled or request.url.path.startswith(EXEMPT) or request.method == "OPTIONS":
            return await call_next(request)
        ip = "ip:" + (request.client.host if request.client else "unknown")
        cred = request.headers.get("authorization") or request.headers.get("x-admin-token")
        caller = "t:" + hashlib.sha256(cred.encode()).hexdigest()[:16] if cred else ip
        wait = self.policy.check(caller, ip)
        if wait > 0:
            METRICS.inc("avd_http_rate_limited_total")
            resp = JSONResponse(error_body(429, "rate limit exceeded; retry later"), status_code=429)
            resp.headers["Retry-After"] = str(max(1, int(wait + 0.999)))
            return resp
        return await call_next(request)


def install(app: FastAPI, settings: Settings) -> None:
    # Order: the request context wraps everything (added last = outermost).
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            error_body(exc.status_code, exc.detail), status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
        ]
        return JSONResponse(error_body(422, errors), status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"route": request.url.path})
        return JSONResponse(error_body(500, "internal error"), status_code=500)

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request) -> Response:
        s = get_settings()
        if not s.metrics_public:
            supplied = request.headers.get("x-admin-token", "")
            if s.admin_token is None or not hmac.compare_digest(supplied, s.admin_token.get_secret_value()):
                return JSONResponse(error_body(403, "metrics require the admin token"), status_code=403)
        gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        with get_sessionmaker()() as session:
            for st, n in session.execute(
                select(PipelineJob.status, func.count()).group_by(PipelineJob.status)
            ).all():
                gauges[("avd_jobs", (("status", st),))] = float(n)
            from avd.workers.budget import llm_spent_today

            gauges[("avd_llm_spend_today_usd", ())] = float(llm_spent_today(session))
        return PlainTextResponse(METRICS.render(gauges), media_type="text/plain; version=0.0.4")

    @app.get("/v1/status")
    def status() -> dict[str, Any]:
        """Operational status without secrets: database reachability, queue counts, evaluator
        mode and versions."""
        from avd.evaluation.pipeline import EvaluationPipeline

        s = get_settings()
        out: dict[str, Any] = {
            "evaluation_mode": "heuristic_baseline" if s.demo_mode else "llm",
            "llm_provider": s.llm_provider.value,
            "llm_credentials_configured": s.anthropic_api_key is not None or s.openai_api_key is not None,
            "embedding_provider": s.embedding_provider.value,
            "versions": EvaluationPipeline.versions(),
        }
        try:
            with get_sessionmaker()() as session:
                session.execute(text("SELECT 1"))
                out["queue"] = dict(
                    session.execute(
                        select(PipelineJob.status, func.count()).group_by(PipelineJob.status)
                    ).all()
                )
            out["database"] = "ok"
        except Exception as exc:  # report, do not leak
            log.warning("status check failed", extra={"error": type(exc).__name__})
            out["database"] = "unavailable"
        out["status"] = "ok" if out["database"] == "ok" else "degraded"
        return out
