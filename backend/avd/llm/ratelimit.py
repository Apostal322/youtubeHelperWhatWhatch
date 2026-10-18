"""Token-bucket rate limiting, in-process or shared across workers through Redis."""

from __future__ import annotations

import threading
import time
from typing import Protocol

from avd.logging import get_logger

log = get_logger(__name__)


class RateLimiter(Protocol):
    def acquire(self, key: str, timeout: float = 60.0) -> bool: ...


class InMemoryTokenBucket:
    def __init__(self, rate_per_minute: int, burst: int | None = None) -> None:
        self.rate = rate_per_minute / 60.0
        self.capacity = float(burst or max(1, rate_per_minute // 6))
        self._tokens: dict[str, float] = {}
        self._ts: dict[str, float] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                tokens = self._tokens.get(key, self.capacity)
                tokens = min(self.capacity, tokens + (now - self._ts.get(key, now)) * self.rate)
                self._ts[key] = now
                if tokens >= 1.0:
                    self._tokens[key] = tokens - 1.0
                    return True
                self._tokens[key] = tokens
                wait = (1.0 - tokens) / self.rate
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 1.0))


# Atomic refill+take in Redis so all Celery workers share one budget per key.
_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + (now - ts) * rate)
local ok = 0
if tokens >= 1 then tokens = tokens - 1 ok = 1 end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, 3600)
if ok == 1 then return -1 end
return tostring((1 - tokens) / rate)
"""


class RedisTokenBucket:
    def __init__(self, redis_url: str, rate_per_minute: int, burst: int | None = None) -> None:
        import redis

        self._r = redis.Redis.from_url(redis_url)
        self._script = self._r.register_script(_LUA)
        self.rate = rate_per_minute / 60.0
        self.capacity = burst or max(1, rate_per_minute // 6)

    def acquire(self, key: str, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            res = self._script(keys=[f"avd:rl:{key}"], args=[self.rate, self.capacity, time.time()])
            wait = float(res)
            if wait < 0:
                return True
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 1.0))
