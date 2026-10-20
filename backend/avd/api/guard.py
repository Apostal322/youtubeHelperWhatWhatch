"""Deployment guard: who may talk to this API at all, before any route runs.

Binding to 127.0.0.1 is not enough on its own: any web page the user visits can send requests to
``http://localhost:8000``, and a DNS-rebinding attack can make the browser treat an attacker's
domain as same-origin with the local API. So, for every HTTP request:

* **Host check (both modes).** The ``Host`` header must be one of ``allowed_hosts``. A rebinding
  attack arrives with the attacker's host name and is rejected (400).
* **Loopback only (local mode).** The TCP peer must be a loopback address (403 otherwise), so a
  local instance accidentally bound to ``0.0.0.0`` or port-forwarded still refuses other machines.
* **Origin check (local mode).** A state-changing request (POST/PUT/PATCH/DELETE) that carries a
  web ``Origin`` other than a browser extension or a local development page is refused (403):
  a malicious site cannot submit content, create accounts or trigger work on your machine.

Server mode relies on the host check plus authentication (bearer tokens in headers, never cookies,
so cross-site requests cannot carry credentials); ``avd serve`` refuses to start a server-mode
instance whose configuration is unsafe (``Settings.server_mode_problems``).
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from urllib.parse import urlparse

from avd.config import DeploymentMode, Settings

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
EXTENSION_SCHEMES = {"chrome-extension", "moz-extension", "safari-web-extension"}
# Starlette's TestClient reports this synthetic peer; real servers always report an IP address.
TEST_CLIENT_PEER = "testclient"


def is_loopback(host: str) -> bool:
    if host == TEST_CLIENT_PEER:
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def host_without_port(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("["):  # [::1]:8000
        return raw[: raw.find("]") + 1] if "]" in raw else raw
    return raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw


def trusted_origin(origin: str, settings: Settings) -> bool:
    if origin in settings.cors_allow_origins:
        return True
    u = urlparse(origin)
    if u.scheme in EXTENSION_SCHEMES:
        return True
    return u.scheme in ("http", "https") and (u.hostname or "") in ("localhost", "127.0.0.1", "::1")


class DeploymentGuard:
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self.allowed = {h.lower() for h in settings.allowed_hosts}
        self.any_host = "*" in self.allowed
        self.peer_nets = [ipaddress.ip_network(n, strict=False) for n in settings.local_peer_networks]

    def _local_peer(self, peer: str) -> bool:
        if is_loopback(peer):
            return True
        try:
            ip = ipaddress.ip_address(peer)
        except ValueError:
            return False
        return any(ip in n for n in self.peer_nets)

    async def _reject(self, send: Send, status: int, code: str, message: str) -> None:
        body = json.dumps({"error": {"code": code, "message": message}}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = host_without_port(headers.get("host", ""))
        # /health (status "ok" only) is exempt from the Host check so container health probes work.
        if not self.any_host and host not in self.allowed and scope.get("path") != "/health":
            await self._reject(send, 400, "invalid_host", "Host header not allowed (AVD_ALLOWED_HOSTS)")
            return
        if self.settings.deployment_mode == DeploymentMode.LOCAL:
            peer = (scope.get("client") or ("", 0))[0]
            if not self._local_peer(peer):
                await self._reject(
                    send,
                    403,
                    "local_only",
                    "This is a local (single-user) instance; it only accepts connections from this "
                    "computer. Use AVD_DEPLOYMENT_MODE=server for a shared server.",
                )
                return
            origin = headers.get("origin")
            if scope.get("method") in UNSAFE_METHODS and origin and not trusted_origin(origin, self.settings):
                await self._reject(
                    send, 403, "cross_site", "cross-site request refused by the local instance"
                )
                return
        await self.app(scope, receive, send)
