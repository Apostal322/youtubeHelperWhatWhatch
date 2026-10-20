from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from avd.api import ops
from avd.api.guard import DeploymentGuard
from avd.api.routers import admin, annotation, public, users
from avd.config import get_settings
from avd.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if problems := settings.server_mode_problems():
        # Fail closed: a shared server never starts with an unsafe configuration.
        raise RuntimeError("unsafe server configuration: " + "; ".join(problems))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        worker = None
        if settings.embedded_worker:
            from avd.workers.context import build_context
            from avd.workers.embedded import EmbeddedWorker

            worker = EmbeddedWorker(build_context(), settings.embedded_worker_poll_seconds)
            worker.start()
        app.state.worker = worker
        yield
        if worker is not None:
            worker.stop()

    app = FastAPI(
        title="AI Video Discovery",
        version="0.1.0",
        description="Popularity-blind, multidimensional video quality assessment and recommendations.",
        lifespan=lifespan,
    )
    app.state.worker = None
    ops.install(app, settings)
    app.add_middleware(DeploymentGuard, settings=settings)
    # Added last = outermost, so error and 429 responses also carry CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_origin_regex=settings.cors_allow_origin_regex,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Admin-Token", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "demo_mode": settings.demo_mode}

    app.include_router(public.router)
    app.include_router(users.router)
    app.include_router(admin.router)
    app.include_router(annotation.router)
    app.include_router(annotation.ui_router)
    return app


app = create_app()
