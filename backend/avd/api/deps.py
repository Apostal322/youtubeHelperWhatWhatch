from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from avd.config import DeploymentMode, Settings, get_settings
from avd.db.models import PipelineJob, User
from avd.db.session import get_sessionmaker
from avd.embeddings.base import Embedder, get_embedder


def db() -> Iterator[Session]:
    """Request session; commits when the endpoint returns.

    Always declare it as ``Depends(db, scope="function")``. With FastAPI's default (``"request"``)
    the code after ``yield`` runs after the response has been sent, so a client could act on a
    response whose transaction had not committed yet, or had failed to commit
    (``tests/test_commit_before_response.py``)."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def current_user(
    authorization: str | None = Header(default=None), session: Session = Depends(db, scope="function")
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    user = session.scalar(select(User).where(User.token_hash == hash_token(authorization[7:].strip())))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
    return user


def optional_reader(
    authorization: str | None = Header(default=None),
    session: Session = Depends(db, scope="function"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Catalog and evaluation reads: open on local instances and on public servers; a private
    server (``AVD_REQUIRE_AUTH_FOR_READ=true``) requires a valid user token."""
    if settings.deployment_mode == DeploymentMode.SERVER and settings.require_auth_for_read:
        current_user(authorization, session)


def backlog_guard(
    session: Session = Depends(db, scope="function"), settings: Settings = Depends(get_settings)
) -> None:
    """Bounded queue: new work is refused while too many jobs wait, so a slow (CPU-only) machine is
    never buried under an unbounded backlog."""
    queued = session.scalar(
        select(func.count()).select_from(PipelineJob).where(PipelineJob.status == "queued")
    )
    if (queued or 0) >= settings.max_queued_jobs:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"evaluation queue is full ({queued} jobs waiting); try again later",
        )


def require_admin(
    x_admin_token: str | None = Header(default=None), settings: Settings = Depends(get_settings)
) -> None:
    if settings.admin_token is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin API disabled (set AVD_ADMIN_TOKEN)")
    if x_admin_token is None or not hmac.compare_digest(
        x_admin_token, settings.admin_token.get_secret_value()
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid admin token")


def embedder() -> Embedder:
    return get_embedder()
