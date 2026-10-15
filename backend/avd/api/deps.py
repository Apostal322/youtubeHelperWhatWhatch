from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.config import Settings, get_settings
from avd.db.models import User
from avd.db.session import get_sessionmaker
from avd.embeddings.base import Embedder, get_embedder


def db() -> Iterator[Session]:
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


def current_user(authorization: str | None = Header(default=None), session: Session = Depends(db)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    user = session.scalar(select(User).where(User.token_hash == hash_token(authorization[7:].strip())))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
    return user


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
