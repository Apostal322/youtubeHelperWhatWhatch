"""Portable column types.

``EmbeddingVector`` maps to pgvector's ``vector(n)`` on PostgreSQL and to JSON elsewhere (SQLite demo
mode / unit tests). Vector search is pushed down to pgvector when available and falls back to an
exact in-process scan otherwise (see ``avd.recsys.index``).
"""

from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator, TypeEngine

EMBEDDING_DIM = 384


class EmbeddingVector(TypeDecorator[list[float]]):
    impl = JSON
    cache_ok = True

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        values = [float(v) for v in value]
        if len(values) != self.dim:
            raise ValueError(f"embedding has dim {len(values)}, expected {self.dim}")
        return values

    def process_result_value(self, value: Any, dialect: Dialect) -> list[float] | None:
        if value is None:
            return None
        return [float(v) for v in value]
