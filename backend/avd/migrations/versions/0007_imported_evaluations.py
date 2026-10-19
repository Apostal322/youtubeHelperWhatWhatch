"""imported evaluations (portable export/import)

* ``imported_evaluations``: evaluations from other installations' ``avd-evaluation-export``
  files, stored apart from local evaluations and never treated as authoritative.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-19 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "imported_evaluations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("origin", sa.String(length=200), nullable=False),
        sa.Column("evaluation_id", sa.String(length=64), nullable=False),
        sa.Column("video_id", sa.String(length=64), nullable=False),
        sa.Column("format_version", sa.String(length=16), nullable=False),
        sa.Column("file_sha256", sa.String(length=64), nullable=False),
        sa.Column("evaluator_source", sa.String(length=16), nullable=False),
        sa.Column("models", sa.JSON(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("origin", "evaluation_id"),
    )
    op.create_index("ix_imported_evaluations_video_id", "imported_evaluations", ["video_id"])


def downgrade() -> None:
    op.drop_index("ix_imported_evaluations_video_id", table_name="imported_evaluations")
    op.drop_table("imported_evaluations")
