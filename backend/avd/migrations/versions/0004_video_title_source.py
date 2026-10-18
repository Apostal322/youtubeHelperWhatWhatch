"""video title source

Records where a video's title came from, so placeholder titles (e.g. derived from a dataset file
path) are not judged for title consistency.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-15 23:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("videos") as batch_op:
        batch_op.add_column(sa.Column("title_source", sa.String(length=32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("videos") as batch_op:
        batch_op.drop_column("title_source")
