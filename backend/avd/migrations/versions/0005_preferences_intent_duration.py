"""preferences intent and duration range

Cold-start preferences: session intent and a minimum duration (with the existing maximum).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-16 00:30:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("user_preferences") as batch_op:
        batch_op.add_column(sa.Column("intent", sa.String(length=24), nullable=False, server_default="any"))
        batch_op.add_column(sa.Column("min_duration_minutes", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("user_preferences") as batch_op:
        batch_op.drop_column("min_duration_minutes")
        batch_op.drop_column("intent")
