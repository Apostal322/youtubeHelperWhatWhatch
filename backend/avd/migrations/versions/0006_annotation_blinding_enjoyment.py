"""annotation blinding and personal enjoyment

* ``datasets.public_blinding``: while annotation is open, model assessments of the dataset's items
  are withheld from public endpoints (raters cannot look them up before rating).
* ``annotations.enjoyment``: optional personal enjoyment (1-5), kept apart from quality labels.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-19 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("datasets") as batch_op:
        batch_op.add_column(
            sa.Column("public_blinding", sa.Boolean(), nullable=False, server_default=sa.true())
        )
    with op.batch_alter_table("annotations") as batch_op:
        batch_op.add_column(sa.Column("enjoyment", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("annotations") as batch_op:
        batch_op.drop_column("enjoyment")
    with op.batch_alter_table("datasets") as batch_op:
        batch_op.drop_column("public_blinding")
