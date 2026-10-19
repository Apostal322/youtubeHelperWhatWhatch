"""selection-policy versioning for recommendation impressions

* ``recommendation_policies``: each distinct selection configuration, written once.
* ``recommendation_impressions.selection_policy_version`` and ``selection_context``.

Existing impressions keep NULL: their selection policy was not recorded and is not guessed.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-20 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recommendation_policies",
        sa.Column("version", sa.String(length=80), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("first_used_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("version"),
    )
    with op.batch_alter_table("recommendation_impressions") as batch:
        batch.add_column(sa.Column("selection_policy_version", sa.String(length=80), nullable=True))
        batch.add_column(sa.Column("selection_context", sa.JSON(), nullable=True))
        batch.create_index(
            "ix_recommendation_impressions_selection_policy_version", ["selection_policy_version"]
        )


def downgrade() -> None:
    with op.batch_alter_table("recommendation_impressions") as batch:
        batch.drop_index("ix_recommendation_impressions_selection_policy_version")
        batch.drop_column("selection_context")
        batch.drop_column("selection_policy_version")
    op.drop_table("recommendation_policies")
