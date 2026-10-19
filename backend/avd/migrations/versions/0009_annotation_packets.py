"""offline annotation packets

* ``annotation_packets``: blind assignments issued to a rater as a self-contained file, so a
  rating round needs no server that raters connect to.
* ``relevance_judgments``: graded human query-item relevance (relevance_v1 queries).

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-20 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "annotation_packets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("rater_id", sa.String(length=36), nullable=False),
        sa.Column("annotation_ids", sa.JSON(), nullable=False),
        sa.Column("secret_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("returned_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["rater_id"], ["raters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_annotation_packets_dataset_id", "annotation_packets", ["dataset_id"])
    op.create_index("ix_annotation_packets_rater_id", "annotation_packets", ["rater_id"])
    op.create_table(
        "relevance_judgments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("query_id", sa.String(length=16), nullable=False),
        sa.Column("video_id", sa.String(length=32), nullable=False),
        sa.Column("rater_id", sa.String(length=36), nullable=False),
        sa.Column("grade", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["rater_id"], ["raters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("query_id", "video_id", "rater_id"),
    )
    op.create_index("ix_relevance_judgments_rater_id", "relevance_judgments", ["rater_id"])
    op.create_index("ix_relevance_judgments_video_id", "relevance_judgments", ["video_id"])


def downgrade() -> None:
    op.drop_index("ix_relevance_judgments_video_id", table_name="relevance_judgments")
    op.drop_index("ix_relevance_judgments_rater_id", table_name="relevance_judgments")
    op.drop_table("relevance_judgments")
    op.drop_index("ix_annotation_packets_rater_id", table_name="annotation_packets")
    op.drop_index("ix_annotation_packets_dataset_id", table_name="annotation_packets")
    op.drop_table("annotation_packets")
