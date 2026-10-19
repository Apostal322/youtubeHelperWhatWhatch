"""Blind annotation: model assessments of items in an open annotation dataset are withheld from
public endpoints, so a rater cannot look up the AI's view of an item (by title search, video id or
recommendation feed) before rating it. Lifted per dataset with ``avd annotate unblind``."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.db.models import Dataset, DatasetItem

WITHHELD_MESSAGE = "assessment withheld while blind human annotation of this item is open"


def blinded_video_ids(session: Session) -> set[str]:
    return set(
        session.scalars(
            select(DatasetItem.video_id)
            .join(Dataset, Dataset.id == DatasetItem.dataset_id)
            .where(Dataset.public_blinding.is_(True))
        ).all()
    )


def is_blinded(session: Session, video_id: str) -> bool:
    return video_id in blinded_video_ids(session)
