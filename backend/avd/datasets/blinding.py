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


def blinding_note(session: Session) -> str | None:
    """A user-facing explanation when open blind annotation hides items, so an empty or short list
    is not mistaken for an empty catalog."""
    rows = session.execute(
        select(Dataset.id, DatasetItem.video_id)
        .join(DatasetItem, DatasetItem.dataset_id == Dataset.id)
        .where(Dataset.public_blinding.is_(True))
    ).all()
    if not rows:
        return None
    datasets = sorted({d for d, _ in rows})
    return (
        f"{len({v for _, v in rows})} imported videos are hidden while blind human annotation of "
        f"{', '.join(datasets)} is open, so raters cannot see the automated view first. To use them "
        f"for yourself on this instance, run: avd annotate unblind --dataset {datasets[0]}"
    )
