"""Turn logged impressions + explicit feedback into pairwise training data for LTR."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from avd.db.models import Feedback, RecommendationImpression

POSITIVE = {"recommendation_useful"}
NEGATIVE = {"recommendation_not_useful", "not_interested"}

Pair = tuple[dict[str, float], dict[str, float]]


def build_pairs(session: Session) -> list[Pair]:
    """Pairs (preferred, other) within the same recommendation request:

    * explicit: a positively rated item vs. a negatively rated item;
    * skip-above: a positively rated item vs. unrated items shown *above* it (the user scrolled past
      them), a standard implicit-feedback heuristic with known position bias caveats.
    """
    label: dict[tuple[str, str], int] = {}
    for user_id, video_id, imp_id, kind in session.execute(
        select(Feedback.user_id, Feedback.video_id, Feedback.impression_id, Feedback.kind)
    ).all():
        val = 1 if kind in POSITIVE else -1 if kind in NEGATIVE else 0
        if val and imp_id:
            label[("imp", imp_id)] = val
        elif val:
            label[(user_id, video_id)] = val
    by_request: dict[str, list[RecommendationImpression]] = defaultdict(list)
    for imp in session.scalars(select(RecommendationImpression)).all():
        by_request[imp.request_id].append(imp)
    pairs: list[Pair] = []
    for imps in by_request.values():
        imps.sort(key=lambda i: i.position)

        def lab(i: RecommendationImpression) -> int:
            return label.get(("imp", i.id), label.get((i.user_id, i.video_id), 0))

        for pos in (i for i in imps if lab(i) > 0):
            for other in imps:
                if other is pos:
                    continue
                if lab(other) < 0 or (lab(other) == 0 and other.position < pos.position):
                    pairs.append((pos.features, other.features))
    return pairs
