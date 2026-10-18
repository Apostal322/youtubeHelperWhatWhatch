"""Transparent, genre-specific aggregation.

group score   = sum(w_d * s_d) / sum(w_d)                 over *scored* applicable dims in the group
group coverage= sum(w_d over scored) / sum(w_d over applicable)
overall       = sum(W_g * group_g) / sum(W_g)               over groups with at least one scored dim
coverage      = sum(W_g * coverage_g) / sum(W_g)            over all groups with applicable dims

Missing dimensions are *renormalised away* (they neither help nor hurt), and the loss of
information is reported via ``coverage``. The overall summary abstains when content coverage or
total coverage is too low, and a note is attached when group scores diverge strongly (an overall
number would hide an important difference).

Each dimension's effective weight in the overall summary is reported so that
``sum(contribution) == overall`` exactly.
"""

from __future__ import annotations

from avd.evaluation.profiles import GenreProfile
from avd.evaluation.schema import (
    AGGREGATION_VERSION,
    AggregationResult,
    Contribution,
    Dimension,
    DimensionGroup,
    DimensionScore,
    DimensionStatus,
    GroupAggregate,
)

METHOD = AGGREGATION_VERSION  # v2: blended profiles for genre mixtures, neutral profile when ambiguous
MIN_CONTENT_COVERAGE = 0.5
MIN_TOTAL_COVERAGE = 0.4
DIVERGENCE_THRESHOLD = 1.5


def aggregate(dimensions: list[DimensionScore], profile: GenreProfile) -> AggregationResult:
    by_dim = {d.dimension: d for d in dimensions}
    groups: dict[DimensionGroup, GroupAggregate] = {}
    within_weights: dict[Dimension, float] = {}

    for group in DimensionGroup:
        applicable = [d for d in Dimension if d.group == group and profile.applicable(d)]
        if not applicable:
            continue
        total_w = sum(profile.weight(d) for d in applicable)
        scored = [d for d in applicable if d in by_dim and by_dim[d].status == DimensionStatus.SCORED]
        scored_w = sum(profile.weight(d) for d in scored)
        if scored_w > 0:
            score = sum(profile.weight(d) * float(by_dim[d].score or 0) for d in scored) / scored_w
            conf = sum(profile.weight(d) * float(by_dim[d].confidence or 0) for d in scored) / scored_w
            for d in scored:
                within_weights[d] = profile.weight(d) / scored_w
            groups[group] = GroupAggregate(
                group=group,
                score=round(score, 4),
                confidence=round(conf, 4),
                coverage=round(scored_w / total_w, 4),
                scored_dimensions=scored,
            )
        else:
            groups[group] = GroupAggregate(
                group=group, score=None, confidence=None, coverage=0.0, scored_dimensions=[]
            )

    present = [g for g in groups if groups[g].score is not None]
    all_gw = sum(profile.group_weights.get(g, 0.0) for g in groups)
    coverage = (
        sum(profile.group_weights.get(g, 0.0) * groups[g].coverage for g in groups) / all_gw
        if all_gw > 0
        else 0.0
    )

    notes: list[str] = []
    abstain_reason: str | None = None
    content = groups.get(DimensionGroup.CONTENT)
    if content is None or content.coverage < MIN_CONTENT_COVERAGE:
        abstain_reason = (
            f"content coverage {0.0 if content is None else content.coverage:.2f} "
            f"< {MIN_CONTENT_COVERAGE}: not enough evidence about the content itself"
        )
    elif coverage < MIN_TOTAL_COVERAGE:
        abstain_reason = f"total coverage {coverage:.2f} < {MIN_TOTAL_COVERAGE}"

    overall: float | None = None
    overall_conf: float | None = None
    contributions: list[Contribution] = []
    if abstain_reason is None and present:
        present_gw = sum(profile.group_weights.get(g, 0.0) for g in present)
        overall = 0.0
        conf_acc = 0.0
        for g in present:
            gshare = profile.group_weights.get(g, 0.0) / present_gw
            for d in groups[g].scored_dimensions:
                eff = gshare * within_weights[d]
                s = float(by_dim[d].score or 0)
                overall += eff * s
                conf_acc += eff * float(by_dim[d].confidence or 0)
                contributions.append(
                    Contribution(
                        dimension=d,
                        effective_weight=round(eff, 6),
                        score=s,
                        contribution=round(eff * s, 6),
                    )
                )
        overall = round(overall, 4)
        # Confidence shrinks with missing coverage: an overall built from half the evidence is
        # less trustworthy even if each component is confident.
        overall_conf = round(conf_acc * (0.5 + 0.5 * coverage), 4)
        missing_groups = [g.value for g in groups if groups[g].score is None]
        if missing_groups:
            notes.append(
                "Groups without any scored dimension were excluded and the remaining group "
                f"weights renormalised: {', '.join(missing_groups)}."
            )

    scores = [a.score for a in groups.values() if a.score is not None]
    if len(scores) >= 2 and max(scores) - min(scores) >= DIVERGENCE_THRESHOLD:
        notes.append(
            "Group scores diverge strongly; inspect the group breakdown rather than relying on "
            "the overall summary."
        )

    notes.extend(profile.notes)
    return AggregationResult(
        profile=profile.genre,
        method=METHOD,
        profile_detail=profile.as_dict(),
        groups=groups,
        overall=overall,
        overall_confidence=overall_conf,
        coverage=round(coverage, 4),
        contributions=contributions,
        abstain_reason=abstain_reason,
        notes=notes,
    )
