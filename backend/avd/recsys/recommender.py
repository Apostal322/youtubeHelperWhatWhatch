"""Retrieval -> feature extraction -> ranking -> diversification -> impression logging.

Three quantities are kept separate end-to-end:

1. **Intrinsic quality** — from the current evaluation; identical for every user.
2. **Relevance** — similarity between the user's interests and the video (user-specific).
3. **Rank score** — the ranker's combination of features, including a *personalised* quality view
   (the user's dimension priorities applied on top of the genre profile).

Popularity metrics are neither features nor part of the response.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from avd.datasets.blinding import blinded_video_ids, blinding_note
from avd.db.models import (
    Channel,
    DimensionScoreRow,
    Evaluation,
    Feedback,
    RecommendationImpression,
    RecommendationPolicy,
    UserPreferences,
    Video,
)
from avd.embeddings.base import Embedder, calibrated_relevance
from avd.evaluation.profiles import get_profile
from avd.evaluation.schema import Dimension, expand_genres, genre_preference
from avd.evaluation.service import assessment_state, genre_or_other, summary
from avd.recsys.index import SearchFilters, vector_search
from avd.recsys.policy import (
    EXPLORATION_SHARE,
    MAX_PER_CHANNEL,
    QUALITY_CANDIDATES,
    RETRIEVAL_PER_INTEREST,
    policy_config,
    policy_version,
)
from avd.recsys.ranker import DEFAULT_WEIGHTS, LinearRanker, Ranker

DEPTH_TARGET = {"intro": 2.5, "intermediate": 3.5, "advanced": 4.5}
# Cold start: a stated intent supplies genre priors when the user has not weighted genres.
INTENT_GENRES: dict[str, dict[str, float]] = {
    "learn": {
        g: 1.0
        for g in ("education", "lecture", "science", "howto", "programming", "technology", "documentary")
    },
    "stay_informed": {
        g: 1.0 for g in ("news_commentary", "documentary", "interview_podcast", "science", "technology")
    },
    "be_entertained": {
        g: 1.0 for g in ("entertainment", "comedy", "gaming", "music", "personal_storytelling", "art")
    },
    "deep_dive": {g: 1.0 for g in ("lecture", "documentary", "interview_podcast", "science", "programming")},
}
UNUSABLE_STATES = {"failed", "insufficient_evidence"}
POSITIVE = {"recommendation_useful"}
NEGATIVE = {"recommendation_not_useful", "not_interested"}

DEFAULT_RANKER = LinearRanker(weights=dict(DEFAULT_WEIGHTS))


@dataclass
class UserProfile:
    user_id: str
    interests: list[str] = field(default_factory=list)
    interest_vectors: list[list[float]] = field(default_factory=list)
    genre_weights: dict[str, float] = field(default_factory=dict)
    dimension_weights: dict[str, float] = field(default_factory=dict)
    depth_preference: str = "any"
    intent: str = "any"
    genre_weights_source: str = "none"  # "user" | "intent" | "none"
    languages: list[str] = field(default_factory=list)
    min_duration_minutes: int | None = None
    max_duration_minutes: int | None = None
    diversity: float = 0.3
    min_quality: float = 0.0
    small_creator_boost: float = 0.0
    include_synthetic: bool = True
    liked: list[list[float]] = field(default_factory=list)
    disliked: list[list[float]] = field(default_factory=list)
    excluded: set[str] = field(default_factory=set)
    impressions: dict[str, int] = field(default_factory=dict)


def build_profile(session: Session, user_id: str, embedder: Embedder) -> UserProfile:
    prefs = session.get(UserPreferences, user_id)
    p = UserProfile(user_id=user_id)
    if prefs is not None:
        p.interests = [str(i) for i in prefs.interests or []]
        p.genre_weights = {k: float(v) for k, v in (prefs.genre_weights or {}).items()}
        p.dimension_weights = {k: float(v) for k, v in (prefs.dimension_weights or {}).items()}
        p.depth_preference = prefs.depth_preference
        p.intent = prefs.intent or "any"
        p.languages = list(prefs.languages or [])
        p.min_duration_minutes = prefs.min_duration_minutes
        p.max_duration_minutes = prefs.max_duration_minutes
        p.diversity = prefs.diversity
        p.min_quality = prefs.min_quality
        p.small_creator_boost = prefs.small_creator_boost
        p.include_synthetic = prefs.include_synthetic
    if p.genre_weights:
        p.genre_weights_source = "user"
    elif p.intent in INTENT_GENRES:
        p.genre_weights, p.genre_weights_source = dict(INTENT_GENRES[p.intent]), "intent"
    if p.intent == "deep_dive" and p.depth_preference == "any":
        p.depth_preference = "advanced"
    if p.interests:
        p.interest_vectors = embedder.embed(p.interests)
    p.excluded |= blinded_video_ids(session)  # open blind annotation: never surface these
    for vid, kind in session.execute(
        select(Feedback.video_id, Feedback.kind).where(Feedback.user_id == user_id)
    ).all():
        v = session.get(Video, vid)
        if kind == "not_interested":
            p.excluded.add(vid)
        if v is None or v.embedding is None or v.embedding_model != embedder.model_id:
            continue
        if kind in POSITIVE:
            p.liked.append(v.embedding)
        elif kind in NEGATIVE:
            p.disliked.append(v.embedding)
    p.impressions = dict(
        session.execute(
            select(RecommendationImpression.video_id, func.count())
            .where(RecommendationImpression.user_id == user_id)
            .group_by(RecommendationImpression.video_id)
        ).all()
    )
    return p


def _cos(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / ((np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0))


def personal_quality(ev: Evaluation, dimension_weights: dict[str, float]) -> float | None:
    """User-weighted view of quality: genre-profile weights x user priority multipliers."""
    # The exact (possibly blended) weights used for this evaluation; v1 rows fall back to the genre.
    stored = ((ev.aggregation or {}).get("profile") or {}).get("dimension_weights")
    profile = get_profile(genre_or_other(ev.genre))
    num = den = 0.0
    for d in ev.dimensions:
        if d.status != "scored" or d.score is None:
            continue
        base = float(stored.get(d.dimension, 0.0)) if stored else profile.weight(Dimension(d.dimension))
        w = base * float(dimension_weights.get(d.dimension, 1.0))
        num += w * d.score
        den += w
    if den == 0:
        return ev.overall_score
    return num / den


def intrinsic_value(ev: Evaluation) -> float | None:
    if ev.overall_score is not None:
        return ev.overall_score
    groups = [g.get("score") for g in ev.group_scores.values() if g.get("score") is not None]
    return sum(groups) / len(groups) if groups else None


def _dim(ev: Evaluation, name: str) -> float | None:
    for d in ev.dimensions:
        if d.dimension == name and d.status == "scored":
            return d.score
    return None


def _norm_quality(q: float | None) -> float:
    return 0.0 if q is None else max(0.0, min(1.0, (q - 1.0) / 4.0))


def recommend(
    session: Session,
    user_id: str,
    embedder: Embedder,
    *,
    limit: int = 20,
    ranker: Ranker = DEFAULT_RANKER,
    log_impressions: bool = True,
    trace: dict[str, str] | None = None,
) -> dict[str, Any]:
    """``trace``, when given, receives the fate of every retrieved candidate (``selected``,
    ``selected_exploration``, ``ineligible:<reason>``, ``below_relevance_floor``,
    ``channel_cap``, ``not_selected``). Candidates absent from it were never retrieved."""
    prof = build_profile(session, user_id, embedder)
    config = policy_config(ranker, embedder)
    version = policy_version(config)
    filters = SearchFilters(
        languages=prof.languages or None,
        include_synthetic=prof.include_synthetic,
        max_duration_seconds=prof.max_duration_minutes * 60 if prof.max_duration_minutes else None,
        min_duration_seconds=prof.min_duration_minutes * 60 if prof.min_duration_minutes else None,
        exclude_ids=prof.excluded or None,
    )
    # ---- retrieval: per-interest vector search + quality-based candidates for preferred genres
    relevance: dict[str, tuple[float, str | None]] = {}
    for interest, vec in zip(prof.interests, prof.interest_vectors, strict=True):
        for vid, sim in vector_search(session, vec, RETRIEVAL_PER_INTEREST, embedder.model_id, filters):
            if vid not in relevance or sim > relevance[vid][0]:
                relevance[vid] = (sim, interest)
    preferred = expand_genres([g for g, w in prof.genre_weights.items() if w > 0]) or None
    q = (
        select(Video.id)
        .join(Evaluation, (Evaluation.video_id == Video.id) & Evaluation.is_current.is_(True))
        .where(Evaluation.status.in_(("complete", "partial")))
        .order_by(Evaluation.overall_score.desc().nulls_last())
        .limit(QUALITY_CANDIDATES)
    )
    if preferred:
        q = q.where(Video.genre.in_(preferred))
    if prof.languages:
        q = q.where(Video.language.in_(prof.languages))
    if not prof.include_synthetic:
        q = q.where(Video.is_synthetic.is_(False))
    if prof.max_duration_minutes:
        q = q.where(
            (Video.duration_seconds.is_(None)) | (Video.duration_seconds <= prof.max_duration_minutes * 60)
        )
    if prof.min_duration_minutes:
        q = q.where(
            (Video.duration_seconds.is_(None)) | (Video.duration_seconds >= prof.min_duration_minutes * 60)
        )
    for (vid,) in session.execute(q).all():
        if vid not in prof.excluded:
            relevance.setdefault(vid, (0.0, None))

    hidden = blinding_note(session)
    if not relevance:
        # Say *why* nothing is shown: an empty catalog, items hidden for annotation, or no match.
        any_evaluated = session.scalar(select(Evaluation.id).where(Evaluation.is_current.is_(True)).limit(1))
        if hidden:
            reason, note = "hidden_for_blind_annotation", hidden
        elif any_evaluated is None:
            reason, note = (
                "empty_catalog",
                "The catalog is empty: nothing has been imported and analyzed yet.",
            )
        else:
            reason, note = "no_match", "No analyzed videos match your filters yet."
        return {
            "request_id": None,
            "ranker_version": ranker.version,
            "selection_policy_version": version,
            "items": [],
            "eligibility": {"candidates": 0, "ineligible": {}},
            "empty_reason": reason,
            "notes": [note],
        }

    videos = {v.id: v for v in session.scalars(select(Video).where(Video.id.in_(relevance.keys()))).all()}
    evals = {
        e.video_id: e
        for e in session.scalars(
            select(Evaluation)
            .where(Evaluation.video_id.in_(relevance.keys()), Evaluation.is_current.is_(True))
            .options(selectinload(Evaluation.dimensions).selectinload(DimensionScoreRow.evidence))
        ).all()
    }
    scored: list[dict[str, Any]] = []
    # Eligibility is explicit: every candidate that is dropped is counted under a reason.
    ineligible: dict[str, int] = {}

    def drop(reason: str, vid: str) -> None:
        ineligible[reason] = ineligible.get(reason, 0) + 1
        if trace is not None:
            trace[vid] = f"ineligible:{reason}"

    for vid, (sim, matched_interest) in relevance.items():
        v, ev = videos.get(vid), evals.get(vid)
        if v is None or ev is None:
            drop("no_current_evaluation", vid)
            continue
        state = assessment_state(ev)[0]
        if state in UNUSABLE_STATES:
            drop(f"assessment_{state}", vid)
            continue
        pq = personal_quality(ev, prof.dimension_weights)
        if pq is not None and pq < prof.min_quality:
            drop("below_min_quality", vid)
            continue
        if pq is None and prof.min_quality > 0:
            drop("no_quality_estimate_with_min_quality_set", vid)
            continue
        depth = _dim(ev, "depth")
        target = DEPTH_TARGET.get(prof.depth_preference)
        depth_match = 0.5 if target is None or depth is None else 1.0 - abs(target - depth) / 4.0
        affinity = 0.0
        if v.embedding is not None:
            if prof.liked:
                affinity += max(_cos(v.embedding, e) for e in prof.liked)
            if prof.disliked:
                affinity -= max(_cos(v.embedding, e) for e in prof.disliked)
        ch = session.get(Channel, v.channel_id)
        conf_values = [
            g.get("confidence") for g in ev.group_scores.values() if g.get("confidence") is not None
        ]
        features = {
            "relevance": calibrated_relevance(embedder, sim) if prof.interests else 0.5,
            "quality": _norm_quality(pq),
            "confidence": ev.overall_confidence
            if ev.overall_confidence is not None
            else (sum(conf_values) / len(conf_values) if conf_values else 0.0),
            "genre_preference": genre_preference(prof.genre_weights, v.genre) if prof.genre_weights else 0.5,
            "depth_match": depth_match,
            "novelty": 0.5 ** prof.impressions.get(vid, 0),
            "feedback_affinity": affinity,
            "small_creator": 1.0 if ch is not None and ch.size_bucket in ("micro", "small") else 0.0,
        }
        rank, contrib = ranker.score(features, {"small_creator": prof.small_creator_boost})
        scored.append(
            {
                "video": v,
                "evaluation": ev,
                "similarity": sim,
                "interest": matched_interest,
                "features": features,
                "rank": rank,
                "contrib": contrib,
                "personal_quality": pq,
                "depth": depth,
                "state": state,
            }
        )

    # Two-stage eligibility: with stated interests, items below the relevance floor are not topical
    # matches; they may only fill a small, explicitly labelled exploration quota.
    if prof.interests:
        topical = [c for c in scored if c["features"]["relevance"] > 0]
        explore = [c for c in scored if c["features"]["relevance"] <= 0]
        quota = min(len(explore), max(1, limit // EXPLORATION_SHARE))
        selected = _mmr(topical, limit - quota, prof.diversity)
        for c in explore:
            c["exploration"] = True
        selected += _mmr(explore, min(quota, limit - len(selected)), prof.diversity, already=selected)
    else:
        selected = _mmr(scored, limit, prof.diversity)
    if trace is not None:
        chosen = {id(c) for c in selected}
        per_channel: dict[str, int] = {}
        for c in selected:
            per_channel[c["video"].channel_id] = per_channel.get(c["video"].channel_id, 0) + 1
        for c in scored:
            vid = c["video"].id
            if id(c) in chosen:
                trace[vid] = "selected_exploration" if c.get("exploration") else "selected"
            elif prof.interests and c["features"]["relevance"] <= 0:
                trace[vid] = "below_relevance_floor"
            elif per_channel.get(c["video"].channel_id, 0) >= MAX_PER_CHANNEL:
                trace[vid] = "channel_cap"
            else:
                trace[vid] = "not_selected"
    request_id = str(uuid.uuid4())
    request_context = {
        "limit": limit,
        "diversity": prof.diversity,
        "min_quality": prof.min_quality,
        "depth_preference": prof.depth_preference,
        "small_creator_boost": prof.small_creator_boost,
        "has_interests": bool(prof.interests),
        "genre_weights_source": prof.genre_weights_source,
    }
    if log_impressions and selected:
        _register_policy(session, version, config)
    items = []
    for pos, c in enumerate(selected):
        v, ev = c["video"], c["evaluation"]
        imp_id = None
        if log_impressions:
            imp = RecommendationImpression(
                request_id=request_id,
                user_id=user_id,
                video_id=v.id,
                evaluation_id=ev.id,
                position=pos,
                ranker_version=ranker.version,
                selection_policy_version=version,
                selection_context={
                    **request_context,
                    "exploration": bool(c.get("exploration", False)),
                    "channel_cap_relaxed": bool(c.get("channel_cap_relaxed", False)),
                    "mmr_penalty": c.get("mmr_penalty", 0.0),
                },
                intrinsic_quality=intrinsic_value(ev),
                relevance=c["similarity"],
                rank_score=c["rank"],
                features=c["features"],
            )
            session.add(imp)
            session.flush()
            imp_id = imp.id
        items.append(
            {
                "impression_id": imp_id,
                "position": pos,
                "video": video_card(session, v),
                "intrinsic_quality": summary(ev),
                "relevance": {
                    "similarity": round(c["similarity"], 4),
                    "normalized": round(c["features"]["relevance"], 4),
                    "matched_interest": c["interest"],
                },
                "ranking": {
                    "score": c["rank"],
                    "ranker_version": ranker.version,
                    "selection_policy_version": version,
                    "channel_cap_relaxed": bool(c.get("channel_cap_relaxed", False)),
                    "personal_quality": None
                    if c["personal_quality"] is None
                    else round(c["personal_quality"], 3),
                    "features": {k: round(x, 4) for k, x in c["features"].items()},
                    "contributions": c["contrib"],
                    "diversified": c.get("mmr_penalty", 0.0) > 0,
                    "exploration": bool(c.get("exploration", False)),
                },
                "explanations": _explain(c, prof),
            }
        )
    notes = [hidden] if hidden else []
    empty_reason = None
    if not scored:
        # Every candidate was dropped by a filter. Without a note the popup took this for an
        # empty catalog and suggested importing videos.
        empty_reason = "no_match"
        reasons = ", ".join(f"{k.replace('_', ' ')}: {n}" for k, n in sorted(ineligible.items()))
        notes.append(f"All {len(relevance)} analyzed candidates were filtered out ({reasons}).")
    topical_shown = sum(1 for c in selected if not c.get("exploration"))
    if prof.interests and scored and topical_shown < limit:
        notes.append(
            f"{topical_shown} of {len(scored)} analyzed candidates match your interests closely enough; "
            "the list is not filled up with unrelated videos."
            + (
                f" {len(selected) - topical_shown} exploration pick(s) outside them are labelled as such."
                if len(selected) > topical_shown
                else ""
            )
        )
    stale = session.scalar(
        select(func.count())
        .select_from(Video)
        .where(Video.embedding.is_not(None), Video.embedding_model != embedder.model_id)
    )
    if stale and prof.interests:
        # Their vectors come from another model, so interest matching leaves them out until they are
        # re-embedded; without this note the list just looked thin.
        notes.append(
            f"{stale} analyzed video(s) are indexed with a different embedding model and are not matched "
            "to your interests until they are re-embedded (`avd embeddings reindex`)."
        )
    if prof.genre_weights_source == "intent":
        notes.append(f"Genre priorities come from your stated intent '{prof.intent}' (no genre weights set).")
    return {
        "request_id": request_id,
        "ranker_version": ranker.version,
        "selection_policy_version": version,
        "items": items,
        "eligibility": {"candidates": len(relevance), "eligible": len(scored), "ineligible": ineligible},
        "empty_reason": empty_reason,
        "notes": notes,
    }


def _register_policy(session: Session, version: str, config: dict[str, Any]) -> None:
    """Store a configuration the first time it is used. Existing rows are never modified."""
    if session.get(RecommendationPolicy, version) is not None:
        return
    try:
        with session.begin_nested():
            session.add(RecommendationPolicy(version=version, config=config))
    except IntegrityError:  # another request registered it concurrently
        pass


def _mmr(
    cands: list[dict[str, Any]], k: int, diversity: float, already: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance over embeddings + a soft per-channel cap. ``already`` holds items
    selected in an earlier pass (they count for diversity and the channel cap).

    The cap spreads slots across creators while there is a choice. When every remaining candidate
    comes from a channel already at the cap, the best of them is still taken (and marked
    ``channel_cap_relaxed``): leaving a slot empty, or filling it with something off-topic, would
    hide relevant videos only because one creator covers the topic."""
    lam = 1.0 - max(0.0, min(1.0, diversity))
    pool = sorted(cands, key=lambda c: -c["rank"])
    if not pool or k <= 0:
        return []
    top = pool[0]["rank"] or 1.0
    prior = list(already or [])
    selected: list[dict[str, Any]] = []
    per_channel: dict[str, int] = {}
    for c in prior:
        per_channel[c["video"].channel_id] = per_channel.get(c["video"].channel_id, 0) + 1
    while pool and len(selected) < k:
        best, best_val, best_pen = None, -1e9, 0.0
        capped_only = all(per_channel.get(c["video"].channel_id, 0) >= MAX_PER_CHANNEL for c in pool)
        for c in pool:
            if not capped_only and per_channel.get(c["video"].channel_id, 0) >= MAX_PER_CHANNEL:
                continue
            pen = 0.0
            emb = c["video"].embedding
            others = prior + selected
            if others and emb is not None:
                pen = max(
                    (_cos(emb, s["video"].embedding) for s in others if s["video"].embedding is not None),
                    default=0.0,
                )
            val = lam * (c["rank"] / top) - (1 - lam) * pen
            if val > best_val:
                best, best_val, best_pen = c, val, pen
        if best is None:
            break
        best["mmr_penalty"] = round(best_pen * (1 - lam), 4)
        if capped_only:
            best["channel_cap_relaxed"] = True
        selected.append(best)
        pool.remove(best)
        per_channel[best["video"].channel_id] = per_channel.get(best["video"].channel_id, 0) + 1
    return selected


def _explain(c: dict[str, Any], prof: UserProfile) -> list[str]:
    """Factual reasons only: each line restates data the ranking actually used."""
    out = []
    v: Video = c["video"]
    ev: Evaluation = c["evaluation"]
    if c.get("exploration"):
        out.append("Exploration pick: outside your stated interests, included to broaden results.")
    if c.get("channel_cap_relaxed"):
        out.append("More from a channel already shown: no other matching creator is left for this slot.")
    elif c["interest"] and not c.get("exploration"):  # "nearest interest" is not a match here
        out.append(f"Matches your interest “{c['interest']}”.")
    best = sorted(
        (d for d in ev.dimensions if d.status == "scored" and d.score is not None and d.group != "technical"),
        key=lambda d: -(d.score or 0),
    )[:2]
    if best:
        out.append(
            "Strongest: " + ", ".join(f"{d.dimension.replace('_', ' ')} {d.score:.1f}/5" for d in best) + "."
        )
    genre_note = f"Genre: {(v.genre or 'unknown').replace('_', ' ')}"
    if v.genre_status in ("mixed", "ambiguous", "unknown"):
        genre_note += f" ({v.genre_status})"
    if prof.genre_weights and v.genre:
        w = genre_preference(prof.genre_weights, v.genre, default=0.5)
        src = "your intent" if prof.genre_weights_source == "intent" else "your genre priorities"
        genre_note += f"; weight {w:.1f} from {src}"
    out.append(genre_note + ".")
    if prof.languages and v.language:
        out.append(f"Language: {v.language} (one of your languages).")
    if v.duration_seconds and (prof.min_duration_minutes or prof.max_duration_minutes):
        lo, hi = prof.min_duration_minutes, prof.max_duration_minutes
        rng = f"{lo or 0}–{hi} min" if hi else f"at least {lo} min"
        out.append(f"Duration {v.duration_seconds // 60} min, within your {rng} range.")
    if prof.depth_preference in DEPTH_TARGET and c["depth"] is not None:
        out.append(f"Depth {c['depth']:.1f}/5 vs. your '{prof.depth_preference}' preference.")
    if c["features"]["feedback_affinity"] > 0.2:
        out.append("Similar to videos you marked useful.")
    conf = c["features"]["confidence"]
    state = c.get("state") or assessment_state(ev)[0]
    out.append(
        f"Assessment: {state.replace('_', ' ')}; confidence {conf:.2f} (not calibrated against people yet)."
    )
    if ev.is_synthetic:
        out.append("Synthetic demo item with a heuristic (non-AI) evaluation.")
    elif ev.mode == "heuristic_baseline":
        out.append("Scores come from a heuristic baseline, not an AI or human assessment.")
    return out


def _video_url(v: Video) -> str | None:
    if v.is_synthetic:
        return None
    if v.external_url:
        return v.external_url
    return f"https://www.youtube.com/watch?v={v.id}" if v.platform == "youtube" else None


def video_card(session: Session, v: Video) -> dict[str, Any]:
    ch = session.get(Channel, v.channel_id)
    return {
        "id": v.id,
        "title": v.title,
        "channel_title": ch.title if ch else None,
        "duration_seconds": v.duration_seconds,
        "language": v.language,
        "genre": v.genre,
        "genre_status": v.genre_status,  # confident / mixed / ambiguous / unknown / overridden
        "topics": v.topics,
        "published_at": v.published_at.isoformat() if v.published_at else None,
        "is_synthetic": v.is_synthetic,
        "platform": v.platform,
        "url": _video_url(v),
        "status": v.status,
    }
