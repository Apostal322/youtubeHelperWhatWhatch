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
from sqlalchemy.orm import Session, selectinload

from avd.db.models import (
    Channel,
    DimensionScoreRow,
    Evaluation,
    Feedback,
    RecommendationImpression,
    UserPreferences,
    Video,
)
from avd.embeddings.base import Embedder, calibrated_relevance
from avd.evaluation.profiles import get_profile
from avd.evaluation.schema import Dimension
from avd.evaluation.service import genre_or_other, summary
from avd.recsys.index import SearchFilters, vector_search
from avd.recsys.ranker import DEFAULT_WEIGHTS, LinearRanker, Ranker

DEPTH_TARGET = {"intro": 2.5, "intermediate": 3.5, "advanced": 4.5}
POSITIVE = {"recommendation_useful"}
NEGATIVE = {"recommendation_not_useful", "not_interested"}
MAX_PER_CHANNEL = 2

DEFAULT_RANKER = LinearRanker(weights=dict(DEFAULT_WEIGHTS))


@dataclass
class UserProfile:
    user_id: str
    interests: list[str] = field(default_factory=list)
    interest_vectors: list[list[float]] = field(default_factory=list)
    genre_weights: dict[str, float] = field(default_factory=dict)
    dimension_weights: dict[str, float] = field(default_factory=dict)
    depth_preference: str = "any"
    languages: list[str] = field(default_factory=list)
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
        p.languages = list(prefs.languages or [])
        p.max_duration_minutes = prefs.max_duration_minutes
        p.diversity = prefs.diversity
        p.min_quality = prefs.min_quality
        p.small_creator_boost = prefs.small_creator_boost
        p.include_synthetic = prefs.include_synthetic
    if p.interests:
        p.interest_vectors = embedder.embed(p.interests)
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
    profile = get_profile(genre_or_other(ev.genre))
    num = den = 0.0
    for d in ev.dimensions:
        if d.status != "scored" or d.score is None:
            continue
        w = profile.weight(Dimension(d.dimension)) * float(dimension_weights.get(d.dimension, 1.0))
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
) -> dict[str, Any]:
    prof = build_profile(session, user_id, embedder)
    filters = SearchFilters(
        languages=prof.languages or None,
        include_synthetic=prof.include_synthetic,
        max_duration_seconds=prof.max_duration_minutes * 60 if prof.max_duration_minutes else None,
        exclude_ids=prof.excluded or None,
    )
    # ---- retrieval: per-interest vector search + quality-based candidates for preferred genres
    relevance: dict[str, tuple[float, str | None]] = {}
    for interest, vec in zip(prof.interests, prof.interest_vectors, strict=True):
        for vid, sim in vector_search(session, vec, 60, embedder.model_id, filters):
            if vid not in relevance or sim > relevance[vid][0]:
                relevance[vid] = (sim, interest)
    preferred = [g for g, w in prof.genre_weights.items() if w > 0] or None
    q = (
        select(Video.id)
        .join(Evaluation, (Evaluation.video_id == Video.id) & Evaluation.is_current.is_(True))
        .where(Evaluation.status.in_(("complete", "partial")))
        .order_by(Evaluation.overall_score.desc().nulls_last())
        .limit(40)
    )
    if preferred:
        q = q.where(Video.genre.in_(preferred))
    if prof.languages:
        q = q.where(Video.language.in_(prof.languages))
    if not prof.include_synthetic:
        q = q.where(Video.is_synthetic.is_(False))
    for (vid,) in session.execute(q).all():
        if vid not in prof.excluded:
            relevance.setdefault(vid, (0.0, None))

    if not relevance:
        return {
            "request_id": None,
            "ranker_version": ranker.version,
            "items": [],
            "notes": ["no analyzed videos match your filters yet"],
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
    for vid, (sim, matched_interest) in relevance.items():
        v, ev = videos.get(vid), evals.get(vid)
        if v is None or ev is None:
            continue
        pq = personal_quality(ev, prof.dimension_weights)
        if pq is not None and pq < prof.min_quality:
            continue
        if pq is None and prof.min_quality > 0:
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
            "genre_preference": float(prof.genre_weights.get(v.genre or "other", 0.5))
            if prof.genre_weights
            else 0.5,
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
            }
        )

    # Two-stage eligibility: with stated interests, items below the relevance floor are not topical
    # matches; they may only fill a small, explicitly labelled exploration quota.
    if prof.interests:
        topical = [c for c in scored if c["features"]["relevance"] > 0]
        explore = [c for c in scored if c["features"]["relevance"] <= 0]
        quota = min(len(explore), max(1, limit // 5))
        selected = _mmr(topical, limit - quota, prof.diversity)
        for c in explore:
            c["exploration"] = True
        selected += _mmr(explore, min(quota, limit - len(selected)), prof.diversity, already=selected)
    else:
        selected = _mmr(scored, limit, prof.diversity)
    request_id = str(uuid.uuid4())
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
    return {"request_id": request_id, "ranker_version": ranker.version, "items": items, "notes": []}


def _mmr(
    cands: list[dict[str, Any]], k: int, diversity: float, already: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance over embeddings + a per-channel cap. ``already`` holds items
    selected in an earlier pass (they count for diversity and the channel cap)."""
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
        for c in pool:
            if per_channel.get(c["video"].channel_id, 0) >= MAX_PER_CHANNEL:
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
        selected.append(best)
        pool.remove(best)
        per_channel[best["video"].channel_id] = per_channel.get(best["video"].channel_id, 0) + 1
    return selected


def _explain(c: dict[str, Any], prof: UserProfile) -> list[str]:
    out = []
    if c.get("exploration"):
        out.append("Exploration pick: well rated, outside your stated interests.")
    elif c["interest"]:
        out.append(f"Matches your interest “{c['interest']}”.")
    ev: Evaluation = c["evaluation"]
    best = sorted(
        (d for d in ev.dimensions if d.status == "scored" and d.score is not None and d.group != "technical"),
        key=lambda d: -(d.score or 0),
    )[:2]
    if best:
        out.append(
            "Strongest: " + ", ".join(f"{d.dimension.replace('_', ' ')} {d.score:.1f}/5" for d in best) + "."
        )
    if prof.depth_preference in DEPTH_TARGET and c["depth"] is not None:
        out.append(f"Depth {c['depth']:.1f}/5 vs. your '{prof.depth_preference}' preference.")
    if c["features"]["feedback_affinity"] > 0.2:
        out.append("Similar to videos you marked useful.")
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
        "topics": v.topics,
        "published_at": v.published_at.isoformat() if v.published_at else None,
        "is_synthetic": v.is_synthetic,
        "platform": v.platform,
        "url": _video_url(v),
        "status": v.status,
    }
