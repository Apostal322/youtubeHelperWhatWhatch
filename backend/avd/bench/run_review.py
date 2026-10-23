"""What a local-model run actually said, in a form a maintainer can check.

A contributed pack carries counts (validity, citation correctness, abstention rate). Those cannot
show whether a cited quote supports its score, whether a justification speculates, or how the
model abstained. ``review.json`` carries the per-dimension output of every evaluation in a run:
status, score, confidence, reason code, the model's justification, evidence references (segment,
timestamps, verification) and the deterministic support-audit flags.

Rights: quote text and justifications (which may quote the transcript) are included only for
items whose every source permits public quotation (``avd.portability``). Otherwise they are
``null``, and only quote hashes and segment references remain. Transcript text is never included
beyond the quotes. ``avd community pack`` additionally caps how much transcript text a review
file may contain.

``quality_control`` turns the file into counts and flags a reviewer starts from. It separates
four things that are easy to conflate:

1. structural validity: the output parsed and matched the schema;
2. evidence existence: the quote is at the cited place;
3. semantic support: not automated; listed for human review (``avd evidence-review``);
4. accuracy: needs independent human ratings, which this file cannot provide.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

REVIEW_NAME = "review.json"
REVIEW_FORMAT = "avd-run-review"
POPULARITY = re.compile(
    r"\b(views?|view counts?|subscribers?|likes?|popular(?:ity)?|trending|viral|followers?)\b", re.I
)


def build_review(session: Any) -> dict[str, Any]:
    """Every current evaluation in a run database, with justifications where quotation is permitted."""
    from sqlalchemy import select

    from avd.db.models import Evaluation
    from avd.portability import _evaluation_record

    items = []
    for ev in session.scalars(
        select(Evaluation)
        .where(Evaluation.is_current.is_(True))
        .order_by(Evaluation.video_id, Evaluation.stage)
    ):
        record = _evaluation_record(session, ev, include_quotes=True)
        quotable = bool(record["rights"]["quotes_included"])
        items.append(
            {
                "video_id": ev.video_id,
                "evaluation": record,
                "justifications": {
                    d.dimension: (d.justification if quotable else None) for d in ev.dimensions
                },
                "support_audit": (ev.coverage_report or {}).get("support_audit") or {},
            }
        )
    return {
        "format": REVIEW_FORMAT,
        "format_version": "1.0",
        "notice": (
            "Model output for review. Quotes and justifications appear only where the source permits "
            "public quotation (attribution: see each item's content_sources). Semantic support and "
            "accuracy are not established by this file."
        ),
        "items": items,
    }


def write_review(run_dir: Path) -> Path | None:
    """Write ``review.json`` next to a run's ``run.db``. Returns the path, or None without a database."""
    from sqlalchemy.orm import sessionmaker

    from avd.db.session import make_engine

    db = run_dir / "run.db"
    if not db.exists():
        return None
    engine = make_engine(f"sqlite:///{db}")
    try:
        with sessionmaker(bind=engine)() as s:
            doc = build_review(s)
    finally:
        engine.dispose()
    out = run_dir / REVIEW_NAME
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def quality_control(doc: dict[str, Any]) -> dict[str, Any]:
    """Counts and flags for a first look at real output. Nothing here judges accuracy."""
    scored = abstained = claimed = verified = 0
    reasons: Counter[str] = Counter()
    audit: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    popularity: list[str] = []
    unverified: list[str] = []
    for item in doc.get("items", []):
        ev = item["evaluation"]
        sources[ev.get("evaluator_source", "?")] += 1
        for d in ev.get("dimensions", []):
            if d["status"] == "scored":
                scored += 1
            else:
                abstained += 1
                reasons[d.get("reason_code") or d["status"]] += 1
            for e in d.get("evidence", []):
                claimed += 1
                verified += bool(e.get("verified"))
                if not e.get("verified"):
                    unverified.append(f"{item['video_id']}:{d['dimension']}")
            text = (item.get("justifications") or {}).get(d["dimension"]) or ""
            if POPULARITY.search(text):
                popularity.append(f"{item['video_id']}:{d['dimension']}")
        for codes in ((item.get("support_audit") or {}).get("flags") or {}).values():
            audit.update(codes)
    return {
        "items": len(doc.get("items", [])),
        "evaluator_sources": dict(sources),
        "structural": {
            "dimensions_scored": scored,
            "dimensions_abstained": abstained,
            "abstention_reasons": dict(reasons),
        },
        "evidence_existence": {"claimed": claimed, "verified": verified, "unverified_at": unverified},
        "support_audit_flags": dict(audit),
        "mentions_popularity_at": popularity,
        "not_established_here": [
            "whether verified quotes semantically support the scores (human review: avd evidence-review)",
            "whether the scores are accurate (needs independent human ratings)",
        ],
    }


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _safe(text: Any) -> str:
    """Titles and model text are untrusted: no escape sequences reach the reviewer's terminal."""
    return _CONTROL.sub("", str(text))


def render_markdown(doc: dict[str, Any]) -> str:
    qc = quality_control(doc)
    lines = [
        "# Run review",
        "",
        f"{qc['items']} evaluation(s); sources {qc['evaluator_sources']}. "
        f"Scored {qc['structural']['dimensions_scored']}, abstained {qc['structural']['dimensions_abstained']} "
        f"({qc['structural']['abstention_reasons']}). Evidence verified {qc['evidence_existence']['verified']} "
        f"of {qc['evidence_existence']['claimed']}. Support-audit flags: {qc['support_audit_flags'] or 'none'}. "
        f"Popularity mentioned in: {qc['mentions_popularity_at'] or 'none'}.",
        "",
        "Semantic support and accuracy are **not** established by this report.",
    ]
    for item in doc.get("items", []):
        ev = item["evaluation"]
        flags = (item.get("support_audit") or {}).get("flags") or {}
        lines += [
            "",
            f"## {_safe(item['video_id'])}: {_safe(ev['video'].get('title'))}",
            "",
            f"Source {ev['evaluator_source']} · models {', '.join(ev['models']) or '-'} · genre {ev.get('genre')} · "
            f"overall {ev['overall'].get('score')} · quotes {'included' if ev['rights']['quotes_included'] else 'withheld'}",
            "",
            "| dimension | status | score | conf | evidence (verified/cited) | audit flags | justification |",
            "|---|---|---:|---:|---|---|---|",
        ]
        for d in ev["dimensions"]:
            ev_list = d.get("evidence", [])
            refs = ", ".join(
                f"[{e['segment_index']}]{'' if e.get('verified') else '✗'}"
                for e in ev_list
                if e.get("segment_index") is not None
            )
            text = _safe(
                ((item.get("justifications") or {}).get(d["dimension"]) or "")
                .replace("|", "/")
                .replace("\n", " ")
            )
            lines.append(
                f"| {d['dimension']} | {d['status']}{' (' + d['reason_code'] + ')' if d.get('reason_code') else ''} | "
                f"{d.get('score') if d.get('score') is not None else ''} | {d.get('confidence') if d.get('confidence') is not None else ''} | "
                f"{sum(1 for e in ev_list if e.get('verified'))}/{len(ev_list)} {refs} | "
                f"{', '.join(flags.get(d['dimension'], [])) or ''} | {text[:300]} |"
            )
    lines.append("")
    return "\n".join(lines)
