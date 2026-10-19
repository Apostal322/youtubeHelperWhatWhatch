"""Human review of model evidence: does a verified quote actually support the score?

``validation.verify_quote`` proves a quote exists at (or next to) the cited segment, and
``support.audit`` flags structural problems. Neither decides whether the quote *supports* the
judgement: no offline entailment model with a measured error rate on this task is available. This
module turns a local ``evaluate-real`` / ``bench-local`` run into a review sheet. A person labels
each cited quote, and the summary reports the rates and how many items were reviewed.

The sheet contains transcript text, so it stays on the reviewer's machine. Share only the
summary. Only development-split runs can be reviewed.

Verdicts:

* ``supports``: the quote is evidence for the score given on this dimension;
* ``partially_supports``: relevant, but weaker or narrower than the score suggests;
* ``insufficient``: true but too thin to justify the score;
* ``contradicts``: points the other way;
* ``wrong_segment``: the quote does not mean what the model took it to mean in context;
* ``needs_outside_material``: the score relies on things the transcript does not contain (the
  video, domain facts not stated);
* ``cannot_judge``: the reviewer cannot tell.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

VERDICTS = (
    "supports",
    "partially_supports",
    "insufficient",
    "contradicts",
    "wrong_segment",
    "needs_outside_material",
    "cannot_judge",
)
COLUMNS = (
    "video_id",
    "dimension",
    "score",
    "method",
    "model",
    "evidence_kind",
    "segment_index",
    "start_seconds",
    "quote",
    "segment_text",
    "justification",
    "verdict",
    "reviewer",
    "note",
)


class ReviewError(ValueError):
    pass


def write_sheet(run_dir: Path, out: Path) -> int:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload, sessionmaker

    from avd.content.store import load_bundle
    from avd.db.models import DimensionScoreRow, Evaluation
    from avd.db.session import make_engine

    plan = json.loads((run_dir / "plan.json").read_text())
    if plan.get("meta", {}).get("split") != "dev":
        raise ReviewError("only development-split runs are reviewed (test and held-out items stay unseen)")
    db = run_dir / "run.db"
    if not db.exists():
        raise ReviewError(f"{db} not found: review needs the local run database")
    engine = make_engine(f"sqlite:///{db}")
    sf = sessionmaker(bind=engine)
    n = 0
    with sf() as s, out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        evals = s.scalars(
            select(Evaluation)
            .where(Evaluation.is_current.is_(True))
            .options(selectinload(Evaluation.dimensions).selectinload(DimensionScoreRow.evidence))
        ).all()
        for ev in sorted(evals, key=lambda e: e.video_id):
            transcript = load_bundle(s, ev.video_id).transcript
            for d in sorted(ev.dimensions, key=lambda d: d.dimension):
                if d.status != "scored" or not d.method.startswith("llm"):
                    continue  # heuristic scores cite nothing to review
                for e in d.evidence:
                    if not e.verified or e.segment_index is None:
                        continue  # unverified quotes are already counted as invalid citations
                    seg = transcript.segments[e.segment_index].text if transcript else ""
                    w.writerow(
                        {
                            "video_id": ev.video_id,
                            "dimension": d.dimension,
                            "score": d.score,
                            "method": d.method,
                            "model": d.model,
                            "evidence_kind": e.kind,
                            "segment_index": e.segment_index,
                            "start_seconds": e.start_seconds,
                            "quote": e.quote,
                            "segment_text": seg,
                            "justification": d.justification,
                            "verdict": "",
                            "reviewer": "",
                            "note": "",
                        }
                    )
                    n += 1
    engine.dispose()
    return n


def summarize(sheet: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    if rows and set(COLUMNS) - set(rows[0]):
        raise ReviewError(f"not a review sheet: missing columns {sorted(set(COLUMNS) - set(rows[0]))}")
    reviewed = [r for r in rows if (r.get("verdict") or "").strip()]
    for i, r in enumerate(reviewed, 2):
        if r["verdict"].strip() not in VERDICTS:
            raise ReviewError(f"row {i}: verdict {r['verdict']!r} is not one of {', '.join(VERDICTS)}")
        if not (r.get("reviewer") or "").strip():
            raise ReviewError(f"row {i}: reviewer pseudonym missing")
    counts = Counter(r["verdict"].strip() for r in reviewed)
    supporting = [
        r for r in reviewed if r["evidence_kind"] == "supporting" and r["verdict"] != "cannot_judge"
    ]
    ok = sum(1 for r in supporting if r["verdict"] in ("supports", "partially_supports"))
    return {
        "status": "human_reviewed_sample" if reviewed else "not_reviewed",
        "rows": len(rows),
        "reviewed": len(reviewed),
        "reviewers": len({r["reviewer"].strip() for r in reviewed}),
        "verdicts": dict(counts),
        "supporting_quotes_judged": len(supporting),
        "support_rate": round(ok / len(supporting), 4) if supporting else None,
        "models": sorted({r["model"] for r in reviewed if r.get("model")}),
        "note": "A reviewed sample of one run, judged by the listed number of people; not a validated "
        "error rate. Unreviewed rows are excluded, not assumed correct.",
    }


def add_parser(sub: Any) -> None:
    p = sub.add_parser("evidence-review", help="human review of whether model quotes support the scores")
    p.add_argument("action", choices=["sheet", "summarize"])
    p.add_argument("path", help="run directory (sheet) or filled-in sheet (summarize)")
    p.add_argument("--out", default="evidence_review.csv")
    p.set_defaults(fn=main)


def main(args: argparse.Namespace) -> None:
    try:
        if args.action == "sheet":
            n = write_sheet(Path(args.path), Path(args.out))
            print(
                json.dumps(
                    {"sheet": args.out, "rows": n, "warning": "contains transcript text: do not share"}
                )
            )
        else:
            print(json.dumps(summarize(Path(args.path)), indent=2))
    except ReviewError as exc:
        raise SystemExit(str(exc)) from exc
