"""Phase 3.3: human relevance benchmark plumbing. Judgments here are TEST FIXTURES written in temp
directories, never committed and never reported as results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from avd.bench import relevance as rv


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_blocked_without_judgments(tmp_path: Path) -> None:
    out = rv.evaluate(tmp_path / "none.jsonl")
    assert out["status"] == "blocked_missing_labels" and out["queries"] == 10 and out["items"] == 10


def test_queries_are_valid_and_items_are_dev_only() -> None:
    qs, items = rv.load_queries(), rv.dev_items()
    assert len({q.query_id for q in qs}) == len(qs) == 10
    assert len(items) == 10 and all(it["split"] == "dev" for it in items.values())


@pytest.mark.parametrize(
    "row,message",
    [
        ({"query_id": "q99", "video_id": "chl_gotime_47", "rater": "r1", "grade": 1}, "unknown query"),
        (
            {"query_id": "q01", "video_id": "not_a_dev_item", "rater": "r1", "grade": 1},
            "not a dev-split item",
        ),
        ({"query_id": "q01", "video_id": "chl_gotime_47", "rater": "r1", "grade": 3}, "Input should be"),
    ],
)
def test_invalid_judgments_are_refused(tmp_path: Path, row: dict[str, object], message: str) -> None:
    p = _write(tmp_path / "j.jsonl", [row])
    with pytest.raises(rv.RelevanceError, match=message):
        rv.load_judgments(p, rv.load_queries(), rv.dev_items())


def test_duplicate_judgments_are_refused(tmp_path: Path) -> None:
    row = {"query_id": "q01", "video_id": "chl_gotime_47", "rater": "r1", "grade": 2}
    p = _write(tmp_path / "j.jsonl", [row, row])
    with pytest.raises(rv.RelevanceError, match="duplicate"):
        rv.load_judgments(p, rv.load_queries(), rv.dev_items())


def test_pooling_agreement_and_graded_metrics(tmp_path: Path) -> None:
    rows = []
    for rater, shift in (("r1", 0), ("r2", 0), ("r3", 1)):
        for vid, g in (("chl_gotime_47", 2), ("chl_gotime_179", 1), ("chl_backstage_4", 0)):
            rows.append(
                {
                    "query_id": "q01",
                    "video_id": vid,
                    "rater": rater,
                    "grade": min(2, g + shift) if vid != "chl_gotime_47" else g,
                }
            )
    judgments = rv.load_judgments(_write(tmp_path / "j.jsonl", rows), rv.load_queries(), rv.dev_items())
    grades, agreement = rv.pooled_grades(judgments)
    assert grades[("q01", "chl_gotime_47")] == 2 and grades[("q01", "chl_gotime_179")] == 1
    assert agreement["raters"] == 3 and agreement["min_raters_per_pair"] == 3
    assert agreement["krippendorff_alpha_ordinal"] is not None
    g = {"a": 2.0, "b": 1.0, "c": 0.0}
    perfect = rv.graded_metrics(["a", "b", "c"], g, k=3)
    assert (
        math.isclose(perfect["ndcg@3"], 1.0)
        and perfect["mrr"] == 1.0
        and math.isclose(perfect["off_topic@3"], 1 / 3)
    )
    worst = rv.graded_metrics(["c", "b", "a"], g, k=3)
    assert worst["ndcg@3"] < perfect["ndcg@3"] and worst["mrr"] == 0.5


def test_sheet_lists_every_pair(tmp_path: Path) -> None:
    n = rv.write_sheet(tmp_path / "s.csv", rv.load_queries(), rv.dev_items())
    assert n == 100 and len((tmp_path / "s.csv").read_text().splitlines()) == 101
