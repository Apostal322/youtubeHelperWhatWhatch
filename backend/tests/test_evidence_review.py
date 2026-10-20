"""Human evidence review: sheets from a local run, verdict validation, summary.

The run uses a scripted judge (no real model), and the verdicts are test inputs; nothing here is
a review result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from avd.bench import evaluate_real as er
from avd.bench import evidence_review as rv
from tests.conftest import ScriptedProvider
from tests.test_evaluate_real import _cfg, _settings, grounded_judge, manifest  # noqa: F401


@pytest.fixture
def run_dir(tmp_path: Path, manifest: tuple[Path, Path]) -> Path:  # noqa: F811
    cfg = _cfg(tmp_path, manifest, budget_usd=5.0)
    er.run(cfg, base_settings=_settings(tmp_path), raw_provider=ScriptedProvider(grounded_judge))
    return tmp_path / "run"


def test_sheet_lists_verified_model_quotes_with_context(run_dir: Path, tmp_path: Path) -> None:
    n = rv.write_sheet(run_dir, tmp_path / "sheet.csv")
    rows = list(csv.DictReader((tmp_path / "sheet.csv").open()))
    assert n == len(rows) > 0
    assert all(r["method"].startswith("llm") and r["verdict"] == "" for r in rows)
    assert all(r["quote"] and r["quote"].lower()[:20] in r["segment_text"].lower() for r in rows)


def test_summary_counts_only_reviewed_rows_and_validates_verdicts(run_dir: Path, tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.csv"
    rv.write_sheet(run_dir, sheet)
    rows = list(csv.DictReader(sheet.open()))
    assert rv.summarize(sheet)["status"] == "not_reviewed"
    for r, v in zip(rows[:4], ("supports", "insufficient", "contradicts", "cannot_judge"), strict=True):
        r.update(verdict=v, reviewer="rev-1")
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rv.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    out = rv.summarize(sheet)
    assert out["reviewed"] == 4 and out["rows"] == len(rows) and out["reviewers"] == 1
    assert out["supporting_quotes_judged"] == 3 and out["support_rate"] == pytest.approx(1 / 3, abs=1e-4)
    rows[0]["verdict"] = "probably_fine"
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rv.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(rv.ReviewError, match="not one of"):
        rv.summarize(sheet)


def test_only_development_runs_are_reviewed(run_dir: Path, tmp_path: Path) -> None:
    plan = json.loads((run_dir / "plan.json").read_text())
    plan["meta"]["split"] = "test"
    (run_dir / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(rv.ReviewError, match="development-split"):
        rv.write_sheet(run_dir, tmp_path / "s.csv")
