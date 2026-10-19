# Contributing to the research

The software is useful only if its assessments can be checked. This page is for people who want
to run models, collect ratings, reproduce benchmarks or propose changes to prompts and rubrics.
For code contributions see [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Rules that protect the results

1. **Development data only.**
   * Experiments use `real_pilot_v1` **dev** (10 episodes).
   * The `test` split (17 items) is partly exposed; see
     `benchmarks/datasets/real_pilot_v1/exposure.json`.
   * `real_holdout_v2` (12 episodes) is **sealed**. It is used once, for a locked configuration
     (`avd gates lock`, then `avd evaluate-real --confirm-test`).

   Commands that could spend the hold-out refuse to: `bench-local`, `bench-recs-stages`,
   `bench-relevance` and the diagnostics.
2. **Record exposure.** If you look at test-split outputs for any reason, add an entry to
   `exposure.json` with the component, commit, what you saw and what it influenced. A
   contaminated result disguised as independent validation is worse than no result.
3. **Never simulate people.**
   * Ratings come from independent human raters through the annotation UI.
   * LLM-generated or invented labels are never accepted, not even as a placeholder.
4. **Keep real and synthetic apart.** Synthetic fixtures are for software tests. Research
   summaries report real data only, and say which evaluator produced each number.
5. **Agreement between models is not validity.** Only agreement with independent human
   judgment, on data not used for tuning, says anything about quality.

## What you can contribute, and how

| Contribution | How | Where it lands |
|---|---|---|
| A real local-model run | [`volunteer-pilot.md`](volunteer-pilot.md): `avd bench-local`, then `avd community pack` | `benchmarks/community/` (contributor-reported) |
| Quality ratings | run an annotation instance (`docs/annotation.md`): 3 independent raters per item, blind | exported with `avd annotate export`, then `avd bench-real --labels` |
| Relevance judgments | `avd bench-relevance sheet`, protocol in `benchmarks/datasets/relevance_v1/README.md` | `avd bench-relevance --judgments` |
| A new evaluator or model provider | [`extending.md`](extending.md); experimental evaluators cannot replace official scores | PR with tests |
| A prompt or rubric change | register a new version (`avd/evaluation/prompts.py`), never edit one in place; justify it with dev-split evidence | PR; old and new versions stay comparable |

Ratings are personal judgments. Contributors' annotations are exported pseudonymously, and raw
annotations are not published automatically. Publishing an aggregate needs the raters' consent
(`docs/annotation.md`, "Approve consent").

## Reproducing the benchmarks

Every committed report names its commit. Re-run any of them on that commit:

| Report (in `benchmarks/reports/`) | Command | Needs |
|---|---|---|
| `baseline-heuristic-synthetic_v1` | `avd bench` | nothing (CI runs it) |
| `baseline-heuristic-real_pilot_v1-dev` | `avd bench-real --split dev` | pilot cache (`avd dataset import`) |
| `recs-stages-real_pilot_v1-dev` | `avd bench-recs-stages` | pilot cache, MiniLM (`avd embeddings fetch-model`) |
| `recs-diagnostic-real_pilot_v1-dev` | `avd bench-recs-diagnose` | pilot cache, MiniLM |
| `perf-local-linux-4vcpu` | `python scripts/perf/local_perf.py --avd backend/.venv/bin/avd` | pilot cache |
| local-model runs | `avd bench-local --models …` | Ollama and a model |
| release gates | `avd gates status --report <report.json>` | a report |

The end-to-end check of a whole installation, including the extension in Chromium:
`python scripts/e2e/full_local.py --chromium <path>` (needs `pip install playwright` in the
interpreter that runs it).

## Current state of the evidence (2026-10-19)

| Question | Status |
|---|---|
| Does a real local model produce valid, grounded output? | **unknown**: no run yet (model registries unreachable from the development environment) |
| Do human raters agree with each other? | **unknown**: no ratings collected |
| Does the model agree with humans? | **unknown**: needs both of the above |
| Is retrieval topical? | on proxy labels only (show membership); human relevance judgments not collected |
| Is the quality score independent of popularity and production? | yes, by construction and by metamorphic tests (`tests/test_metamorphic.py`) |

Release gates: 1 of 12 passes; 7 are blocked by missing human labels and 4 by missing real
inference (`avd gates status`). They stay blocked until real data exists. They are never
relaxed to pass.
