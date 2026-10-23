# Release readiness

Two different milestones, which must never be confused:

* **Technical beta**: the software installs, runs, protects its users' data and says honestly
  what its scores are. It can ship as an *experimental* tool with a heuristic or unvalidated
  model evaluator, provided the status is shown everywhere.
* **Scientifically validated assessments**: the evaluator's scores agree with independent human
  judgment on data not used for tuning, within the pre-registered gates. Until then, no score
  may be described as validated, accurate or objective.

## Experimental alpha (status 2026-10-20, `main` after Phase 3.5)

| Area | Requirement | Status |
|---|---|---|
| Install | one-command local start on Linux | ✅ fresh public clone of `main`: install 35 s, dataset import 6 s, `avd start` ready in ~2 s |
| Install | Windows / macOS | ❌ untested (instructions written; Windows hardware detection exists but never ran on Windows) |
| Install | self-hosted Docker Compose | ⚠️ `docker compose config` valid and images built in CI; `up` not run in the dev environment |
| Real AI | at least one real local-model run reported | ❌ **zero real model calls**: no GPU, runtime or weights here, model registries unreachable. Volunteer workflow tested up to the model call (`docs/first-experiment.md`) |
| Extension | local and remote backends, consent, recovery, empty catalog, unreachable model | ✅ 34-step Chromium run (`scripts/e2e/full_local.py`) + 44 component tests |
| Search | an unmatched query returns nothing | ✅ since Phase 3.5 (relevance floor). ⚠️ default `hashing` embedder still lets collisions through; MiniLM recommended |
| Security | protections and regression tests; host validation, CORS, loopback binding | ✅ suite + live checks (Phase 3.5 report) |
| Privacy | no telemetry; export, deletion, backup, restore | ✅ (PostgreSQL restore is manual `pg_restore`) |
| Writes are durable before success is reported | the API commits before it responds | ✅ |
| Data rights | rights review per source; no transcript text in exports | ✅ Talk Python remains restricted |
| CI | lint, types, tests (SQLite + PostgreSQL + Redis), migrations, audits, Docker build | ✅ green on `main` |
| License | `LICENSE` file present | ✅ Apache-2.0 (PR #5, merged with owner approval). Data, transcripts and model weights keep their own licences (`docs/licensing.md`) |
| Status shown to users | heuristic / AI / synthetic labels; "not human-validated" | ✅ |
| Branch protection on `main` | reviews required, status checks required | ❌ **owner setting**: `main` is unprotected |
| Release artefacts | tag, release notes, changelog | ❌ none yet; nothing is published automatically |

**Verdict:** the engineering system qualifies for a clearly labelled **experimental alpha** (Linux,
heuristic baseline by default, local models supported but unmeasured). It does not qualify as a
beta: there is no real model run, Windows and macOS are untested, and Compose `up` has not been
exercised. Publishing a release is the owner's decision. The release notes must say "experimental,
not scientifically validated, zero real model evaluations, zero human ratings".

## Scientific validation

| Gate group (`benchmarks/release_gates.json`) | Status |
|---|---|
| Human reliability (precondition) | blocked: no human labels |
| Validity: Spearman, MAE non-inferiority, pairwise, calibration, selective risk, abstention vs humans | blocked: no human labels |
| Operational: invalid citations, popularity invariance, cost, latency | blocked: no real inference (heuristic numbers do not count) |
| Injection benign-flag rate | passes (does not depend on the evaluator) |

**Verdict:** not validated. The path is fixed in advance:
1. a real model run on the dev split (operational gates);
2. 3 independent raters per item on the dev and test splits (reliability, then validity);
3. lock the configuration (`avd gates lock`);
4. a single confirmatory run on `real_holdout_v2`.
