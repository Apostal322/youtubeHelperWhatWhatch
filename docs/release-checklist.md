# Release readiness

Two different milestones, which must never be confused:

* **Technical beta**: the software installs, runs, protects its users' data and says honestly
  what its scores are. It can ship as an *experimental* tool with a heuristic or unvalidated
  model evaluator, provided the status is shown everywhere.
* **Scientifically validated assessments**: the evaluator's scores agree with independent human
  judgment on data not used for tuning, within the pre-registered gates. Until then, no score
  may be described as validated, accurate or objective.

## Technical beta (status 2026-10-19, commit on the Phase 3.3 branch)

| Area | Requirement | Status |
|---|---|---|
| Install | one-command local start on Linux | ✅ `avd start`, verified end to end |
| Install | Windows / macOS | ❌ untested (instructions written, not verified) |
| Install | self-hosted Docker Compose | ⚠️ config validated and image built in CI; `up` not run in the dev environment |
| Real AI | at least one real local-model run reported | ❌ none: model registries unreachable here; volunteer workflow ready (`docs/volunteer-pilot.md`) |
| Extension | local and remote backends, consent, recovery | ✅ 28-step Chromium run (`scripts/e2e/full_local.py`) + 42 component tests |
| Security | Phase 3.2 protections and regression tests | ✅ in the suite (guard, registration, imports, network guard) |
| Privacy | no telemetry; export, deletion, backup, restore | ✅ (PostgreSQL restore is manual `pg_restore`) |
| Data rights | rights review per source; no transcript text in exports | ✅ Talk Python remains restricted |
| CI | lint, types, tests (SQLite + PostgreSQL + Redis), migrations, audits, local smoke test, Docker build | ✅ |
| Docs | install, self-hosting, privacy, security, export format, contributing | ✅ |
| License | `LICENSE` file present | ❌ **owner decision**: draft PR #5 adds Apache-2.0 |
| Dependencies | license compatibility reviewed | ✅ `docs/licensing.md` (psycopg is LGPL-3.0, used as a library) |
| Status shown to users | heuristic / AI / synthetic labels; "not human-validated" | ✅ |
| Known limitations published | README "Status", phase reports | ✅ |

**Verdict:** not yet a technical beta. The blocking items are the license file (owner) and one
real model run (volunteer hardware). Everything else needed for an experimental beta is in
place.

## Scientific validation

| Gate group (`benchmarks/gates.json`) | Status |
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
