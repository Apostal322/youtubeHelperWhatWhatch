# Phase 3.5: stable main, contributor onboarding, usability pilot, honest search

Date 2026-10-20. Branch `claude/eager-galileo-cn31qb`, started from `main` at `0a725f7` (the
merge of PRs #5, #2, #3, #4, #6 and #7, in that order).

**Headline: ZERO REAL MODEL CALLS. Zero human ratings.** The engineering system improved and was
verified. The central research question (do the model's scores agree with people?) is exactly as
open as before.

## 1. Main after the merge stack

| Check | Result |
|---|---|
| History | 6 merge commits on `main`, in the approved order; `main` equals the reviewed #7 head plus `LICENSE` |
| CI on `main` | green on every merge commit, including `0a725f7` |
| Migrations | linear 0001 → 0009, single head; `alembic check` clean on SQLite and PostgreSQL, full downgrade and re-upgrade on PostgreSQL |
| Backend suite on `main` | 583 passed, 1 skipped (live Ollama) with PostgreSQL + pgvector + Redis: identical to the pre-merge baseline |
| Extension | 43 passed, `npm audit` 0, build OK |
| Browser E2E on `main` | full local scenario 34/34, offline rating packet 14/14 |
| Dependencies | `pip-audit` on the backend environment (76 packages): no known vulnerabilities. The 32 it reports by default are in the container's system Python, not the project |
| Docker Compose | `docker compose config` valid |
| Repository | public; licence detected as Apache-2.0; no open issues; no releases; **`main` is not protected** (owner setting) |
| Docs vs code | every `avd …` invocation in README, CONTRIBUTING and `docs/*.md` (about 230 lines) matches the real argparse tree; every documented `AVD_*` variable is a settings field or a Compose variable |

Nothing was broken by the merge. Fixes in this phase are improvements, not regressions.

## 2. Inference resources

4 vCPU, 15.7 GB RAM, no GPU, no swap, 25 GB free disk. No inference runtime, no model weights,
nothing listening on the Ollama or OpenAI-compatible ports. The Ollama and Hugging Face
registries were found blocked in Phase 3.4 and were not probed again. **Real inference is
externally blocked; the milestone is not complete.** No mock or heuristic output is reported
as a model result.

## 3. Contributor workflow, from a fresh public clone of `main`

| Step | Command | Result (Linux, measured) |
|---|---|---|
| Clone + install | `git clone …`, `python3 -m venv .venv`, `pip install -e ".[dev]"` | 35 s |
| Database + dataset | `avd dataset import ../benchmarks/datasets/real_pilot_v1/manifest.json` | 29 items, SHA-256 checked, 6 s; schema created automatically |
| Item | `chl_practicalai_178` (Practical AI #178) | dev split; Changelog, CC BY-SA 4.0; every external use confirmed (`avd dataset rights`) |
| Hardware | `avd local doctor` | recommends `qwen3.5:4b`; downloads nothing |
| Model run | `avd community smoke --model qwen3.5:4b --out smoke-run` | **not possible here**; exercised against the loopback protocol mock (plumbing only) |
| Inspect | `avd community review smoke-run` | per-dimension table |
| Pack + verify | `avd community pack smoke-run --out smoke.zip`, `avd community verify smoke.zip` | the mock run is **refused** |

Found and fixed:

* **Verification label.** `verify`'s "real inference (not a mock)" check passed for an HTTP mock
  (an HTTP mock is a real transport). The run was still refused, by the runtime-version check.
  The check itself now fails when the runtime reports itself as a mock (new test, which fails
  on the old code).
* **Accidental commits of run data.** After a smoke run, `backend/smoke-run/` (whose `run.db`
  holds transcript text) and `smoke.zip` showed as untracked, ready to be `git add`-ed into a
  pull request. They, and `benchmarks/local/`, are now ignored.
* **Noisy first run.** Every HTTP request (29 transcript downloads, each model call) printed a
  JSON log line. The CLI now keeps httpx/httpcore at WARNING.
* **Stale branch.** The guides told volunteers to check out the merged `claude/phase3-4-evidence`.
  They now use `main` (or a commit or tag named in an issue). The run records its commit and
  refuses a dirty checkout.
* The packet E2E script failed with a relative `--avd` path (its own usage line).

Windows and macOS remain **untested**.

## 4. Demonstration and user-facing defects

[`docs/demo.md`](demo.md) walks through `avd start --demo`: submitting a self-written
transcript, job progress, the dimension breakdown, evidence, uncertainty, search and
recommendations. Every output is labelled synthetic or heuristic.

Walking through it found three defects, all fixed:

| Defect | Before | After |
|---|---|---|
| Search had no relevance floor | every query returned the whole catalog: "quantum chromodynamics" and "xyzzy plugh" → 20 of 20 videos | hits below the embedder's calibrated floor (as the recommender already used) without a title match are dropped; unmatched queries return `[]`; the E2E search step returns 2 instead of 20 |
| Exploration picks contradicted themselves | "Exploration pick: outside your stated interests" **and** "Matches your interest 'chemistry'" (relevance 0.0) | exploration picks no longer claim a match |
| Untitled videos | a submitted transcript without metadata rendered as a blank, unlabelled title button in the extension | "Untitled video (<id>)" |

Measured on the local instance (4 vCPU, heuristic baseline, 20 videos; 30 requests each):

| Request | p50 | p95 |
|---|---|---|
| Search | 17.7 ms | 24.0 ms |
| Recommendations (10) | 56.9 ms | 99.0 ms |
| Full evaluation | 7.7 ms | 10.3 ms |

Startup was ready in 1.9 s, with an RSS of 112 MB including the embedded worker. The local rate
limiter answered 429 after about 90 requests per minute from one client, which is expected. No
LLM throughput exists to report.

**Known limit, documented, not fixed:** the default `hashing` embedder's collisions still rank
unrelated items. For "bread yeast", a synthetic Spanish video on study technique scored 0.223
against 0.214 for the bread transcript. MiniLM separates them (0.54 against 0.05). A
good-first-issue proposes using MiniLM when its model is present.

## 5. Model unavailable, security

* **Model unavailable.** Started with `--provider ollama` against a port with nothing listening:
  * `avd start` prints `NOT READY (Ollama unreachable …)`;
  * the job retries with backoff and returns to the queue: it is never silently evaluated by the
    heuristic;
  * `/v1/status?check_inference=true` reports `reachable: false`, and the extension's
    diagnostics show "model server unreachable".
* **Live security checks on a local instance:**
  * a forged `Host` header gets 400 (DNS rebinding);
  * a web origin gets no `Access-Control-Allow-Origin`, and its preflight gets 400;
  * admin endpoints return 403 without a token;
  * the server binds `127.0.0.1` by default;
  * `nosniff`, `no-referrer` and `DENY` headers are present.

  **No new finding.** The suite's regression tests for archives, imports, telemetry and the
  network guard pass.

## 6. Human annotation

* **Consent.** `docs/templates/rating-consent.txt` remains a **draft** awaiting owner approval;
  packet issuing refuses it.
  * It covers purpose, tasks, time, data recorded, pseudonymity, storage, sharing and
    publication, voluntariness, withdrawal and its limits, and contact.
  * It now also covers the usability trial.
  * Approval has not been given. Approval of software changes or merges is not approval of
    consent terms.
* **Usability pilot (new):** `avd annotate usability-packet` / `usability-report`.
  * **What it is:** the first ~1,500 words (whole segments) of the shortest calibration
    episode, about 15–20 minutes. Every screen labels it "Usability pilot (excerpt, not a
    rating)".
  * **Nothing stored:** issuing stores nothing, `import-packet` refuses the return, and the
    report stores nothing.
  * **What the report shows:** reported time, answered dimensions, statuses, citations found in
    the excerpt, flags and notes.
  * **Separate from labels:** full-episode annotation remains the separate scientific task.
  * **Verified:** 5 unit tests, plus 5 new Chromium steps on the real pilot (`chl_friends_37`,
    1,525 of 10,769 words). Packet E2E 19/19.
* **Private round-1 files.** The coordinator database and the three packets from Phase 3.4 are
  not in this workspace's repository. They were not recreated, and they embed the superseded
  consent text.

## 7. Recommendation evaluation

No human relevance judgments exist, so no retrieval or recommendation quality is measured. The
proxy (show membership) is not evidence.

**New:** queries carry a `source`. The existing 10 are `catalog_aware_author`, since the
audit found 8 of 10 written from the shows' topics. Needs written by volunteers before they
see the catalog are `blind_rater`, collected with
[`templates/query-elicitation.txt`](templates/query-elicitation.txt). `bench-relevance`
reports `results_by_query_source` next to the pooled results. No threshold was tuned.

## 8. Tests

| Suite | Count |
|---|---|
| Backend, PostgreSQL + pgvector + Redis + SQLite | **597 passed**, 1 skipped (live Ollama: not a pass) |
| Backend, SQLite only | 510 passed, 88 skipped (PostgreSQL variants and live) |
| Extension (vitest) | 44 passed |
| Browser E2E, full local scenario | 34/34 |
| Browser E2E, offline rating packet + usability pilot | 19/19 |
| Lint, format, mypy (114 files) | clean |
| Migrations | single head; `alembic check` clean on SQLite and PostgreSQL |

New tests: mock detection in verify; search with no match; exploration explanations;
`displayTitle`; 5 usability-pilot tests; relevance by query source. Each new behaviour test
was checked to fail on the old code where applicable.

## 9. Release readiness

[`release-checklist.md`](release-checklist.md): the engineering system qualifies for a clearly
labelled **experimental alpha** on Linux. It is not a beta: there is no real model run, Windows
and macOS are untested, and Compose `up` was not exercised. Research readiness:

| Gate group | Status |
|---|---|
| Release gates | 12 gates: 1 passes, 7 blocked by missing human labels, 4 by no real inference (`avd gates status`) |
| Genuine model inference | none |
| Evidence validity on real output | unmeasured |
| Human labels, model–human agreement | none |
| Independent test performance | not run (sealed hold-out untouched) |
| Benchmark reproducibility | the tooling is reproducible; there are no real results to reproduce |

Nothing was released, merged or published.

## 10. Owner decisions and the next action

1. **Next concrete action (highest value):** one volunteer with a 16 GB machine runs
   `docs/first-experiment.md` Part A on `main` and attaches `smoke.zip`. That produces the
   project's first real model calls.
2. Approve or edit the consent text, then send one usability-pilot packet before any full
   calibration packet.
3. Approve the recruitment drafts (`docs/recruitment/`: model volunteers, rating volunteers,
   contributors). Nothing has been posted.
4. Consider protecting `main` (required reviews and status checks).
5. Review and merge this PR (not merged automatically).
