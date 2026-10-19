# Phase 3.2 engineering report: open-source, self-hosted, local-first

Date: 2026-10-19. Branch `claude/phase3-2-self-hosted`, stacked on Phase 3.1
(`claude/phase3-1-local-ollama`, PR #3), which is stacked on Phase 3 (PR #2). Neither earlier
PR is merged; their history is untouched.

## Bottom line

* **No central server is needed.** Nothing in the code calls a project-operated service, and
  nothing ever did. One local-first problem was found: the extension tolerated any origin via
  CORS and reported viewed videos to whichever backend was configured. Both are fixed.
* **Local mode works without PostgreSQL, Redis or Celery.** `avd start` runs SQLite in a
  per-user data directory with an embedded worker. Verified end to end on Linux with real
  processes, from start to restart (below).
* **The self-hosted full stack is preserved and hardened:**
  * Postgres and Redis are no longer published;
  * ports bind to loopback;
  * restart policies and an API healthcheck;
  * a one-container `lite` profile;
  * server mode fails closed without safe settings.

  It was validated with `docker compose config` only: this environment has no Docker daemon.
* **Providers.** Implemented: heuristic, Ollama, OpenAI-compatible and Anthropic. **Live-tested:
  none.** Ollama and model registries are blocked here (Phase 3.1), and no paid API was used.
  All model behaviour was tested against mocks, which say so.
* **Not changed:** scientific safeguards, release gates and score semantics. Infrastructure
  readiness is not scientific validation; the validity gates remain blocked on human labels.

## 0. Audit (P0)

| Item | Finding |
|---|---|
| PRs | #2 (Phase 3 → main) and #3 (Phase 3.1 → #2's branch) open, not merged. #3's second backend CI run failed in `pip-audit` (PyPI connection reset before any test ran); the first run on the same commit passed. The failed job was re-run once |
| Tests at start | 395 backend passed (1 skipped live), 32 extension |
| Central-server assumptions | none in the backend. The extension defaulted to `localhost:8000` (no hardcoded remote). **But:** the CORS regex trusted every browser extension; the badge sent every opened YouTube video ID to the configured backend, remote included; the remote URL field had no validation or consent |
| Database / queue | the SQLite + inline queue path existed (`avd demo`, `avd worker`), but there was no way to serve the API and process jobs together without Celery, and no per-user data directory |
| Authentication | anonymous bearer tokens, open registration, admin header token; no deployment-mode distinction; **no Host or Origin checks** (DNS rebinding and cross-site requests against a local instance) |
| Docker Compose | **PostgreSQL 5432 and Redis 6379 published on all interfaces with default credentials**; API on 0.0.0.0; no restart policies |
| License | Apache-2.0 declared in README and `pyproject.toml`, **no LICENSE file** |
| Content rights | Phase 2–3 controls intact (rights review, blinding, revocation) |

The smallest safe migration path, which this phase followed: keep one codebase and one job
queue; add a deployment mode and a request guard; add an embedded executor for the existing
queue; harden Compose with profiles; add portable exports; make the extension's backend a user
choice with consent.

## 1. Architecture changes

| Change | Why |
|---|---|
| `AVD_DEPLOYMENT_MODE=local\|server` and `DeploymentGuard` (ASGI) | local: loopback peers, Host allowlist (DNS rebinding), Origin check on writes (cross-site requests). Server: Host allowlist plus tokens. Server mode refuses to start without an admin token ≥ 24 chars, allowed hosts, a registration policy, private metrics and a non-default DB password |
| Registration `open\|invite\|closed`; `AVD_REQUIRE_AUTH_FOR_READ` | public or private shared instances |
| `EmbeddedWorker` + `avd start` | one process: SQLite, the same durable queue, pausable, stops between jobs; interrupted jobs are requeued on restart |
| Bounded backlog (`AVD_MAX_QUEUED_JOBS`) and `avd jobs cancel` / admin cancel / pause / resume | CPU-only machines cannot be buried; users can stop work safely |
| `avd doctor` and an extended `/v1/status` (mode, registration, worker, inference reachability) | onboarding and support without secrets |
| `avd.portability`: export format v1, import as `imported_unverified` (migration 0007) | reproducible, rights-aware sharing; never authoritative |
| `avd.datactl`: backup, restore, clear; `GET /v1/users/me/export` | data ownership |
| `AVD_EXPERIMENTAL_EVALUATORS` (`avd.plugins`) | extension without core changes; lowest precedence, recorded in versions and config hash |
| Compose profiles `lite`, default full stack, `local-ai` | one file, no duplicates |
| Extension: `lib/backend.ts`, `Backend.tsx`; loopback host permissions on any port; optional permissions HTTPS-only | user-selected backend, validation, diagnostics, consent, no browsing disclosure |

Scores are unchanged: the evaluation, aggregation and ranking code is shared by every mode. No
scoring logic was added or altered.

## 2. Deployment modes: what actually works

| Mode | Verified how | Result |
|---|---|---|
| `avd start` (Linux, SQLite, embedded worker) | real processes, end-to-end script (section 4) and a CI step | works |
| `avd serve` + `avd worker` (inline queue) | unchanged; unit tests | works |
| Docker Compose full stack / lite / local-ai | `docker compose config` (all profiles), image build in CI | **not run**: no Docker daemon here |
| Server mode | API tests (refusals, invite and closed registration, private reads, host enforcement) | works under TestClient; no real TLS proxy deployment tested |
| Windows / macOS | not available | **untested** (documented commands only) |

## 3. Providers

| Provider | Implemented | Live-tested |
|---|---|---|
| Heuristic baseline | yes | yes (it is deterministic code, not a model) |
| Ollama (local) | yes (Phase 3.1) | **no**: registry blocked |
| OpenAI-compatible | yes | no |
| Anthropic | yes | no (no paid calls made) |

There is no fallback between providers. `/v1/status?check_inference=true` checks local model
availability. It never contacts a hosted API.

## 4. End-to-end validation (executed)

Script: real `avd start` processes, `curl`, the CLI, and Chromium 1194 loading the built
extension (Playwright). No mocks; heuristic evaluator (no model available).

1. `avd start --demo` on a fresh data dir: healthy in **1.9 s**; 20 synthetic videos evaluated
   by the embedded worker.
2. `avd dataset import --offline --evaluate` (licensed pilot, pinned cache): 29 items imported
   and evaluated (`heuristic_baseline`). They are **withheld from the API**, because the
   dataset's blind annotation is open. This is the Phase 3 safeguard working; it was not
   bypassed.
3. Provider reported as `heuristic` (no model configured).
4. Search: 25–45 ms. Recommendations: 10–65 ms; 10 items, all labelled synthetic.
5. `avd evaluations export --include-quotes`: 20 evaluations, quotes included only because every
   source is a project-authored fixture; no transcript text.
6. Ctrl+C stopped cleanly. The restart was healthy in **1.35 s**, with the same 20 evaluated
   videos and the preferences intact (HTTP 200).
7. Import into a second instance: 20 records stored as unverified (`trusted: false`).
8. Extension in Chromium against the instance:
   * the Discover feed rendered with "Heuristic mode (not AI)" and "synthetic demo" labels;
   * the breakdown showed the Synthetic labels and evidence with timestamps;
   * *Preferences → Backend*: "Connected to http://127.0.0.1:8791 (8 ms) · personal instance ·
     framework 2026.10.1", "Evaluator: heuristic baseline (no AI model configured)".

   This run **found a real bug**: with the CORS wildcard removed, a local backend on a
   non-8000 port was unreachable. It was fixed by granting loopback hosts on any port (checked
   again in Chromium).

The mocked demonstrations (Ollama protocol mock, evaluate-real and bench-local through it) are
separate tests and are labelled as such. They are not real AI evaluation.

## 5. Measured performance (4-vCPU Xeon VM, 15.7 GB RAM, no GPU, Linux)

| Measure | Result |
|---|---|
| `avd start` to healthy | 1.3–1.9 s (incl. migrations on first run) |
| API + embedded worker RSS | 113–125 MB |
| Data directory | 5.4 MB (20 synthetic) / 11 MB (20 synthetic + 29 licensed transcripts) |
| Queue throughput (heuristic) | 29 pilot items (58 jobs, Stage A+B) in 4.7 s |
| Search latency | 25–45 ms |
| Recommendation latency | 10–95 ms |
| Export / import | 20 evaluations: 469 ms (141 KiB, incl. git commit lookup) / 26 ms |
| Inference throughput | **not measured**: no model available |

Nothing was measured on other hardware. Recommendations elsewhere are estimates and are marked
as such.

## 6. Security

The full threat model and findings are in [`security.md`](security.md#phase-32-local-first-and-self-hosted-deployment).
Fixed in this phase:

* DNS rebinding and cross-site writes against a local instance (open before);
* local instances serving other machines;
* published Postgres/Redis with default credentials;
* the empty admin token accepted as a credential (latent);
* open registration on servers;
* unsafe server configurations;
* the extension disclosing viewing history to remote backends;
* CORS trust in every extension.

Remaining gaps are listed in that document: local mode trusts all local users, plugins run
in-process, exports are unsigned, there is no built-in TLS and tokens do not expire.

## 7. Open-source readiness

* Added:
  * `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1 by reference),
    `SECURITY.md` and `SUPPORT.md`;
  * issue templates (bug, setup help, feature/research) and a PR template with an integrity
    checklist;
  * `docs/extending.md` with a tested example evaluator;
  * `docs/licensing.md`, `docs/infrastructure-costs.md`, `docs/privacy.md`,
    `docs/self-hosting.md`, `docs/local-setup.md` and `docs/evaluation-export.md`;
  * an architecture section and a roadmap update.
* **License:** the `LICENSE` file is missing; the repository declares Apache-2.0. A comparison
  of MIT, Apache-2.0, GPLv3 and AGPLv3 is in `docs/licensing.md`. **Owner decision needed.**
  Nothing was relicensed.
* Not invented: maintainers, governance and community members. Labels such as
  `good first issue` must be created by the owner.

## 8. Tests

| Suite | Result |
|---|---|
| Backend (SQLite + PostgreSQL/pgvector + Redis) | **447 passed, 1 skipped** (the live Ollama test: skipped, not passed); ruff, ruff format and mypy clean; `alembic upgrade head` + `alembic check` clean on PostgreSQL |
| No hidden network traffic | new session guard in `tests/conftest.py`: an audit hook records every DNS lookup and socket connection, and the session **fails** on anything not loopback or private (proxy variables cleared, so nothing can hide behind a proxy). The full suite passes it. A probe test resolving a public name made the session fail, as intended |
| New backend tests | `test_local_first.py` (guard, modes, registration, private reads, backlog, cancellation, embedded worker, restart recovery, CLI bind checks, data dir, plugins), `test_portability.py` (export shape, no transcript, quote rights, blinding filter, round trip, tampering, quote-hash forgery, oversize, backup/restore/clear, user export) |
| Extension | 41 passed (URL validation, consent gating, invite codes, diagnostics, consent UI) |
| CI | new "local mode smoke test" step: `avd start`, status, rebinding → 400, cross-site POST → 403 |

While the suite ran, the sandbox's egress proxy logged rejected connections to
`mobile.events.data.microsoft.com`. They continued, at the same rhythm, after the tests had
finished and with none of the project's processes running. The network guard above shows that
no test made a non-local connection. The traffic comes from the development container itself,
not from this software.

During this phase a test bug leaked `AVD_LLM_PROVIDER=ollama` into later tests:
`monkeypatch.delenv` restores what it removes. It was found because the suite slowed from about
100 s to about 290 s, and was fixed by restoring the environment exactly.

## 9. Remaining blockers

| Blocker | Needs |
|---|---|
| Real model results (local or hosted) | a machine with Ollama and network access to its registry (or an approved API budget) |
| Validity of any score | human labels (blind annotation protocol ready) |
| LICENSE file, private vulnerability reporting, labels | the repository owner |
| Docker deployment run, Windows/macOS validation | an environment with a Docker daemon and those OSes |
| Merging PRs #2, #3 and this PR | owner approval |

## 10. Next milestone

**The first real local evaluation on volunteer hardware.** Run `avd local doctor`, then
`avd bench-local` on the dev split, then publish the report and an `avd-evaluation-export`
alongside it. Then label the 10-item dev set with two raters, so that the first model–human
agreement numbers exist. Everything needed is in place. Only hardware with network access, and
people to rate, are missing.
