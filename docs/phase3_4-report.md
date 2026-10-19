# Phase 3.4 report: first real evidence (still blocked), human validation workflow, reliability fixes

Date: 2026-10-20. Branch `claude/phase3-4-evidence`, stacked on Phase 3.3
(`claude/phase3-3-validation`, PR #6). The stack below it (#4, #3, #2) is unchanged and unmerged.

## Bottom line

* **Real model inference: still blocked; 0 real model calls.**
  * `ollama.com`, `registry.ollama.ai` and `huggingface.co` were checked once each, and all are
    refused by the egress proxy (HTTP status 000).
  * There is no GPU, no Ollama binary and no model weight anywhere on the machine.
  * Nothing was run, simulated or relabelled as real.
  * The volunteer route is now practical: a Windows-first kit, a one-transcript smoke test, and a
    package that is checked by content before it is shared.
* **Human evidence: 0 ratings; the workflow is complete and verified.** Volunteers can now rate
  with no server at all: one offline HTML file per rater, returned as a ratings file. It was
  verified in headless Chromium against the real pilot transcripts. The same packets collect
  query relevance judgments. Data collection is pending: no raters have been recruited.
* **Two real defects found and fixed:**
  * **Writes were committed after the response was sent.** This was the root cause of the
    Phase 3.3 intermittent deletion failure. A client could be told "deleted" and still read the
    account, and a failed commit was reported as success. Reproduced on the old code (1 stale
    read in 300 under contention; deterministic tests fail), fixed for every endpoint.
  * **ONNX Runtime telemetry.** The MiniLM embedder's runtime connected to
    `mobile.events.data.microsoft.com` and kept a device identifier. Found through the proxy log
    and `strace`, now turned off, with a regression test.
* **Recommendation policy is versioned.** Every impression records which selection
  configuration produced it. Historical rows stay `NULL`.
* **Release gates unchanged: 1 of 12 passes.** 7 are blocked by missing human labels and 4 by
  missing real inference. No gate was weakened.

## 1. Repository and baseline (P0)

| | |
|---|---|
| Open PRs | #2 (Phase 3 → main), #3 (3.1 → #2), #4 (3.2 → #3), #6 (3.3 → #4), #5 (draft: Apache-2.0 `LICENSE` → main). None merged; the stack is preserved |
| PR #1 | GitHub lists it **closed, not merged**, but its head (247232b) is in `main` through the merge commit 4e1c887. The content is merged; the PR record says otherwise |
| CI | green on #4 and #6 heads before this phase |

**Phase 3.3 baseline, reproduced** (PostgreSQL 16 + pgvector + Redis + SQLite):

| | Reported | Measured |
|---|---|---|
| Backend tests | 473 passed, 1 skipped | **473 passed, 1 skipped** (the skip is the live-Ollama test) |
| Extension tests | 42 | **42 passed** |
| Chromium E2E | 28/28 | **28/28 PASS** |
| Real model calls | 0 | 0 |
| Release gates | 1 of 12 | pass 1, blocked_missing_labels 7, blocked_no_inference 4 |
| Human annotations | 0 | 0 |

The first run hit 52 errors because the container had restarted and PostgreSQL and Redis were
down. The numbers above come from the rerun after restarting them.

**Documentation versus implementation, inconsistencies found:**

| Inconsistency | Status |
|---|---|
| The relevance README promised results "for the full recommender", but the code ranked by similarity only | fixed: the final recommendation list is now evaluated |
| One query with no relevant item made every relevance mean NaN | fixed: such queries are counted and kept out of the nDCG and Recall means |
| The volunteer guide said nothing personal was shared beyond CPU and OS, but reports held absolute paths (`/home/<user>/…`), the disk path and the Ollama URL | fixed: redacted, then checked by content |
| "No telemetry" was claimed while ONNX Runtime sent telemetry | fixed (§6) |
| The volunteer guide said "about an hour", which was never measured | removed; the smoke test measures it |
| Extension evidence links were built as `page&t=…`, which breaks podcast episode pages | fixed: deep links only for YouTube |
| A new user who imported the licensed pilot saw an empty feed that said "no analyzed videos match your filters". The real reason: dataset import turns on blinding for annotation | fixed: the reason is stated, with the command to lift it (§7) |

## 2. Model inference (P3): blocked

| Resource | Finding |
|---|---|
| Ollama, llama.cpp, vLLM, LM Studio | none installed |
| Model weights (`*.gguf`, `*.safetensors` > 50 MB) | none on disk |
| GPU | none (no `nvidia-smi`, no device nodes) |
| CPU / RAM / disk | 4 vCPU Xeon (AVX-512, AMX), 15 GB RAM, 26 GB free |
| Registries | ollama.com, registry.ollama.ai, huggingface.co: connection refused by the proxy, checked once each |

**Real model calls: 0.** There is no model digest, latency, output or failure rate to report.
The `community smoke` command, run here, refuses before doing anything: "Ollama unreachable … No
local inference was run." The live test stays *skipped*, not passed.

`avd local doctor` estimates for volunteer hardware (estimates from parameter count and
quantization, **not measurements**):

| Model | Download | Memory | Fits 16 GB, CPU only |
|---|---|---|---|
| `qwen3.5:4b` (Q4_K_M), the starter | ~2.4 GB | ~3.5 GB | yes |
| `qwen3.5:9b` (Q4_K_M) | ~5.5 GB | ~6.8 GB | yes |
| `qwen3.5:27b` | ~16 GB | ~19 GB | no |

## 3. Selection-policy versioning (P1)

**Problem confirmed.** Impressions recorded only `ranker_version`, which was `linear_v1`
throughout. The Phase 3.3 soft channel cap, and any change to retrieval sizes, the relevance
floor, the embedder or its document recipe, left no trace in the logs.

**Fix:**

* `avd/recsys/policy.py` builds the policy configuration, and its version is a digest of that
  configuration (`two_stage_mmr_r2:<12 hex>`). The configuration covers:
  * the algorithm and its revision;
  * retrieval sizes;
  * the embedder and document recipe;
  * the relevance floor and ceiling;
  * the ranker version, features and a digest of its weights (a ranker that reuses a version
    name with other weights still changes the digest);
  * the channel-cap mode and limit;
  * diversification;
  * the exploration strategy and quota.
* Migration 0008 adds:
  * `recommendation_policies`, holding each configuration once and never updated;
  * `recommendation_impressions.selection_policy_version`;
  * `selection_context`, holding the request's diversity, limit, minimum quality, depth and
    small-creator boost, and the per-item exploration, cap-relaxed and MMR-penalty flags.
* **Historical impressions keep `NULL`**, tested through a real 0007 → head upgrade. Their
  policy is unknown and is not guessed.
* API responses carry `selection_policy_version`. Learning-to-rank pairs carry it too, so pairs
  from different candidate sets can be kept apart.
* `tests/test_selection_policy.py` pins the current digest. It checks that each of the 6
  policy constants, the ranker weights, the embedder floor and the document recipe changes the
  version, and that two requests store one policy row.

## 4. The intermittent deletion failure (P1): root cause found

**Cause.** The request session commits in a FastAPI `yield` dependency. With FastAPI 0.143's
default, a dependency's exit code runs **after the response is sent**: measured with a real
uvicorn server, the response arrived at 0.00–0.04 s while a 1.5 s teardown was still running.
So:

1. `DELETE /v1/users/me` answered 200 before the deletion was committed. The E2E's next request
   (step 19) could still authenticate with the old token. That is exactly the Phase 3.3 symptom.
2. When the commit failed (SQLite `database is locked` under write contention), the client had
   already been told it succeeded. The deletion silently did not happen.

The same applied to every write endpoint (41 uses of the dependency).

**Evidence:**

* `tests/test_commit_before_response.py` runs a real server. With the old code, *the response
  was sent before the deletion was committed*, and a failing commit was answered 200 (expected
  500). Both tests fail on the old code and pass on the new code; 25 of 25 repeated runs passed.
* A stress test (300 × create, delete, read, with a second thread writing to the same SQLite
  database):

  | | Stale reads after a successful delete | Failures reported to the client |
  |---|---|---|
  | Old (Phase 3.3) | 1 | errors hidden behind 200 responses |
  | New | 0 | 11 reported as errors; none hidden |

**Fix.** Every session dependency is `Depends(db, scope="function")`, which commits before the
response. A static test fails if a plain `Depends(db)` is added again. The deletion now logs only
counts (no id, token or content).

**Residual.** Under heavy write contention, SQLite still refuses some transactions that upgrade
from read to write (`database is locked`); the busy timeout cannot help those. They are now
reported as errors, so the client can retry. Making every transaction `BEGIN IMMEDIATE` would
serialise reads behind long worker transactions, so it was not done without measurements.

## 5. Volunteer benchmark kit (P2)

**Audit of `pack`, `verify` and `import`:**

* Files were filtered by name only. Their content was not checked.
* Absolute paths leaked: `run_dirs`, `manifest`, `hardware.disk_path`.
* `base_url` could be a LAN address.
* Symlinks were followed.
* There was no bound on total decompressed size.
* Verify did not check prompt and rubric versions, the runtime version, quantization,
  durations, structured-output validity, actual calls, or whether the commit exists.

**Now:**

| Stage | Behaviour |
|---|---|
| `community inspect` (new, dry run) | Lists every file that would be shared (with its redaction count) and every file that stays, with the reason: run databases contain transcript text, and per-item evaluations contain quotes. Lists blocking findings. Writes nothing |
| `community pack` | Redacts absolute paths and non-loopback base URLs, then **refuses** on any of: transcript text; a non-empty `quote`, `transcript` or `excerpt` field; credentials, API keys or tokens; e-mail addresses; non-loopback addresses; symlinks; items whose source has unconfirmed uses; claimed human labels or trust status. Transcript text is detected by comparing every 8-word run of every cached pilot transcript |
| `community verify` | All earlier checks, plus: commit exists in the repository; prompt and rubric versions recorded and known; quantization; runtime version (a mock is refused); model calls > 0; structured-output validity present; durations > 0; the content scan above; rights. Zip hardening: no symlink entries, duplicates or unlisted files, at most 500 entries and 50 MB expanded |
| `community import` | Status is always `contributor_reported`, with `independently_reproduced: false` and `human_labels: 0`. Never overwrites; the destination cannot escape the directory |
| `community smoke` (new) | One real evaluation of the shortest shareable dev transcript, then the package dry run. Prints model fingerprint, runtime, calls, validity, citation correctness, wall time, tokens/s and memory |

`docs/volunteer-pilot.md` is rewritten Windows-first, covering:

* hardware;
* the starter model;
* Ollama kept on loopback;
* checkout and dependencies (SQLite mode, no PostgreSQL or Redis);
* authorized material and its rights;
* the model check;
* the smoke test, the benchmark, verification and export;
* submission through a new issue form;
* troubleshooting.

**Only Linux was tested, never with a real model.** The guide says so.

**Rights.** The volunteer benchmark uses the 10 dev episodes, all from Changelog (CC BY-SA 4.0,
every use confirmed). Talk Python items (licence scope unconfirmed) are test-split only and are
refused by both `pack` and `verify`. Transcripts never leave the machine in any case.

Tests: `tests/test_community.py`, 33 tests. They include 8 forbidden-content cases, symlinks
(on disk and in zip entries), unlisted files, an oversized expansion, Talk Python items, an
unknown commit, an unpacked report with paths, and the smoke path.

## 6. Security (P19)

| Area | Result |
|---|---|
| Commit before response | **fixed** (§4) |
| **Telemetry** | **fixed**. ONNX Runtime 1.31 (the MiniLM runtime) opened two connections per run to `mobile.events.data.microsoft.com` (logged by the egress proxy, traced with `strace` to its native threads) and kept `~/.cache/Microsoft/DeveloperTools/.onnxruntime/{deviceid,onnxruntime.db}`. The embedder now sets `ORT_DISABLE_TELEMETRY=1` before import and calls `disable_telemetry_events()`. After the fix, `strace` showed 0 network connects and the store was untouched. `tests/test_no_telemetry.py` (child process with its own `HOME` and a listening proxy) fails on the old code and passes now |
| Other dependencies | no other telemetry client found in the Python or extension dependencies (`opentelemetry-api` is the API without an exporter) |
| Volunteer packages | content scanning, redaction, rights, zip hardening (§5) |
| Offline rating packets | content security policy `default-src 'none'`; data escaped against `</script>`; returns authenticated by a per-packet secret stored hashed; each item validated |
| Dependency vulnerabilities | `pip-audit`: no known vulnerabilities; `npm audit`: 0 |
| Phase 3.2 / 3.3 protections | loopback guard, Host/Origin, DNS-rebinding defence, remote consent, auth, admin token, network guard in tests, export/import, deletion: all in the passing suite and E2E |

**Residual risks:**

* A pack checksum proves integrity, not authorship. Contributed results stay
  `contributor_reported`.
* A packet secret is a bearer credential: whoever holds the file can return ratings.
* Offline timing is self-reported (flagged `client_timing`).
* SQLite lock contention (§4).

## 7. User experience (P17)

The Chromium E2E now includes a **new-user flow**. Measured: 34 of 34 steps PASS in 2 runs.

1. A fresh backend with an empty catalog. Discover shows the first-run hint, and no cards or
   scores appear. Diagnostics say "Catalog: empty …".
2. The licensed pilot is imported while the user watches. Diagnostics move to "Catalog: 29
   videos (29 real, 0 synthetic demo) · 29 evaluated · 29 hidden during blind annotation".
3. **Defect found:** the feed was empty with a wrong reason.
   * The backend now returns `empty_reason`: `empty_catalog`, `hidden_for_blind_annotation` or
     `no_match`.
   * The hidden case shows: "29 imported videos are hidden while blind human annotation of
     real_pilot@v1 is open … run: `avd annotate unblind --dataset real_pilot@v1`".
   * The first-run hint appears only for an empty catalog.
4. After `unblind`: 18 recommendation cards from real licensed items, labelled heuristic.

Also added:

* `/v1/status` reports catalog counts;
* the extension shows them;
* a short feed says it is not filled with unrelated videos;
* model evidence carries the note "Quotes are checked to exist … whether they support the score
  is not checked automatically".

## 8. Human validation (P4)

* **Offline rating packets.** `avd annotate packet | import-packet | release-packet`, migration
  0009 and `docs/annotation-pilot.md`.
  * Blind assignment as online: calibration first, at most 3 raters per item (assigned items
    count), test–retest repeats.
  * Status before score; 1–5 anchors; confidence; citations made by selecting transcript text,
    verified against the transcript on import; abstention statuses.
  * Enjoyment kept separate.
  * No model output and no popularity in the file.
  * Items without confirmed `external_annotation` rights are never packed.
* **Verified:** 12 unit tests, each run on SQLite and PostgreSQL (24 runs), and the browser E2E
  `scripts/e2e/annotation_packet.py` (12/12) on the real pilot. That run includes a citation
  made by selection, persistence across a reload, zero network requests, import, an idempotent
  re-import, and a forged return refused.
* **Templates:** invitation, participant instructions, and a consent text the owner must
  approve.
* **Time, estimated from word counts** (9,062–13,768 words per dev item, 200–250 wpm, plus
  rating), **not measured:**
  * 50–85 minutes per item;
  * 2.5–3 hours of calibration;
  * about 40 rater-hours for the 10-item, 3-rater pilot.

  Suggested split: 6 volunteers × 5 items.
* **Collected: 0 ratings. Agreement: not computable.** Pending recruitment.

## 9. Relevance benchmark (P4)

Query audit (in the README):

* **Leakage by construction.** 8 of 10 queries were written from the shows' topic labels and
  map onto one show each, so they partly encode the proxy.
* **Cross-show queries.** q08 and q10 are the real test of topical retrieval, and may have no
  relevant item.
* **Conjunctions** (q02, q06) and **broad queries** (q04, q09).
* **Recommendation:** add queries written by raters who have not seen the catalog, and report
  them separately.

Judgments are collected in the rating packets, at the end of each dev item, which the rater has
just read. `avd bench-relevance` now compares lexical retrieval, MiniLM, hybrid score fusion and
the **final recommendation list**, reporting:

* nDCG@5, Recall@5, precision@5, MRR and the off-topic rate;
* agreement (ordinal Krippendorff's alpha) and disputed pairs (grades spanning 0–2);
* the number of queries without a relevant item.

**No judgments exist**, so it reports `blocked_missing_labels`. No improvement is claimed from
proxy labels.

## 10. Recommendation quality (P5): proxy labels, dev split

Unchanged from Phase 3.3; telemetry off and policy versioning do not alter results:

| | hashing | MiniLM chunks | MiniLM keywords | fusion |
|---|---|---|---|---|
| Retrieval nDCG@5 | 0.8485 | 0.906 | 0.8559 | 0.9127 |
| Full recommender nDCG@5 (diversity 0.0) | 0.5891 | 0.7881 | 0.8105 | n/a |

Three policy versions appear in the report, one per embedder and document recipe.

**Candidate-level trace (new; MiniLM chunks, diversity 0.3):**

* **Retrieval is not losing items.** Every proxy-relevant item is retrieved.
* **Gopher persona.** `chl_gotime_179` (similarity 0.406) is dropped by the channel cap while
  two items from other shows are shown, with calibrated relevance 0.35 and 0.21. The soft cap
  still trades relevance for creator diversity whenever another creator matches. Whether people
  prefer that is a question for the relevance judgments; it was **not tuned**.
* **Podcast insider.** `chl_backstage_21` is lost to ordering: two Founders Talk episodes are
  more similar to "podcast production" in MiniLM. The proxy label may itself be wrong here.
* **Founder and ML personas.** Below-floor items appear only in the single labelled exploration
  slot, so feeds return 3–4 items rather than padding.

Scenario tests (13) cover:

* small and large catalogs;
* one dominant channel, which spreads first and relaxes only for leftover slots;
* many small channels;
* cold-start users;
* specific, ambiguous, unmatched and multi-interest queries;
* empty catalog versus no match versus hidden items.

They assert that the feed is never padded with off-topic items beyond the labelled quota.

## 11. Quality invariants and evidence semantics (P13, P14)

| Invariant | Coverage |
|---|---|
| Popularity independence | metamorphic (×10⁹ statistics), unchanged |
| Production independence | metamorphic, unchanged |
| Missing media is not zero | metamorphic, unchanged |
| Genre awareness | **new metamorphic test:** changing only factual accuracy (1 → 5) moves a lecture's and a science video's content score, but not a comedy sketch's or entertainment's |
| Research integrity | synthetic, heuristic, local-model, imported and contributor results, and human ratings stay labelled apart. New: offline ratings are flagged `offline_packet` and `client_timing`; community imports are always `contributor_reported` |

Evidence semantics, as implemented:

| Question | Automated? |
|---|---|
| Does the quote exist? | yes (`verify_quote`) |
| Is it at the cited segment? | yes: re-anchored within ±1 segment, otherwise unverified. Timestamps come from the segment, never from the model; times in a justification without cited evidence are flagged |
| Does it support the score? | **no** |
| Does it contradict the score? | only structurally (one segment cited for and against) |
| Is it insufficient? | only the "generic evidence" heuristic |
| Does the score need material not in the transcript? | only abstention for visual and audio dimensions |

There is no semantic entailment model, and none is claimed. The new `avd evidence-review sheet`
and `summarize` give a human the tooling for the open questions. Its verdicts are supports,
partially supports, insufficient, contradicts, wrong segment, needs outside material, and cannot
judge. It works on local dev runs only; the sheet holds transcript text and stays local.

## 12. Performance (P18), measured on 4 vCPU Linux, no GPU, heuristic evaluator

`benchmarks/reports/perf-local-linux-4vcpu-phase3_4.json`:

| | |
|---|---|
| Cold / warm start | 1.79 s / 1.23 s |
| Idle | 119 MB RSS, 0.2 % CPU |
| Pilot import / queue drain (29 items) | 1.44 s / 4.98 s |
| Search p50 / p95 | 23.1 / 26.2 ms |
| Recommendations p50 / p95 | 53.9 / 133.1 ms |
| Evaluation read p50 / p95 | 6.8 / 8.6 ms |
| Export (20 evaluations) / import | 1.02 s / 0.97 s |
| Shutdown | 0.42 s |

**Model performance: not measured** (no model). Heuristic throughput says nothing about local-LLM
throughput, and it is not used as a proxy for it.

## 13. Verification (final, this branch)

| Check | Result |
|---|---|
| `ruff check`, `ruff format --check`, `mypy avd` | clean (112 source files) |
| `alembic upgrade head && alembic check`, SQLite and PostgreSQL (plus downgrade to base and upgrade again on PostgreSQL) | no new operations |
| Backend, PostgreSQL + pgvector + Redis + SQLite | **553 passed, 1 skipped** (live Ollama). 80 new tests |
| Backend, SQLite only | 484 passed, 70 skipped |
| Extension `typecheck`, `vitest`, `build` | clean, **43 passed**, built |
| `docker compose config -q` | valid |
| `avd bench` (synthetic) | ran; unchanged |
| Chromium E2E (`scripts/e2e/full_local.py --chromium`) | **34/34 PASS**, 2 runs |
| E2E without a browser | 15/15 PASS, 1 SKIP (browser) |
| Offline packet E2E (`scripts/e2e/annotation_packet.py`) | **12/12 PASS** |
| Gates | pass 1, blocked_missing_labels 7, blocked_no_inference 4 |
| Live inference test | **skipped** (no model); not counted as a success |

## 14. License (P16)

PR #5 is reviewed and recommended for merge (`docs/licensing.md`):

* contributors are the owner plus assistant-authored commits on the owner's behalf;
* the PR adds only the unmodified Apache-2.0 text;
* the dependencies are compatible;
* no `NOTICE` file is needed.

Code, model, dataset and content licenses are tabulated separately. **Not merged; the owner
decides.**

## 15. Remaining blockers

1. **Real inference.** Allow the model registries in the environment's network settings, or
   have one volunteer run `avd community smoke`.
2. **Raters.** 3 or more volunteers for calibration plus 5 items each, or 6 volunteers for even
   coverage. Approve the consent text first.
3. Talk Python rights (test split only).
4. The license decision (PR #5).
5. Windows and macOS remain untested.

## 16. Next milestone

The highest-value next step supported by the evidence: **one volunteer smoke run**. It produces
the first real model call: digest, latency, validity and grounding on one real transcript.
**At the same time, the first calibration packets** go to 3 volunteers. Both now need only
people, not code. Their results decide whether the full benchmark and the 10-item rating round
are feasible at the measured time per item.
