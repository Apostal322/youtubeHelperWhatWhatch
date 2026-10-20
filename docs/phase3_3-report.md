# Phase 3.3 report: real AI experiments, evaluation reliability, recommendation quality, release hardening

Date: 2026-10-19. Branch `claude/phase3-3-validation`, stacked on Phase 3.2
(`claude/phase3-2-self-hosted`), which is stacked on Phase 3.1 and Phase 3. None of them is merged.

## Bottom line

* **No real model was run.** This environment still cannot install Ollama or any model
  weights:
  * `ollama.com`, `registry.ollama.ai` and `huggingface.co` are refused by the egress proxy
    (connection status 000). Each was checked once, not retried.
  * The machine has no GPU (4 vCPU, 15 GB RAM).

  Nothing in this report is a model result, and nothing was simulated as one. Instead, a
  volunteer can now run a model on their own computer and submit the result, and the result is
  checked mechanically when imported (`avd community pack | verify | import`). Imported results
  are labelled `contributor_reported`, never `verified`.
* **Evidence of independence.** New metamorphic tests show that the quality assessment does not
  move when popularity is inflated a billion-fold, or when production quality is poor. Missing
  media is excluded from scoring, not scored zero. One leak was found and fixed: popularity
  wording in titles and descriptions reached the retrieval documents.
* **Recommendations: stage losses measured on real data.**
  * Retrieval was never the bottleneck: recall@5 is 1.0 for every embedder.
  * The loss came from a hard per-channel cap, which emptied slots when the dev pilot has few
    channels. It is now a soft cap, explained to the user. Full-recommender nDCG@5 rose by
    0.05–0.06 for every embedder.
  * The labels are proxies (show membership). A human relevance protocol is written; no human
    judgments exist.
* **Reliability fixes found by new tests:**
  * a job that crashes the embedded worker is now requeued at once, instead of staying
    `running` until its lease expires;
  * the extension now re-registers once when its token has expired, instead of failing.
* **Release gates: unchanged**, at 1 of 12 passing. 7 are blocked by missing human labels and 4
  by missing real inference. No gate was relaxed.
* **License:** the exact Apache-2.0 change is prepared as a separate draft pull request for the
  owner. It is not merged and not part of this branch.

## 1. Inspection and test reproduction (P0)

### Baseline, before any change

| Suite | Result |
|---|---|
| Backend on SQLite + PostgreSQL/pgvector + Redis | 447 passed, 1 skipped (the live Ollama test) |
| Backend on SQLite only | 391 passed, 57 skipped (PostgreSQL/Redis tests) |
| Backend, test order shuffled, 2 seeds (SQLite) | 391 passed, 57 skipped in both runs: no order dependence |
| Extension (vitest) | 41 passed |
| Gates | pass 1, blocked_missing_labels 7, blocked_no_inference 4 |

The shuffled runs used a small local pytest plugin, because `pytest-randomly` cannot be
installed in this environment. The plugin is not committed.

### Classification of subsystems

| Subsystem | Verified by | Status |
|---|---|---|
| Ingestion, jobs, workers, API, auth, export/import, local mode | unit + integration tests; real-process E2E (§5) | **tested with real processes** |
| Extension | vitest; Chromium E2E with the built extension (§5) | **tested in a real browser** (Linux) |
| Embeddings (hashing, MiniLM ONNX) | real model on real dev data | **real** |
| Recommender | real dev data with proxy labels; synthetic benchmark | real data, **proxy labels** |
| Heuristic evaluator | real dev data | real, baseline only; its validity is unknown |
| LLM evaluator (Ollama, hosted) | protocol mocks | **plumbing only**. No real inference here |
| Human validation (annotation, agreement) | workflow tests | **no labels exist** |
| Windows, macOS | — | **not tested** |

## 2. Real local AI inference (P1)

**Blocked.** No model could be installed (see the bottom line). The block was not bypassed.

What exists instead:

* **Volunteer package** ([`volunteer-pilot.md`](volunteer-pilot.md)): run the dev split on your own
  computer with `avd bench-local`, then pack the result with `avd community pack`. The archive
  holds summaries and reports only. The run database, transcripts and media are excluded, so it
  contains no licensed text.
* **Import with verification** (`avd community verify`, `avd community import`). An archive is
  refused unless:
  * its format and version are known;
  * it says real inference (`transport_kind: real`; a mock run says `injected`);
  * it is on the dev split, and its manifest hash matches this repository's dev manifest;
  * it ran on a clean commit;
  * every model has a weights digest (`@<12 hex>:`) and hardware is recorded;
  * every rate is within [0, 1] and the cost is $0;
  * it has no unsafe paths (zip-slip).

  Accepted results land in `benchmarks/community/` with a `VERIFICATION.json`. Their status is
  `contributor_reported`: the checks prove consistency, not that the model really ran.
* `bench-local` reports now carry `format`, `format_version`, `dataset`,
  `manifest_items_sha256` and `transport_kind`. The Ollama fingerprint now drops the `sha256:`
  prefix, so digests are always 12 hex characters.

Tests: `tests/test_community.py` covers packing, each refusal, tampering, unsafe paths and
import.

## 3. Evaluation quality and evidence (P2)

New in `tests/test_metamorphic.py`:

| Property | Test | Result |
|---|---|---|
| Popularity does not change quality | views, likes and subscribers in the DB multiplied by 10⁹ | evaluator input, cache fingerprint and every dimension score unchanged |
| Production is independent | a stub evaluator with poor production signals | content and presentation scores unchanged |
| Missing media | no audio/video available | visual/audio dimensions excluded with coverage reduced, **never scored 0** |
| Retrieval documents | titles/descriptions containing "10M views", "viral hit" and similar | popularity wording stripped before embedding (**this failed before the fix**) |
| Diversity | soft channel cap | at most 2 per channel while alternatives exist |

* **The fix:** `avd/embeddings/documents.py` now applies the same sanitizers as the evaluator
  (`sanitize_title`, `sanitize_description`, `sanitize_tags`).
* **Already covered, re-checked:**
  * genre suitability: genre profiles weight dimensions, and a dimension that does not apply
    abstains;
  * quote *existence* is verified separately from quote *support* (Phase 3 evidence checker).

  Whether a model's supported quotes really support its claims needs human labels: blocked.

## 4. Recommendation quality (P3)

`avd bench-recs-stages` on `real_pilot_v1` dev: 4 personas, k = 5, proxy labels (an item is
relevant if it is from the persona's show). Reports:
`benchmarks/reports/recs-stages-real_pilot_v1-dev{,-before}.{json,md}`.

### Retrieval only (similarity ranking)

| Embedder | recall@5 | nDCG@5 |
|---|---|---|
| hashing (lexical) | 1.0 | 0.8485 |
| MiniLM, chunk text | 1.0 | 0.9060 |
| MiniLM, keywords | 1.0 | 0.8559 |
| score fusion (hashing + MiniLM) | 1.0 | **0.9127** |

The hybrid is score fusion, the mean of the two cosines. A concatenated hybrid vector is not
possible, because the vector column is fixed at 384 dimensions.

### Full recommender, diversity 0.0: before and after the soft cap

| Embedder | nDCG@5 before | after | items returned (of 5) before → after |
|---|---|---|---|
| hashing | 0.5386 | 0.5891 | 3.25 → 3.5 |
| MiniLM chunks | 0.7294 | 0.7881 | |
| MiniLM keywords | 0.7518 | 0.8105 | |

**Where relevant items were lost.** Each candidate's fate is traced: selected,
ineligible:<reason>, below the relevance floor, channel cap, or not selected.

* The **hard channel cap** dropped relevant items and left slots empty, because the dev split
  has few channels. It is now **soft**: when only capped channels remain, the item is taken and
  explained ("More from a channel already shown: no other matching creator is left for this
  slot").
* With the **hashing embedder**, 4 of 10 relevant items fall below the 0.06 relevance floor.
  The floor was **not tuned** on this data, because the dev split is too small to tune on
  without overfitting. This is recorded as a known limitation.

The synthetic benchmark (`avd bench`) is unchanged: nDCG@5 0.8494, with identical exposure.

**Human relevance protocol.** `benchmarks/datasets/relevance_v1/`, with
`avd bench-relevance sheet|run`:

* 10 queries over dev items only, graded 0–3, judged independently by 3 or more people;
* pooled by median, with ordinal Krippendorff's alpha;
* judgments are validated: unknown queries, non-dev items and duplicates are refused;
* without judgments, the command reports `blocked_missing_labels` and no number.

No judgments exist, and none were generated.

## 5. End-to-end, reliability and performance (P4)

### `scripts/e2e/full_local.py`

Runs real processes (`avd start`, embedded worker, SQLite) and, with `--chromium`, the built
extension in Chromium via Playwright. It covers:

* the first start and database initialization;
* importing the licensed pilot, with jobs created, processed and stored;
* the evaluated videos indexed and listed;
* search, recommendations and feedback;
* exporting evaluation records;
* a clean Ctrl+C stop, a restart, and the data persisting;
* importing into a second instance, where the results stay `imported_unverified`;
* deleting user data, after which the old token is rejected.

In the browser it checks:

* that the extension connects and shows connection diagnostics;
* recommendations, the quality breakdown with labels and evidence, search, preferences and
  feedback;
* that malformed or plain-HTTP remote URLs are refused;
* that a remote backend needs consent, and that consent is blocked while the backend is
  unreachable;
* that expired credentials recover;
* a clear error while the backend is down, and recovery after it restarts.

Host/Origin rejection and the crashed-job release are covered by pytest, not by this script.

| Run | Result |
|---|---|
| With Chromium (2 runs) | 28/28 PASS |
| Without a browser (2 runs) | 15/15 PASS, 1 SKIP (browser) |
| Earlier run | step 19 (delete) failed once. Its HTTP status was not captured, so the step now prints it; it passed in the 4 runs since. **Not root-caused**; recorded as a possible flake |

### Bugs found and fixed

* **A crashed job stayed `running`.** If a job raised a database error inside the embedded
  worker, it stayed `running` until its lease expired. It is now released at once: requeued, or
  marked dead once its attempts are spent (`last_error: "worker crash: …"`). Test:
  `test_embedded_worker_survives_database_errors`, which fails without the fix.
* **An expired token broke the extension.** It now drops the token and re-registers once
  (`extension/src/lib/api.ts`, plus a vitest case).

### Performance

`scripts/perf/local_perf.py`, Linux, 4 vCPU, no GPU, heuristic evaluator. Output:
`benchmarks/reports/perf-local-linux-4vcpu.json`.

| Measure | Value |
|---|---|
| Cold start / warm start | 1.86 s / 1.24 s |
| Idle | 118 MB RSS, 0.2 % CPU |
| Demo import drain (20 items) | 0.66 s |
| Pilot import / drain (29 items) | 1.41 s / 5.23 s |
| Search p50 / p95 | 23.7 / 30.1 ms |
| Recommendations p50 / p95 | 53.5 / 129 ms |
| Video evaluation read p50 / p95 | 6.7 / 10 ms |
| Export (20 evaluations, 198 KiB) / import | 1.13 s / 0.92 s |
| Shutdown | 0.52 s |
| Database size | 3.68 MB |

These are heuristic-mode numbers. Local-model latency is unknown until a model runs.

**Cross-platform:** [`local-setup.md`](local-setup.md) gives Windows PowerShell and macOS
steps, but **only Linux was tested**.

## 6. Security

* **Phase 3.2 protections still pass:** the loopback guard, Host/Origin validation, server-mode
  fail-closed, the test network guard, and remote-backend consent in the extension.
* **New this phase:**
  * popularity text is stripped from retrieval documents;
  * crashed jobs are released at once;
  * an expired token is retried once;
  * community archives are checked for path traversal, and only allow-listed files are
    extracted.
* No endpoint is exposed beyond loopback. No cloud service, paid infrastructure or commercial
  API key was added.

## 7. Open-source readiness (P5)

| Item | Status |
|---|---|
| License | Apache-2.0 declared; the `LICENSE` text is **prepared in a separate draft PR** for the owner (SHA-256 `cfc7749b…`). Dependency licenses are all compatible ([`licensing.md`](licensing.md)) |
| Contributor guides | [`CONTRIBUTING.md`](../CONTRIBUTING.md), [`research-contributing.md`](research-contributing.md), [`volunteer-pilot.md`](volunteer-pilot.md), relevance protocol, [`extending.md`](extending.md) |
| Model evaluation / benchmark reproduction | [`research-contributing.md`](research-contributing.md) (one command per committed report) |
| Beginner issues | [`good-first-issues.md`](good-first-issues.md): ready to file; filing them is the maintainer's call |
| Release checklist | [`release-checklist.md`](release-checklist.md) |
| Release gates | 1 pass, 7 blocked (labels), 4 blocked (inference). **Not release-ready** as a validated quality tool; usable as a heuristic-mode preview |

## 8. Verification (final, on this branch)

| Check | Command | Result |
|---|---|---|
| Lint / format | `ruff check avd tests ../scripts`, `ruff format --check` | clean |
| Types | `mypy avd` | no issues (109 files) |
| Migrations | `alembic upgrade head && alembic check` on a fresh DB | no new operations |
| Backend, full | `AVD_TEST_DATABASE_URL=… AVD_TEST_REDIS_URL=… pytest -q` | **473 passed, 1 skipped** (live Ollama) |
| Backend, SQLite only | `pytest -q` | 417 passed, 57 skipped |
| Extension | `npm run typecheck && npm test && npm run build` | clean, **42 passed**, built |
| Compose | `docker compose config -q` | valid |
| Benchmark | `avd bench` | ran; synthetic metrics unchanged |
| Gates | `avd gates status --report …real_pilot_v1-dev.json` | pass 1 / labels 7 / inference 4 |

The 26 new backend tests are in `test_community.py`, `test_metamorphic.py` and
`test_relevance.py`, plus additions to `test_local_first.py` and `test_research.py`.

## 9. Remaining blockers

1. **Real inference:** allow the model registries in the environment's network settings, or
   have a volunteer run the pilot.
2. **Human labels:** quality ratings (3 raters per item) and relevance judgments. Both need
   people, and consent for publishing aggregates.
3. **Talk Python rights:** annotation, public quotes and redistribution are still pending.
4. **License:** the owner's decision on the draft PR.
5. **Windows and macOS:** untested.
6. **Hashing relevance floor:** suspected too strict; needs more dev data to set without
   overfitting.

## 10. Next milestone

First real evidence:

* one real local-model run on the dev split (here once registries are reachable, or contributed);
* the first 3-rater annotation round on the 10 dev items;
* relevance judgments for the 10 queries.

Then compute the 11 blocked gates from real data, and decide from those numbers whether to lock
a configuration for the sealed hold-out.
