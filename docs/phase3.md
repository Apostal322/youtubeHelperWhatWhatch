# Phase 3: real-world AI evaluation, validation and production readiness

**Status in one paragraph.** Phase 3 made the system ready to produce evidence, and found and
fixed several things that would have invalidated it. It did not produce the evidence itself:
- **No live model has been called.** This environment has no model API key, and no budget was
  approved.
- **No human has rated anything.** No raters are available here.

So the central research question is **still open**: *can the AI evaluator reliably separate
higher- from lower-quality content, in agreement with independent people?* Nothing below should
be read as an accuracy claim. Section 7 lists exactly what is missing and who must provide it.

## 1. Pre-flight audit (verified against code and runs, 2026-10-19)

Baseline reproduced on `main` (4e1c887): 297 backend tests and 28 extension tests pass, and CI
was green.

| Capability | Status | Evidence |
|---|---|---|
| LLM provider adapters (Anthropic, OpenAI-compatible) | Implemented, untested live | Scripted providers and a local OpenAI-protocol mock only |
| Prompts and rubric (14 dimensions, 5 anchors each) | Implemented and verified offline | Contract tests; base prompt now versioned in a registry |
| Schema validation, retries, timeouts, spend caps | Implemented and verified | Unit tests; Phase 3 added a hard run ceiling |
| Config change ⇒ re-evaluation | **Partially implemented (defect found)** | Handler-level check worked; the job path never re-queued Stage B. Fixed (Section 5) |
| Evidence verification (quote exists) | Implemented and verified | Quote-matching tests, now a full matrix |
| Evidence *support* (quote supports claim) | Missing → partially implemented | Deterministic support audit added; semantic entailment not automated |
| Long transcripts (chunk, map-reduce) | Implemented, untested live | Scripted tests on real 600k-character transcripts |
| Prompt-injection defence | Implemented and verified offline | Framing, detector, grounding; real-attack recall unmeasured |
| Genre v2 | Implemented; real accuracy unknown | Dev split 8/10 interview_podcast; no human genre labels; test split exposed |
| Fact verification | Implemented, untested live; no reference corpus shipped | v3 taxonomy (Section 5) |
| Real dataset (29 items) | Implemented and verified (checksums in CI) | Splits verified: calibration 2, dev 10 (all Changelog), test 17 (10 Changelog + 7 Talk Python). No exclusion split |
| Dataset rights | **Partially verified (issue found)** | Talk Python licence scope unconfirmed (Section 2) |
| Annotation workflow | Implemented and verified technically; **0 human ratings** | API, end-to-end and browser tests |
| Benchmark v2 + 12 locked gates | Implemented and verified | Gate report mislabelled heuristic results as passes; fixed (Section 5) |
| Embeddings (MiniLM ONNX) | Implemented and verified | Retrieval experiment; recommendation advantage unresolved (Section 4) |
| Recommender v2 | Implemented; only proxy-evaluated | Topical proxy labels only |
| Media path (ffmpeg) | Implemented and verified on generated media | **Docker image lacked ffmpeg** (fixed); no real creator media processed |
| Visual / editing evaluation | Interface only (`FrameJudge`) | Not implemented |
| API rate limiting | **Partially implemented (issue found)** | Per process; token-rotation bypass. Fixed (Section 5) |
| Auth, deletion, revocation | Implemented and verified | Tests; derived-data retention gaps documented in `docs/security.md` |
| Extension labels | Implemented and verified | 29 vitest tests; AI-mode label changed (Section 5) |
| CI | Implemented and verified | Runs on `claude/**` and pull requests: lint, types, tests (SQLite, PostgreSQL, Redis, ffmpeg, MiniLM), migrations, audits, licensed-pilot verification, Docker |

## 2. Data integrity and rights

**Rights review** (`avd dataset rights`, `backend/avd/datasets/rights.py`). Verified at the
pinned commits:

* **Changelog: explicit content licence.** The README's "License & Copyright" section applies
  CC BY-SA 4.0 to the transcripts and excludes `bigtent/` (Grafana Labs).
  * Remaining assumption: Changelog holds the rights to guests' spoken words.
* **Talk Python: licence scope unconfirmed.** The root `LICENSE` is the GPL v3, a software
  licence. Nothing says it covers the transcript text. The README invites people to "use the
  files to help you explore the conversations". Phase 2 had recorded "GPL-3.0" as if that
  settled the question.
  * **Enforced now:** internal analysis only. No assignment to external raters, quotes
    withheld from public responses, no redistribution.
  * **Remediation:** a written confirmation from the publisher, recorded in
    `benchmarks/datasets/rights_confirmations.json`.

**Held-out exposure** (`benchmarks/datasets/real_pilot_v1/exposure.json`). The 17-item test
split influenced four Phase 2 decisions:

| # | Decision | What it saw |
|---|---|---|
| 1 | Genre v2 design | Classifier outputs on all 29 items |
| 2 | Injection-detector v2 false-positive tuning | 5 of the 29 pilot items were flagged |
| 3 | Keeping hashing as the embedding default | The test-split recommendation check |
| 4 | MiniLM's relevance floor and ceiling | Set from a retrieval run that included 5 of the 7 Talk Python test items |

Consequences:
- **Still clean for model-versus-human quality agreement:** no labels exist, and no prompt,
  rubric, aggregation or gate was tuned on test outputs.
- **Not usable for confirmatory claims about those four components.** The injection gate on that
  split is reported as `not_applicable_exposed`.
- **Remediation:** `real_holdout_v2`, a sealed set of 12 unseen Changelog episodes from the test
  shows. Only eligibility statistics were computed for it, and nothing has been evaluated. Brain
  Science had no remaining eligible episode, and the eligibility rules were not relaxed.

**Domain limits of all real data here:**
- English only;
- interview and discussion podcasts only;
- mostly about software;
- two publishers;
- transcripts only, no media.

Nothing here can support claims about other genres, other languages or visual quality.

## 3. Experimental protocol and status

| Experiment | Goal | Status | Ready to run |
|---|---|---|---|
| A. Real LLM baseline | Schema-valid, grounded evaluations run end to end | **Blocked: no API key or approved budget.** Dry run done: 10 dev items, Opus 5.5, Stage B, **expected $1.21, worst case $4.14** | `avd evaluate-real --budget-usd 5` |
| B. Human agreement | Reproducibility per dimension among people | **Blocked: no raters.** Workflow ready, blinded and browser-checked | `docs/annotation.md` (recruitment steps) |
| C. AI vs humans | Agreement, dimension-specific weaknesses | Blocked by A and B | `avd bench-real --labels ... --evaluator configured` |
| D. Prompt stability | Prompt sensitivity | Blocked by A. Two one-change variants registered | `--prompt rubric_prompt_v2-evidence_first` / `-strict_abstain` |
| E. Model comparison | Cost / reliability / agreement trade-offs | Blocked by A | `--model claude-sonnet-5-5` etc. |
| F. Evidence reliability | Groundedness, not just JSON validity | Machinery done. Existence check + support audit, 15-case matrix. Real rates need A | Runner reports claimed/verified/ungrounded/support flags |
| G. Popularity independence | Scores invariant to popularity | Heuristic pipeline: passed on dev (×1000 popularity, 0 changes). LLM path: prompt-level invariance tested with scripted judge. Live run blocked by A | `bench-real` |
| H. Semantic recommendation quality | Real relevance judgements | **No human relevance labels.** Exploratory dev diagnostic done (Section 4) | Relevance annotation protocol: Section 4 |
| I. Final held-out | Generalisation | Blocked by A–C and a locked candidate | `real_holdout_v2`, `--confirm-test` |

**Engineering completion vs scientific validation.**
- **Engineering:** complete for the transcript path (Section 6).
- **Scientific validation:** none of its criteria are met. That needs real-model evaluations,
  independent human judgements, a valid held-out comparison, uncertainty analysis against labels
  and baselines.

## 4. Measurements this phase (all offline; no accuracy claims)

**Release gates** (`avd gates status --report ...`, dev split, heuristic run):
- 1 passes: the injection detector, which does not depend on the evaluator.
- 7 are `blocked_missing_labels`.
- 4 are `blocked_no_inference`.

Release-ready: **no**. Phase 2's report had shown cost, latency and popularity invariance as
"pass": values measured on the heuristic baseline, not the AI candidate.

**Embedding diagnostic** (`recs-diagnostic-real_pilot_v1-dev`; exploratory; 4 dev personas ×
10 items; topical proxy labels; nDCG@5):

| Embedder | Full recommender | Similarity only |
|---|---:|---:|
| hashing (keyword recipe) | 0.54 | 0.85 |
| MiniLM (transcript windows) | 0.73 | 0.91 |
| MiniLM (keyword recipe) | 0.75 | 0.86 |

Readings:
1. **The test-split result (hashing 0.79 vs MiniLM 0.69) does not replicate on dev.** With this
   few items the ordering is unstable.
2. **The ranking stage loses relevant items for every embedder.** Hashing's relevance floor
   leaves slots empty (3.25 of 5 returned). The floor/ceiling calibration (hashing: synthetic
   personas; MiniLM: an exposed experiment) is the most likely cause, more than the embedder.
3. **Document representation is not decisive here.**

No default changes on this evidence.

**Separate retrieval tasks for a proper relevance dataset (Experiment H):**
1. Exact-episode retrieval: already measured, R@1 MiniLM 0.71 vs hashing 0.28.
2. Topical search: queries written by annotators, graded 0–2 per item.
3. Related-content recommendation: item-to-item, graded.
4. Personalised relevance: intents and interests from consenting users.

Each needs human judgements on held-out items. The blind annotation infrastructure supports raters;
a relevance task is the next addition.

**Throughput and cost** (`throughput-real_pilot_v1`; the call plan is measured, the rest
projected):

| Videos/day | Projected LLM $/day | Notes |
|---:|---:|---|
| 100 | ≈ $20 | |
| 1k | ≈ $197 | |
| 10k | ≈ $1,969 | Discovery needs 20,200 YouTube units/day, over the 10k default quota |
| 100k | ≈ $19,694 | |

At 3× output tokens the cost is about 1.8× higher. Owner-authorised captions cost 250 units per
video and need each creator's OAuth grant.

**Authorised content acquisition, not inference cost, is the first scaling constraint.** Failure
rates are unmeasured until Experiment A.

## 5. Problems found and fixed in Phase 3

| Problem | Category | Fix | Regression test |
|---|---|---|---|
| Talk Python recorded as GPL-licensed content; licence scope never confirmed | Data / rights | Use-time rights review; pending uses fail closed | `test_rights.py` |
| Test split had informed four decisions | Data / protocol | Exposure ledger, gate status, sealed hold-out | `test_benchmark_v2.py` |
| Config change (model/prompt) never re-ran Stage B via the job path (dedup key lacked the config hash); D2's test bypassed dedup | Infrastructure | Config hash in Stage B/C keys | `test_phase2_llm.py::test_configuration_change_requeues_stage_b_through_the_job_path` |
| Gate report showed heuristic measurements as candidate passes | Protocol | Seven explicit gate statuses; heuristic values kept as `baseline_value` | `test_benchmark_v2.py` |
| Raters could look up the AI's assessment via search, id or recommendations | Protocol | Per-dataset public blinding | `test_e2e_scenario.py` |
| UI defaulted every dimension to "abstain" (silent abstentions) | Rubric / data | A status must be chosen; new statuses | Browser check |
| Rate limit bypass by rotating fake bearer tokens; limits per process | Security | Caller + IP buckets, shared via Redis | `test_ratelimit_shared.py` |
| Docker image had no ffmpeg (media path returned 503); ffmpeg unbounded except timeouts | Infrastructure / security | ffmpeg + prlimit in image; OS limits; duration cap | `test_media.py`, CI docker step |
| One disputed claim of two dropped factual accuracy to 1.0–2.0 | Rubric | Smoothed, severity-weighted error rate | `test_factcheck_v2.py` |
| No "outdated" verdict; old-but-once-true claims would be "contradicted" | Rubric | `outdated` verdict, publication date and passage dates | `test_factcheck_v2.py` |
| Duplicate feedback events on retry | Infrastructure | Idempotent feedback; daily cap | `test_ops.py` |
| AI-mode header read "LLM · provider", no human-validation caveat | Product truthfulness | "AI evaluation · not human-validated" | extension tests |

**Error taxonomy for live runs.** Each category has a detector that runs today. Counts will come
from Experiment A.

| Category | Detection |
|---|---|
| Hallucinated evidence / quote | Quote verification (`evidence.invalid_rate`), `invented_quote_in_justification` |
| Timestamp mismatch | Timestamps re-derived from our data; `timestamp_not_in_cited_evidence` |
| Unsupported explanation | Support audit flags; human evidence comparison in Experiment C |
| Production/content confusion | Prompt rule; technical dimensions only from media evaluators |
| Prompt-injection influence | Detector + grounding; flagged segments cannot be evidence |
| Model refusal / truncation / parse failure | Ledger error types (`LLMRefusalError`, `LLMOutputError`) |
| Incorrect genre | Genre status; manual override; genre labels pending |
| Poor calibration | ECE / risk–coverage once labels exist |
| Subjective disagreement | Human–human α per dimension (Experiment B) |
| Multilingual failure | Untested: no non-English real data |

## 6. Testing

* **Backend:** 335 tests on SQLite, PostgreSQL (pgvector) and Redis with ffmpeg and MiniLM; ruff
  and mypy clean; migration 0006 upgrades and downgrades and matches the models.
* **Extension:** 29 vitest tests; typecheck; build.
* **Audits (2026-10-19):** pip-audit 0 known vulnerabilities in 76 packages; npm audit 0.
* **Browser:** the rater UI against a live server with the real pilot. Status guide, draft
  restore and refusal of unset statuses work; blinded items return 403 and are absent from
  search.
* **CI:** see the PR for the current run. Public CI uses mocked inference only.

## 7. Blockers: what the project owner must provide

1. **An API key and an approved budget** for Experiment A. Add the key as an environment secret;
   a $5 ceiling covers the 10 dev items (worst case $4.14). The command is in
   `docs/model-configuration.md`.
2. **At least 3 raters and 1 adjudicator** for Experiment B, with consent and pay approved
   (`docs/annotation.md`, "Recruiting"). The dev split first, about 25–40 rater-hours.
3. **Written confirmation from Talk Python** if its transcripts are to be shown to raters or
   quoted.
4. **A decision on more data:** a broader licensed corpus (other genres and languages) is
   needed before any cross-genre claim.

## 8. Phase 4 (multimodal): prepared, not started

Start only after Experiments A–C show that transcript-based evaluation agrees with people. A
reliable text evaluator is the yardstick for the multimodal one.

* **Data.** Media whose licence and source permit download and processing:
  * creator partnerships (explicit grants via the existing upload-with-attestation path);
  * openly licensed lecture and documentary collections that offer downloads.

  Public YouTube availability is not permission. Wikimedia Commons, the Internet Archive and MIT
  OpenCourseWare were unreachable from this environment and must be checked case by case for
  licence terms.
* **Audio.** Measurable signals first:
  * loudness and range (EBU R128 via ffmpeg's `ebur128` filter);
  * clipping, noise floor and SNR (implemented);
  * dropouts;
  * speech intelligibility proxies.

  Then validate them against human audio ratings. No semantic claims from signal metrics.
* **Video.** Behind the existing `FrameJudge` interface:
  * frame sampling (implemented, bounded);
  * scene-change detection;
  * text/diagram readability and visual explanation judged by a vision-language model on
    sampled frames, with frame-index evidence;
  * audio–visual sync.
* **Evaluation.** Genre-specific multimodal rubrics, human raters with the media (not
  transcripts), and the same blinding, gates and held-out discipline.
* **Cost.** Vision calls on 24 frames per item add input tokens per frame. Measure with the
  dry-run planner before any live run.

## 9. Phase 5 (continuous scale): constraints first

* **Acquisition.**
  * YouTube Data API quota: search costs 100 units per 50 results, against a 10k default.
  * Captions need each owner's OAuth.

  Scale therefore comes from creator opt-in and licensed corpora, not crawling. Track coverage
  by genre, language and channel size (`/v1/admin/coverage`).
* **Routing.** A cheap first pass (heuristic or small model) and the expensive model only when
  the first pass is uncertain. Accept routing only where Experiment C shows it keeps agreement.
* **Batching and caching.** Use the Batch API where available; the response cache and config
  hash already prevent paying twice.
* **Operations.**
  * Queue depth and cost per hour in `/metrics`.
  * Shared rate limits (done).
  * Separate media workers in network-less containers.
  * Small-creator exposure audits, already in the benchmark.
