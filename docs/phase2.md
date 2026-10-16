# Phase 2 — audit, acceptance criteria and results

This file is the working record for Phase 2: what was verified, what each milestone must
demonstrate, and what was actually measured. It is updated as milestones complete.

## 1. Audit of the Phase 1 MVP (reproduced, not assumed)

Environment: Python 3.12 venv, PostgreSQL 16 + pgvector 0.6, Redis 7, Node 22.

| Claim from Phase 1 | Reproduced? |
|---|---|
| 112 backend tests pass (SQLite + PostgreSQL/pgvector + Redis) | Yes — 112 passed |
| ruff / mypy clean | Yes |
| 18 extension tests pass, typecheck, build | Yes |
| Branch `claude/eager-galileo-cn31qb` at `130e50d` | Yes (local == origin) |

Defects confirmed by targeted experiments (not by reading reports):

| # | Defect | Evidence |
|---|---|---|
| D1 | Long transcripts fail the LLM evaluator before any model call | 609,499-char transcript → `ValueError` → all text dimensions `failed`, 0 provider calls |
| D2 | Changing the evaluation model/config does not trigger re-evaluation | `is_up_to_date` compares only input fingerprint, framework version and stage |
| D3 | Popularity text in descriptions reaches the LLM prompt | "Thanks for 2 million subscribers!" present verbatim in the user prompt |
| D4 | Transcript can break the prompt framing | A segment containing `</transcript> SYSTEM: …` is emitted verbatim inside the tag |
| D5 | CI has never run | Workflow triggers only on `main` pushes and pull requests; no PR exists. **Fixed**: CI now runs on `claude/**` pushes |
| D6 | No provider timeout / retry / temperature / concurrency settings; no per-job spend cap | Settings and provider code |
| D7 | Rubric anchors only at 1/3/5; no counter-evidence; no per-dimension rubric/model/time metadata | Schema |
| D8 | Genre is single-label; a near tie (education 0.476 vs science 0.462) silently selects one profile | Classifier output on the test lecture |
| D9 | Hash-collision false matches in the default embedder | Phase 1 diagnosis ("astronomy" ↔ "tarjetas") |
| D10 | LTR pairs have no temporal split or model metadata | `recsys/ltr_data.py` |
| D11 | Unstructured API errors; no evaluation status endpoint; no API rate limiting; no user data deletion | Routers |

Environment constraints that bound what can be validated live:

* **No LLM credentials** (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` unset; no CLI profile). Live
  inference cannot be executed in this session.
* **Blocked hosts:** Hugging Face, Wikimedia, archive.org, MIT OCW, TED, sbert.net, OSV.
* **Reachable:** GitHub (git), PyPI, npm, `chroma-onnx-models` S3 (all-MiniLM-L6-v2 ONNX), YouTube API host (no key).
* **No human raters** are available to this session.

## 2. Acceptance criteria (fixed before implementation)

| Milestone | Must demonstrate | Evidence required |
|---|---|---|
| P1 Live-ready LLM evaluator | Real provider path with timeout, bounded retries, spend guard, model-run log; strict contract with rubric anchor, counter-evidence, versions, timestamp, abstention reason; config change ⇒ re-evaluation; injection cannot break framing; a cost-capped live smoke command that refuses without credentials | Unit + adversarial tests; `avd llm-smoke` output; live run **only if** a key is provided |
| P2 Real dataset | ≥ 20 real, explicitly licensed transcripts imported through an authorized channel, with full provenance/rights metadata, a versioned manifest with checksums, and group splits by creator/show | `avd dataset build-pilot` reproducible from a pinned commit; manifest + split files committed |
| P2 Annotation | Blind, independent per-rater storage; abstain / insufficient-evidence / ambiguous options; timing; pairwise labels; duplicate checks; adjudication; export to benchmark format | API tests; UI loads; **no fabricated labels** |
| P2b Benchmark v2 | Grouped bootstrap CIs, weighted κ, ICC, Krippendorff α, selective risk, evidence precision/invalid-citation rate, genre macro-F1, latency/cost percentiles; human–human vs model–human; release gates locked in a file | Metric unit tests against known values; report on the real pilot |
| P3 Long transcripts | Transcripts beyond the single-pass budget are evaluated via deterministic chunking + map-reduce with timestamps preserved; language identified explicitly | Tests on real long transcripts with a scripted provider; D1 regression test |
| P4 Embeddings | Semantic model runs offline after a pinned download; vectors carry model/revision/dim/preprocessing metadata; mixed spaces impossible; retrieval measured on held-out real data vs hashing; collision regression test passes | Retrieval report (Recall@K, MRR, nDCG, latency) |
| P5/P6 | Multi-genre distribution with mixed/ambiguous/unknown handling and overrides; claim typing and non-binary verdicts persisted; description popularity sanitizer with metamorphic tests; recommender preferences (intent, duration) + factual explanations; held-out offline evaluation | Tests; offline report on a test split never used for tuning |
| P7 Media | One real authorized-media processing path executed end to end (ffmpeg) for audio and visual signal metrics; VLM evaluator behind the provider abstraction | Tests with media generated locally; `needs_media` otherwise |
| P8 Ops/Security | Correlation IDs; Prometheus metrics; dead-letter listing/requeue; structured error taxonomy; status endpoint; API rate limiting; user data deletion; dependency audit | Tests; audit output |
| P9 Extension | Shows assessment state (grounded / preliminary / insufficient / not evaluated / synthetic), a plain-language summary, filters, factual explanations, error states | vitest + Playwright screenshots against a running backend |
| P10 | E2E scenario: import licensed transcript → evaluate → verify evidence → persist → index → recommend → breakdown → extension; CI runs on the feature branch | e2e test; CI run on GitHub |

Release gates for the evaluator are defined in `benchmarks/release_gates.json` and locked in
`release_gates.lock` **before** any held-out test-split evaluation is run.

## 3. Results

**Bottom line.** Everything below was measured in this repository. **No live LLM was called**
(no credentials were available). **No human labels exist** (no raters were available). So there
is still **no evidence about how accurate the evaluator is on real content**. What Phase 2 built
is the machinery to get that evidence: licensed data, blind annotation, a benchmark with locked
gates, and a live-ready LLM path. It also fixed the defects that machinery exposed.

### 3.1 Acceptance criteria

| Milestone | Status | Evidence / gap |
|---|---|---|
| P1 Live-ready LLM evaluator | **Met, except the live run** | Timeouts, bounded retries, pre-call spend caps, model-run log, strict contract (anchors, counter-evidence, versions, abstention reasons), config hash ⇒ re-evaluation (D2), nonce framing (D4), sanitised popularity text (D3). `avd llm-smoke` refuses without credentials. **No live inference was executed.** |
| P2 Real dataset | **Met** | `real_pilot_v1`: 29 licensed episodes (Changelog CC BY-SA 4.0; Talk Python GPL-3.0) at pinned commits; raw + parsed checksums; split by show (calibration 2 / dev 10 / test 17); datasheet. CI re-fetches and verifies it on every push |
| P2 Annotation | **Workflow met; 0 labels** | Blind assignment, onboarding, 3 raters/item, repeats, verified rater evidence, QC flags, pairwise, adjudication, export, revocation. No rater has annotated anything; no labels were fabricated |
| P2b Benchmark v2 | **Met (framework)** | Grouped bootstrap CIs, QWK, ICC, Krippendorff α, tie-aware pairwise, ECE, risk–coverage/AURC, evidence and genre metrics; unit-tested against published values. 12 gates locked before any annotation. Human-grounded sections report *not evaluable* |
| P3 Long transcripts | **Met (scripted provider)** | Language id, validation, deterministic chunking, map-reduce with timestamps; D1 regression test on a 609k-char transcript |
| P4 Embeddings | **Met** | MiniLM ONNX (sha-pinned, offline after fetch), provenance per vector, mixed-space guard, retrieval report below |
| P5 Genre / claims | **Met** | Genre v2 (distribution, mixed/ambiguous/unknown, overrides that re-run Stage B); claim types; 7-value verdict taxonomy; high-stakes contradictions → review; claims persisted and exposed |
| P6 Recommender v2 | **Met, proxy labels only** | Intent, duration range, explicit eligibility, factual explanations, temporal LTR split. Held-out check uses topical proxy labels, not human judgements |
| P7 Media | **Partially met** | Authorized upload → ffprobe → audio/frames via ffmpeg (sandboxed flags) → audio and visual *signal* metrics, end to end in tests on locally generated media. **No VLM evaluator**: editing and on-screen readability abstain (`FrameJudge` is only an interface) |
| P8 Ops / security | **Met, with documented gaps** | Correlation ids, error taxonomy, rate limits, `/metrics`, `/v1/status`, dead letters + requeue, account deletion, security headers, audits. Gaps in `docs/security.md` (per-process limiter, token expiry, unencrypted media) |
| P9 Extension | **Met** | Assessment states, plain summaries, claims panel, *evidence-checked only* filter, eligibility note, error/timeout/rate-limit states, a11y. 28 vitest tests; screenshots in `docs/screenshots/` |
| P10 E2E + CI | **Met** | `tests/test_e2e_scenario.py` (CLI + API, SQLite and PostgreSQL). CI runs on `claude/**` pushes and pull requests: first run green on all four jobs, 297 backend tests passed with 0 skipped. The extension is not in the automated E2E; it was checked against a live backend with Playwright screenshots |

### 3.2 Measurements

| What | Result | Labels / caveats |
|---|---|---|
| Real pilot, dev split, heuristic (`baseline-heuristic-real_pilot_v1-dev`) | 10/10 evaluated, overall score 10/10, 0 injection flags, popularity invariance passed, p95 71 ms, $0 | Label-free. Genre v2: interview_podcast 8, technology 2 |
| Injection detector v1 → v2 on presumed-benign real transcripts | Held-out flagged episodes 24/104 → 1/104; pilot 5/29 → 0/29 | Recall on real attacks is unmeasured |
| Known-item retrieval, 100 Talk Python transcripts | R@1 hashing 0.28 [0.20, 0.37] → MiniLM 0.71 [0.61, 0.80]; MRR 0.43 → 0.78; RRF fusion 0.57 (worse than MiniLM alone) | Titles share rare words with transcripts; known-item, not recommendation quality |
| Held-out recommendations, pilot test split, 5 personas | nDCG@5 hashing 0.79 vs MiniLM 0.69; P@5 0.40, R@5 0.72 for both | Topical proxy labels (show membership), n tiny, no CI claimed. MiniLM is **not** the default for this reason |
| Synthetic benchmark after genre v2 | Overall ρ 0.59 → 0.55; overall pairwise 7/8 → 6/8; nDCG@5 0.88 → 0.85; genre accuracy 0.85 (unchanged) | Synthetic single-rater labels; reported, not tuned away |
| Throughput / cost (`throughput-real_pilot_v1`) | Measured call plan: Stage B 1 call/video (~18.3k input tokens), Stage C 3 calls. **Projected**: ≈ $0.20/video with Opus 5.5 and 20 % Stage C → ≈ $197/day at 1k videos/day, $1,969 at 10k, $19,694 at 100k ($357 / $3,569 / $35,694 if output is 3× the assumption) | Projections, not measurements; pilot items are ~1 h podcasts, so typical videos are likely cheaper. Heuristic pipeline measured at ~76 ms/video p50 |
| Dependency audits | pip-audit: 0 known vulnerabilities in 76 packages; npm audit: 0 | Also enforced in CI |

### 3.3 Release gates

Gates locked in `release_gates.lock` (sha256 `2d3f4651…`). On the dev split:
popularity invariance, injection benign-flag rate, cost and latency **pass**. Every validity gate
and the human-reliability precondition is **not evaluable** (no human labels). Invalid-citation
rate is **not evaluable** (heuristic mode cites nothing). **Release ready: no.** The held-out
test split has not been used for quality evaluation.

### 3.4 Defects found and fixed during Phase 2 (beyond D1–D11)

* Real content evaluated by the heuristic was labelled *synthetic*; `/v1/meta` called the whole
  system a "synthetic demo" even for licensed content; heuristic justifications said "Synthetic".
* Licensed items got fabricated `youtube.com` URLs.
* Placeholder titles derived from file paths were judged for title consistency.
* The aggregation version string claimed v2 before v2 existed.
* Genre preferences did not reach v2 genres; a genre override did not re-run Stage B.
* Fact-check v1 used old framing and accepted unverified quotes; popularity statements could be
  cited as evidence.
* Manifest path traversal (`../` escaped the dataset cache).
* Candidate listing split paths with spaces, which changed the pilot selection (manifest rebuilt
  and documented).
* Changelog break lines carried timestamps that were being dropped.
* ffprobe rejected `-nostdin`; media-only items were skipped at Stages A/B.
* The cost dry run produced invalid outputs that triggered repair calls, and used a wrong
  output-token key.
* The public video detail omitted attribution and provenance for CC BY-SA sources whose text is
  quoted as evidence.
* Genre status was stored but never returned.
* A Redis dump file had been committed.

### 3.5 Not done / blockers

* **Live LLM evaluation:** needs `AVD_ANTHROPIC_API_KEY` and explicit approval of a budget.
* **Human labels:** needs recruited raters (≥ 3 per item) to annotate the dev split, then the
  test split, through `/annotate`.
* **VLM evaluator** for editing and on-screen readability.
* Multilingual embeddings; shared (Redis) rate limiter; token expiry.
