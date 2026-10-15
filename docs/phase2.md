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
| D5 | CI has never run | Workflow triggers only on `main` pushes and pull requests; no PR exists |
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

Release gates for the evaluator are defined in `benchmarks/release_gates.yaml` **before** any held-out
test-split evaluation is run.

## 3. Results

Filled in as milestones complete (see sections below).
