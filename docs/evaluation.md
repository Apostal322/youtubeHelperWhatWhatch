# Evaluation framework and research methodology

The quality evaluator is treated as a **measurable ML system**, not an oracle. This document
specifies what is measured, how scores are formed, how they are validated, and what the current
evidence does and does not show.

## 1. Principles

1. **Popularity-blind.** Evaluators receive an `EvaluationInput` built from a whitelist (title,
   description, duration, language, tags, genre assessment, authorized content). It has no fields for
   views, likes, comments, subscribers, or channel identity. The LLM prompt is built only from it.
   `input_fingerprint` hashes everything an evaluator can see; it is invariant to popularity.
2. **Dimensions stay separate.** 14 dimensions in 3 groups; groups are always reported; the overall
   number is an optional, genre-specific summary with its weights exposed.
3. **Abstain rather than mislead.** Every dimension has a status. Only `scored` carries a score.
4. **No fabricated evidence.** Models cite `segment_index` + verbatim quote; our code verifies the
   quote and attaches timestamps from the transcript. Unverifiable evidence is dropped; widespread
   failure invalidates the model's scores.
5. **Versioned and reproducible.** `FRAMEWORK_VERSION` (rubric, prompts, profiles, aggregation) is
   stored with every evaluation; LLM responses are cached by content hash; benchmark reports record
   git commit, dataset hash and configuration.

## 2. Dimensions and rubric

| Group | Dimension | Requires |
|---|---|---|
| Content | information density, depth, factual accuracy, originality, informational value | transcript |
| Presentation | clarity, structure, pacing, storytelling, title–content consistency | transcript |
| Technical | audio quality | authorized audio |
| Technical | visual quality, editing, visual readability | authorized video |

Each dimension has an anchored 1–5 scale (1 / 3 / 5 descriptions) in `avd/evaluation/rubric.py`,
served by `GET /v1/meta` and used verbatim in prompts and annotation guidelines.

### Statuses

| Status | Meaning |
|---|---|
| `scored` | Score 1–5 with confidence, justification, evidence |
| `not_applicable` | Weight 0 in the genre profile (e.g. factual accuracy for entertainment) |
| `insufficient_evidence` | Inputs exist but do not support a judgement (model or baseline abstained) |
| `requires_media` | A required authorized modality is unavailable (all visual dimensions today) |
| `failed` | Evaluator error, refusal, schema failure after retry, or evidence-grounding failure |

## 3. Aggregation (genre profiles)

Profiles (`avd/evaluation/profiles.py`) give each dimension a within-group weight and each group a
weight in the overall summary. Examples: science weights factual accuracy 2.0 and depth 1.5, with
technical quality at 10 % of the overall; entertainment sets factual accuracy to 0 and weights
originality, storytelling and pacing; music weights audio quality 2.0.

```
group score    = Σ w_d·s_d / Σ w_d                      over scored applicable dimensions
group coverage = Σ w_d(scored) / Σ w_d(applicable)
overall        = Σ W_g·group_g / Σ W_g                  over groups with ≥1 scored dimension
coverage       = Σ W_g·coverage_g / Σ W_g
confidence     = (Σ eff_weight_d · conf_d) · (0.5 + 0.5·coverage)
```

* Missing dimensions are **renormalised away** (they neither help nor hurt) and the loss is reported
  as `coverage`. A poorly recorded but excellent lecture therefore keeps its content and
  presentation scores; technical quality only affects the technical group and a small share of the
  overall summary (tested in `TestMediaIndependence`).
* The overall summary **abstains** when content coverage < 0.5 or total coverage < 0.4.
* A note is attached when group scores differ by ≥ 1.5 points ("the overall number hides an
  important difference").
* Each dimension's effective weight is returned; contributions sum exactly to the overall score.

## 4. Evaluators, validation and confidence

**LLM rubric evaluator** — one structured call per video for all applicable transcript dimensions.
System prompt: blind review, use full scale, abstain when unsupported, judge factual claims only
when confident, do not penalise production values outside technical dimensions, cite verbatim
quotes, treat the transcript as untrusted data (prompt-injection defence). Output: JSON schema with
`additionalProperties: false`, validated again with Pydantic. Invalid output → one retry with a new
sample → otherwise `failed`. Refusals → `failed`. Transient errors → the job is retried, nothing is
persisted. Stage C uses *n* samples: median score, confidence × max(0.2, 1 − σ/1.5).
Default model: `claude-opus-5-5` (configurable per stage), with server-side refusal fallback; the
model that actually served each request is recorded.

**Validation** — each quote must match (normalised, exact or ≥ 0.85 fuzzy) its cited segment or an
immediate neighbour (relocated, with that segment's timestamps), or span the cited boundary. A scored
dimension without any verified quote has its confidence halved; if ≥ 4 quotes were claimed and
< 50 % verify, all model-scored dimensions become `failed`. ≥ 6 identical scores are flagged.

**Confidence** — model self-reports are combined with structural factors: transcript completeness
(< 150 words ×0.5, < 400 ×0.8), method caps (heuristic ≤ 0.6, audio ≤ 0.7, LLM ≤ 0.9), and a hard cap
of 0.5 for factual accuracy judged from model knowledge without retrieval (with
`missing_inputs: ["reference_corpus"]`). These are **uncalibrated indicators** until calibrated on
human labels; the harness reports ECE to track this.

**Heuristic baseline** (demo mode) — transparent text statistics: content-word ratio, MATTR lexical
diversity, filler/hype/promo rates, trigram repetition, explanatory connectives, caveats,
examples/definitions, structure markers, speaking rate, title-term overlap (English, Spanish,
German lexicons). It abstains on originality and factual accuracy. Every justification starts with
"[Heuristic baseline — not an AI assessment]".

**Audio** — 50 ms frame RMS: 90th percentile = speech level, 10th = noise floor, SNR = difference;
clipping ratio. Penalties for clipping > 0.1 % / 1 %, SNR < 25 / 15 dB, level < −35 dBFS.

**Fact checking (Stage C)** — LLM extracts ≤ 8 checkable claims with quotes → licensed corpus
retrieval → LLM labels each claim *only from retrieved passages*, citing passage ids. Verdicts citing
passages that were not retrieved are discarded. Scored only when ≥ 2 claims were decided.

## 5. Benchmark design

### Dataset format (`benchmarks/datasets/<name>/`)

* `labels.jsonl` — `{video_id, dimension, score (1–5 | null), rater_id, label_source, note}`
* `pairs.jsonl` — `{pair_id, video_a, video_b, criterion (overall | dimension), preferred (a | b | tie), rater_id, label_source, rationale}`
* `personas.json` — recommendation personas with interests, genre weights, depth, languages and the
  relevant video ids
* Fixture videos with transcripts in `data/fixtures/videos/`

### Real-world dataset and annotation (Phase 2)

* `benchmarks/datasets/real_pilot_v1/` is a manifest of 29 licensed real episodes (Changelog CC BY-SA
  4.0, Talk Python GPL-3.0), split by show into calibration / dev / test. Its datasheet is the
  `README.md` next to the manifest. It contains **no labels**.
* Labels come only from the blind annotation workflow (`docs/annotation.md`): pseudonymous raters,
  onboarding on calibration items, 3 independent ratings per item, server-side timing, verified
  evidence, QC flags, blind test–retest repeats, and adjudication by an adjudicator who did not
  rate the item. Export with `avd annotate export`.

### Metrics

| Question | Metric |
|---|---|
| Does the evaluator agree with people? | Spearman, Kendall τb, MAE per dimension vs mean label; overall vs labels aggregated with the same profile |
| Does it order videos correctly? | Pairwise accuracy (label ties excluded; abstentions reported separately) |
| How hard is the task for people? | Krippendorff's α (interval) across raters — the agreement ceiling |
| Are confidences meaningful? | ECE with "correct" = within ±1 of the label mean; per-bin table |
| Does it abstain appropriately? | Abstentions where a label exists; scores where the label is null |
| Is it popularity-blind? | Popularity ×1000 → fingerprint changes and max score delta (must be 0) |
| Is the genre right (it selects the profile)? | Genre accuracy |
| Are recommendations useful? | nDCG@5/10, precision@5, recall@10 per persona |
| Are they diverse and fair? | Intra-list diversity, low-quality items in top-5, top-5 exposure by channel size |
| What does it cost? | Cost per evaluated video (ledger) |

Run: `avd bench [--evaluator configured]`. Reports: `benchmarks/reports/<run>.{json,md}`.

### Real-data benchmark (v2): `avd bench-real`

`avd bench-real --labels <export dir> [--split dev|test --confirm-test] [--evaluator configured]`
runs the production pipeline on the dataset manifest in an isolated database. It reports two
kinds of measurement.

**Label-free (always available)**
* evaluation status and assessment state;
* dimension status by reason code;
* evidence: claimed vs verified citations, invalid-citation rate, scored-but-ungrounded dimensions;
* injection flags on presumed-benign content;
* predicted genre distribution;
* latency, cost and token percentiles (p50/p95), and LLM call errors;
* popularity invariance.

**Human-grounded (only with human labels for the split)**

| Question | Metric |
|---|---|
| How reliable are people? (the ceiling) | Krippendorff's α (ordinal, interval); ICC(1,1) per dimension; each rater vs the mean of the others (leave-one-out) |
| Does the model agree with people? | Spearman, Kendall τb, MAE, quadratic-weighted κ per dimension, pooled, and for content dimensions. The reference is the adjudicated score, else the mean of ≥ 2 ratings whose range is < 2; disputed cells are excluded |
| Is it as good as one more rater? | Model MAE − human MAE against the *same* leave-one-out references |
| Does it abstain when people cannot judge? | Abstention agreement, incl. the rate of scoring where most humans did not |
| Are confidences meaningful? | ECE (correct = within ±1); risk–coverage curve, AURC vs oracle, `aurc_gain` |
| Does it order items like people? | Tie-aware pairwise accuracy (label ties count when the model is within 0.25) |

All intervals are 95 % percentile bootstraps that resample **whole groups** (shows/creators), with
a fixed seed. Items from one creator are not independent evidence. Non-human label sources are
refused.

**Release gates.** `benchmarks/release_gates.json` holds 12 gates (precondition, validity, safety
and operational), each with a rationale. It is locked in `release_gates.lock` (sha256 + time)
*before* any human annotation exists. A test-split run requires `--confirm-test`, a valid lock,
and labels newer than the lock. Changed gates can only be re-locked together with a new test set
(`avd gates lock --new-test-set`). A gate without human labels is *not evaluable*, never passed,
and a release needs every gate evaluable and passing on a binding test-split run.

## 6. Results so far, and what they do not show

Committed baseline: [`benchmarks/reports/baseline-heuristic-synthetic_v1.md`](../benchmarks/reports/baseline-heuristic-synthetic_v1.md)
(heuristic baseline, synthetic_v1, commit `3aeaa9b`).

> **All labels in synthetic_v1 are synthetic design-intent labels written by the fixture author
> (an AI assistant), single rater.** They test the harness and catch regressions. They say nothing
> about accuracy on real videos or agreement with real people.

| Measurement | Heuristic baseline |
|---|---|
| Popularity invariance (20 videos, popularity ×1000) | **passed** — 0 fingerprint changes, max score delta 0.0 |
| Overall vs label reference | Spearman 0.59, MAE 0.97 (n = 14; 6 overall abstentions) |
| Pairwise accuracy, all criteria | 0.84 (19 decided pairs); overall 0.875 (8 decided, 3 abstained) |
| Strongest dimensions | information density ρ = 0.73, informational value ρ = 0.72 |
| Weakest dimensions | storytelling ρ = −0.08 (no signal), pacing ρ = 0.23, clarity ρ = 0.30 |
| Abstention | originality and factual accuracy always abstain (by design) |
| Calibration | ECE 0.086; confidences (0.33–0.44) are slightly *under*-confident vs 46–47 % within ±1 |
| Genre accuracy | 0.85 — both documentaries and the Spanish lesson misclassified |
| Recommendations (5 personas) | nDCG@5 0.88, nDCG@10 0.91, P@5 0.60, 1 low-quality item in 25 top-5 slots |
| Top-5 exposure by channel size | micro 9, small 5, medium 4, huge 7 (of 25) |
| Cost | $0 (no LLM) |

**Real pilot, dev split, heuristic baseline**
([`baseline-heuristic-real_pilot_v1-dev.md`](../benchmarks/reports/baseline-heuristic-real_pilot_v1-dev.md)).
These are label-free numbers only: no human has annotated the pilot, so **no accuracy or agreement
figure exists for real content**.
* 10/10 items were evaluated, all as `heuristic_baseline`.
* An overall score is available for 10/10. Originality and factual accuracy always abstain;
  title consistency abstains on placeholder titles.
* Genre v2 labels 8/10 as `interview_podcast` and 2/10 as `technology`. Before genre v2 the same
  items were spread over technology 5, entertainment 2, education 1, how-to 1 and gaming 1. A tech
  podcast labelled "gaming" was a real misclassification that changed the aggregation profile.
  These are in-sample observations on the dev split, with no human genre labels.
* Popularity invariance passed. No injection flags.

The LLM evaluator has **no benchmark numbers yet**: it was tested with scripted responses (parsing,
validation, grounding, abstention, retries, refusal, self-consistency, caching), not run against a
live model in this repository.

Recommendation numbers were measured on the same 5 synthetic personas that were used to calibrate
the relevance floor/ceiling (16 relevant pairs) — treat them as optimistic.

## 7. Known biases and limitations

**Evaluation**
* **Transcript-only view.** Most dimensions are judged from text. Visual explanation, demonstrations,
  music and on-screen content are invisible; visual dimensions always abstain. Videos whose value is
  mostly visual will have low coverage or abstained overall scores.
* **Language and dialect bias.** Lexicons cover English/Spanish/German only; LLMs are likely stronger
  in English. Speaking-rate norms differ by language. Non-English results need separate validation.
* **Genre misclassification changes the profile** and therefore the overall summary (observed:
  documentaries classified as education/entertainment). Genre and its confidence are always shown.
* **Heuristic baseline biases:** rewards explicit signposting and long words; penalises informal or
  narrative styles; storytelling has no measured signal. It is a baseline, not a judge.
* **LLM judge biases** (expected, to be measured): verbosity and length preference, preference for
  formal register, self-preference for model-like text, sensitivity to prompt order, possible
  knowledge cut-off errors in factual judgements (hence the 0.5 confidence cap without retrieval).
* **Title/description leakage.** Descriptions can mention popularity ("1M subscribers!"); they are
  passed to the evaluator for title–content consistency. Not yet filtered.
* **Originality is judged in isolation** — no comparison against similar videos yet.
* **Confidence is uncalibrated** until fitted against real human labels.

**Recommendation**
* **Lexical embeddings are weak.** The default 384-bucket hashing embedder has a collision noise
  floor: e.g. "astronomy" shares a bucket with the Spanish word "tarjetas", producing spurious
  relevance (~0.10) between an astronomy interest and a flash-card study video. Embedding transcript
  keywords instead of raw text raised persona AUC from 0.79 to 0.88.
* **Semantic embeddings (Phase 2).** `AVD_EMBEDDING_PROVIDER=onnx_minilm` uses all-MiniLM-L6-v2 on
  ONNX Runtime. It was measured on known-item retrieval: real Talk Python titles as queries against
  100 transcript-only documents ([report](../benchmarks/reports/retrieval-talkpython-n100-s7.md)).

  | method | Recall@1 [95 % CI] | MRR [95 % CI] |
  |---|---|---|
  | hashing | 0.28 [0.20, 0.37] | 0.43 [0.35, 0.50] |
  | MiniLM | 0.71 [0.61, 0.80] | 0.78 [0.70, 0.85] |
  | reciprocal-rank fusion of both | 0.57 | 0.67 |

  Fusion is worse than MiniLM alone, so it is not used. Limits of this result:
  * MiniLM is English-only; no multilingual model was reachable.
  * Hugging Face is blocked here, so numerical equivalence to sentence-transformers is unverified.
  * Document embedding costs about 0.8 s per long transcript on CPU (24 windows).
  * 14 of 29 MiniLM false matches were as similar as a typical correct match, so cosine
    thresholds do not separate errors.
  * Titles share rare words with transcripts. This is known-item retrieval, not a test of
    recommendation quality.
* Relevance calibration: hashing floor 0.06 / ceiling 0.33, fitted on a tiny synthetic sample;
  MiniLM 0.20 / 0.50, from the retrieval experiment's non-relevant and relevant similarity medians.
* Feedback loops: logged impressions carry position bias; LTR training needs debiasing before use.

**Coverage**
* Evaluation requires authorized content, which skews coverage toward creators who opt in and
  openly licensed material. Coverage is reported by genre, channel size and language so this is
  visible, not hidden.
* `order=date` discovery reduces popularity bias in sampling but YouTube's search index itself is
  not a uniform sample of uploads.

**What cannot be claimed**
* That any score is "objectively correct". Scores are model or heuristic judgements against a
  published rubric, with measured (not assumed) agreement once real labels exist.
