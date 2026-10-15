# Roadmap

## Recommended next milestone: **M1 — real-signal validation**

Before scaling anything, establish whether the evaluator agrees with people.

1. **Collect real human labels** for 150–300 videos with authorized transcripts (creator partners,
   CC-licensed lecture/course datasets), stratified by genre × channel size × language.
   *At least 3 raters per video*, using the rubric anchors from `/v1/meta` and pairwise judgements
   for close calls. Add an annotation UI (the `feedback` table already captures dimension-level
   disagreement reports from users).
2. **Run the LLM evaluator** (`AVD_LLM_PROVIDER=anthropic`) on that set via `avd bench --evaluator
   configured`; compare against the heuristic baseline and against inter-rater agreement (the
   ceiling).
3. **Calibrate confidence** (isotonic or temperature scaling per dimension) using the ECE bins the
   harness already reports; publish the calibration with the framework version.
4. **Enable semantic embeddings** (`sentence_transformers`, multilingual model) and re-measure
   recommendation metrics; the lexical baseline's collision floor is a known limitation.
5. Promote the framework version only when the benchmark shows improvement on held-out labels.

## M2 — Content coverage

* Creator OAuth flow (consent, encrypted token storage, revocation handling) → `CreatorCaptionsProvider`.
* Licensed-corpus connectors (e.g. openly licensed lecture series) through `import-dataset`.
* Chunked (map-reduce) evaluation for transcripts beyond the single-call limit instead of failing.
* Audio from authorized sources in more formats (currently 16-bit PCM WAV).

## M3 — Scale-out (10⁵–10⁶ videos)

* Message Batches API for Stage B (≈50 % cost) with the existing cache + ledger.
* Separate worker pools per stage with autoscaling on queue depth; partition `pipeline_jobs` by
  status or move hot queue state to a dedicated queue table with archival.
* Quota-aware discovery planner that allocates daily YouTube units across genre × language ×
  channel-size strata against coverage targets (`/v1/admin/coverage` already measures the gaps).
* Model cascade measured by the harness: cheaper model first, escalate only on low confidence
  (the Stage B/C split already supports it).
* Observability: OpenTelemetry traces per job, Prometheus metrics for queue depth, cost/hour,
  abstention rate, validation failure rate.

## M4 — Better models

* Multimodal evaluator for visual dimensions (authorized video only).
* Specialist evaluators (e.g. a dedicated fact-check agent with broader licensed retrieval,
  originality via similarity to the analyzed catalog).
* Learning-to-Rank trained on logged impressions (`avd train-ranker`), with interleaving or A/B
  evaluation; counterfactual correction for position bias.
* Per-user calibration of dimension priorities from feedback.

## M5 — Product

* Account linking and sync; shareable curated lists.
* Creator dashboard: their evaluations, evidence, and a dispute flow.
* Public methodology page generated from `FRAMEWORK_VERSION`, profiles and benchmark reports.
