# Roadmap

## Recommended next milestone: **M1 — real-signal validation**

Before scaling anything, establish whether the evaluator agrees with people. Phase 2 built the
tooling; what is missing is people and a live model run. Status per step:

1. **Collect real human labels.** *Tooling done* (`docs/annotation.md`: blind assignment,
   calibration onboarding, 3 raters/item, repeats, QC, adjudication, export). *Data started*:
   `real_pilot_v1` has 29 licensed episodes (dev 10 / test 17 / calibration 2). **Open**: recruit
   raters (with rater consent and pay) and annotate the dev split first; then grow the corpus to
   150–300 items stratified by genre × length × language. It is currently English-only, all
   podcast/interview, and from 2 publishers.
2. **Run the LLM evaluator** on the dev split (`avd bench-real --evaluator configured`) once a key
   and budget are approved. Use `avd bench-throughput` for the expected cost first. Compare it with
   the heuristic baseline and with human–human agreement (the ceiling).
3. **Calibrate confidence** per dimension on dev labels (isotonic or temperature scaling); the
   benchmark already reports ECE and risk–coverage.
4. **Run the test split once**, with `--confirm-test`, against the gates locked in
   `benchmarks/release_gates.lock`. Promote the framework version only if every gate is evaluable
   and passes.
5. **Embeddings.** English MiniLM is done (known-item R@1 0.71 vs 0.28). It is *not* the default,
   because the held-out recommendation proxy favoured hashing (nDCG@5 0.79 vs 0.69, tiny n).
   Re-decide with human relevance labels. Still open: a multilingual model.

## M2 — Content coverage

* Creator OAuth flow (consent, encrypted token storage, revocation handling) → `CreatorCaptionsProvider`.
* Licensed-corpus connectors (e.g. openly licensed lecture series) through `import-dataset`.
* Authorized media upload with ffmpeg probing/decoding is done (Phase 2); still open: resumable
  uploads, per-user quotas, encryption at rest, malware scanning.

## M3 — Scale-out (10⁵–10⁶ videos)

* Message Batches API for Stage B (≈50 % cost) with the existing cache + ledger.
* Separate worker pools per stage with autoscaling on queue depth; partition `pipeline_jobs` by
  status or move hot queue state to a dedicated queue table with archival.
* Quota-aware discovery planner that allocates daily YouTube units across genre × language ×
  channel-size strata against coverage targets (`/v1/admin/coverage` already measures the gaps).
* Model cascade measured by the harness: cheaper model first, escalate only on low confidence
  (the Stage B/C split already supports it).
* Observability: correlation ids and Prometheus `/metrics` exist (Phase 2); still open:
  OpenTelemetry traces per job, abstention and validation-failure rates as metrics, and a shared
  (Redis) rate limiter for multi-worker deployments.

## M4 — Better models

* Vision-language evaluator for editing and on-screen readability (authorized video only),
  behind the existing `FrameJudge` interface; visual *signal* metrics already exist.
* Specialist evaluators (e.g. a dedicated fact-check agent with broader licensed retrieval,
  originality via similarity to the analyzed catalog).
* Learning-to-Rank trained on logged impressions (`avd train-ranker`), with interleaving or A/B
  evaluation; counterfactual correction for position bias.
* Per-user calibration of dimension priorities from feedback.

## M5 — Product

* Account linking and sync; shareable curated lists.
* Creator dashboard: their evaluations, evidence, and a dispute flow.
* Public methodology page generated from `FRAMEWORK_VERSION`, profiles and benchmark reports.
