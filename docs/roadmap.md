# Roadmap

## Recommended next milestone: **M1, real-signal validation** (Phase 3 made it runnable)

Phase 3 removed the engineering blockers. What remains needs people and a budget:

1. **Experiment A: first live run** on the 10 dev items. Needs a key and an approved ceiling;
   $5 covers the projected worst case of $4.14. `docs/model-configuration.md`.
2. **Experiment B: human labels** on the dev split. Needs 3 raters and 1 adjudicator, about
   25–40 rater-hours. `docs/annotation.md`, "Recruiting".
3. **Experiments C–E: agreement, prompt sensitivity, model comparison** on dev. Tune only on dev.
   Then lock one candidate (model, prompt key, rubric, aggregation).
4. **Experiment I: one confirmatory run** on `real_holdout_v2` (sealed) with `--confirm-test`
   against the locked gates.
5. **Data breadth:** licensed corpora beyond English software podcasts (lectures, tutorials,
   documentaries; other languages) before any cross-genre claim. The Talk Python rights
   confirmation unblocks 7 test items.
6. **Recommendation relevance labels** (topical search, related items, personalised relevance)
   before changing the embedding default. The dev diagnostic suggests the relevance floor and
   blend, not the embedder, are the weak point.

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

## Phase 4: multimodal (after M1)

See `docs/phase3.md` §8: authorised media sources, audio signal validation, a vision-language
`FrameJudge` with frame-index evidence, genre-specific multimodal rubrics, and the same blinding,
gate and hold-out discipline.

## Phase 5: continuous scale

See `docs/phase3.md` §9:
- acquisition through creator opt-in and licensed corpora (YouTube quota and caption OAuth are
  the binding limits);
- cheap-first routing validated against human agreement;
- batching, caching, media isolation and coverage audits.
