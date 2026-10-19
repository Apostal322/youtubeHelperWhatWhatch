# real_holdout_v2: sealed confirmatory hold-out

12 unseen Changelog episodes: The Changelog ×4, JS Party ×4, Ship It! ×4. All are CC BY-SA 4.0,
with an explicit content licence (see `avd dataset rights`). Built on 2026-10-19 with
`avd dataset build-holdout`, seed 20261019, under the same eligibility rules as `real_pilot_v1`.

**Why it exists.** Three Phase 2 decisions looked at outputs on the `real_pilot_v1` test split
(see `../real_pilot_v1/exposure.json`):
- genre v2 development;
- injection-detector v2 false-positive tuning;
- the choice of embedding default.

Confirmatory claims about those components need data that played no part in any development
decision.

**Composition and limits**
- The four test-split shows of v1 were targeted. Brain Science had no remaining episode with at
  least 15 source timestamps, so it contributed 0 of its quota of 4. The eligibility rules were
  not relaxed after seeing this.
- Same publisher, format and language as the pilot: English interview podcasts, mostly about
  software. It cannot support cross-genre or cross-language claims.
- No Talk Python items, because of the rights confirmation pending for that source.

**Rules (enforced by `avd bench-real --confirm-test`, with locked gates)**
1. Do not import, evaluate, annotate or read these transcripts until both are frozen in a
   committed lock: a candidate (model, prompt version, rubric version, aggregation) and the gates.
2. Evaluate once. Report every result, including failures.
3. Any change after seeing results makes this set exposed, and a new hold-out is required.

Building the manifest read each transcript only to compute eligibility statistics (word count,
timestamps, language). No evaluator, classifier or detector has been run on these items.
