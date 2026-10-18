# Human annotation

The workflow that produces the **only** labels this project treats as ground truth for real
content. It ships empty: no rater has annotated anything yet, and no labels are generated or
imported from model output.

## Roles

| Role | Can | Cannot |
|---|---|---|
| Rater (`role=rater`) | annotate assigned items; make pairwise comparisons of items they annotated | see other raters' scores, the system's evaluation, or popularity data |
| Adjudicator (`role=adjudicator`) | annotate; resolve disagreements on items **they did not rate** | adjudicate their own items |
| Admin (`X-Admin-Token`) | create raters, view QC, exclude annotations (with a reason), export | edit a rater's scores |

Raters are pseudonymous. Their tokens (`rtr_…`) are shown once and stored hashed. No personal
data beyond the pseudonym and the self-declared expertise tags is collected.

## Workflow

1. **Create raters.** `avd annotate create-rater alice --expertise software`. The token is
   printed once to stderr.
2. **Onboarding.** A new rater (`status=onboarding`) receives only the calibration items (split
   `calibration`). After all of them are annotated, the rater becomes `active`. Calibration
   annotations are for training and discussion and are never exported as labels.
3. **Blind assignment** (`GET /v1/annotation/next?dataset=real_pilot@v1`). Each rater gets the
   dev/test item with the fewest independent ratings, in a per-rater pseudo-random order, until
   every item has 3 ratings (`RATERS_PER_ITEM`). A rater never gets the same item twice, except
   for repeats.
4. **Repeats (test–retest).** Once a rater has 5 submissions, every 6th assignment is a blind
   repeat of an item they already rated. Repeats are exported separately and never count as
   independent ratings.
5. **Annotating** (UI at `/annotate`). For each of the 10 transcript dimensions:
   * pick a score 1–5 using the anchors (all five levels are written out), **or** choose a
     status: `insufficient_evidence`, `not_applicable`, `ambiguous` or `abstain`. Never guess.
     An unavailable dimension is not a low score;
   * give a confidence from 0 to 1;
   * cite evidence by selecting words in a segment and pressing **Cite**. Citations can be
     marked as *counter* evidence. The server verifies every quote against the transcript
     (same matcher as model evidence) and rejects quotes it cannot find;
   * optional note. Also give an overall confidence and set any flags (`ambiguous_rubric`,
     `insufficient_material`, `transcript_problem`, `sensitive_content`).
6. **Timing.** Start and submit times are taken by the server. `duration_seconds` is not
   self-reported.
7. **QC flags** are added automatically and recorded, not acted on:
   * `too_fast`: the submission came faster than reading the transcript at 1,000 words per
     minute;
   * `straight_lining`: 5 or more scored dimensions, all with the same score;
   * `missing_evidence:<dimension>`: a score with no citation.

   Admins review `GET /v1/annotation/qc`. Excluding an annotation requires a written reason and
   is logged on the row.
8. **Disagreement.** A dimension on an item is disputed when its independent ratings span 2 or
   more points, or when some raters scored it and others abstained or marked it not applicable.
   Disputed cells appear in `GET /v1/annotation/adjudication-queue`. An adjudicator who did not
   rate the item decides `score` (1–5), `exclude` or `ambiguous`, with a mandatory rationale.
   Original ratings are kept.
9. **Pairwise.** `GET /v1/annotation/pairs/next?criterion=overall|content|presentation` offers
   two items the rater has annotated, from the same split, in randomised order. The answer is
   `a`, `b`, `equal` or `insufficient`, with a rationale.
10. **Export.** `avd annotate export --dataset real_pilot@v1 --out <dir>` writes these files:
    * `labels.jsonl`: one row per independent rating (`label_source=human_annotation`), plus
      adjudicated rows (`human_adjudication`);
    * `repeats.jsonl`;
    * `pairs.jsonl`;
    * `items.jsonl`: split, group and stratum, for grouped bootstrap.

    Every row records the rubric version, split and group.

## Rater guidance (shown in the UI)

* Judge the **transcript**, not the creator's reputation, popularity or production polish.
  Presentation dimensions such as clarity or pacing are judged from what the text shows. Say
  `insufficient_evidence` if the text cannot show it.
* Each dimension is independent. Poor structure does not lower depth.
* Do not use outside information about the episode (views, comments, reviews).
* Factual accuracy: score it only for claims you can check with confidence. Otherwise use
  `insufficient_evidence`. This is a rating of the content, not a fact-check.
* Title consistency on items whose title is derived ("Show #N") is `not_applicable`.
* Timestamps on Changelog items mark source spans of about one minute, so several paragraphs
  share one span.

## What the workflow does not provide

* **Rater recruitment, payment and consent.** These are organisational decisions. A consent
  text and pay rate must be approved before anyone is recruited.
* **Expertise verification.** Expertise tags are self-declared.
* **Agreement thresholds.** Agreement is computed by the benchmark (`docs/evaluation.md`).
  Release gates are in `benchmarks/release_gates.yaml`, fixed before any test-split labels exist.

## Cost estimate (pilot)

27 dev/test items × 3 raters = 81 annotations, plus 2 calibration items per rater and about 15 %
repeats. At 45–75 minutes per item this is roughly **60–100 rater-hours** for the full pilot.
Annotating only the dev split first (10 items, about 25–40 hours) is a reasonable feasibility
check before committing to the test split.
