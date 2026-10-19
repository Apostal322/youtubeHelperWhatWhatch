# Research protocol: can a local open-weight model evaluate transcripts?

**Status:** protocol only. No local model has been run yet. This page fixes the questions,
metrics and decision rules *before* any results exist, so later results cannot shape them.

## Questions

1. **Contract.** Does a local model (Qwen3.5 4B/9B, and 27B / 35B-A3B where hardware allows)
   produce schema-valid, evidence-grounded rubric judgements through the production pipeline?
2. **Grounding.** How often are cited quotes real (verified at the cited segment), and how often
   are dimensions scored without verified evidence?
3. **Reliability.** Are scores stable across repeats (seeds) and across context windows
   (single-pass vs hierarchical)? Is the ranking of item pairs stable?
4. **Robustness.** Does an injected instruction move scores? Does the model abstain when there is
   nothing to judge?
5. **Cost of running.** Latency, throughput (items/hour), peak memory, CPU/GPU load on the
   available hardware.
6. **Validity** (requires human labels; blocked today). How well do local scores agree with
   independent human raters, compared with the heuristic baseline and, if ever run, a hosted
   model?

## Data

* `real_pilot_v1` **dev** split only: 10 items cleared for internal analysis.
* `real_pilot_v1` test (partly exposed) and `real_holdout_v2` (sealed) are **not used**.
  `avd bench-local` refuses them. A single confirmatory run on held-out data needs a locked
  candidate configuration and `avd evaluate-real --confirm-test`.
* No prompt, threshold or chunking parameter may be tuned on anything but dev data.

## Configurations

* Prompt `rubric_prompt_v2` (default), rubric `rubric_v2`, Stage B only.
* `num_ctx` ∈ {8192, 16384}: 8k forces hierarchical evaluation on most items, so the two windows
  form the context-length comparison.
* 2 repeats with seeds 0 and 1 (`ollama_seed`), temperature at the model default.
* `think=false`. Thinking mode is a separate, later comparison, because it changes latency by
  an order of magnitude.
* Each run records the model fingerprint (tag, digest, quantization), the Ollama version, the
  sizing in force, the hardware and the git commit.

## Metrics (all from `avd bench-local`)

| Metric | Definition |
|---|---|
| Structured validity | (calls − output errors) / calls |
| Citation correctness | verified quotes / claimed quotes |
| Invalid-citation rate | 1 − citation correctness |
| Ungrounded dimensions | scored dimensions without verified evidence |
| Abstention rate | non-scored LLM dimensions / all LLM dimensions |
| Repeat consistency | exact agreement and mean absolute difference of per-dimension scores across repeats; scored-vs-abstained flips |
| Pairwise stability | share of item-pair orderings (by overall score) preserved across repeats |
| Injection effect | mean score shift, dimensions pushed to 5, injected segment cited |
| Abstention probe | text dimensions abstained on a contentless transcript |
| Throughput | p50/p95 call latency, output tokens/s, items/hour |
| Resources | peak Ollama RSS, peak VRAM, mean CPU and GPU utilisation |

Citation correctness checks that a quote *exists* at the cited segment, not that it *supports*
the score. Support is audited separately (the Phase 3 support audit) and ultimately needs people.

## Decision rules (fixed in advance)

These thresholds are proposed defaults. The project owner may revise them, but only before
the first run.

A local model is **usable for operational trials** (not for publication of scores as validated)
if, on dev:

* structured validity ≥ 0.95;
* citation correctness ≥ 0.90 and no more than 1 ungrounded scored dimension per 10 items;
* the injection probe moves the mean score by < 0.5 and the injected segment is never cited;
* at least 80% of text dimensions abstain on the contentless transcript;
* repeat exact agreement ≥ 0.6 and pairwise stability ≥ 0.8.

Among usable models, prefer the smallest one within 0.05 on every rate above, unless its
throughput is too low for the workload.

**No validity claim** is made from any of these numbers. The release gates on human agreement
stay *blocked* until labels exist (`avd gates status`). Agreement with the heuristic baseline
is reported only as a sanity check.

## Human comparison (when labels exist)

The 10-item dev experiment in [`../local-ai.md`](../local-ai.md#7-human-evaluation-of-local-results-10-dev-items):

* blind rating by at least 2 raters, then export;
* `avd bench-real --labels` for weighted κ, ICC and Spearman with bootstrap CIs, per model;
* inter-rater agreement is the ceiling against which model–human agreement is read.

Ten items give wide intervals. This is a feasibility check, not a validation.

## Threats to validity

* **Small n**: 10 items, few genres (podcast episodes); intervals will be wide.
* **Prompt written for hosted models**: the prompt was not tuned for Qwen; a local model may be
  disadvantaged. A local-specific variant would be a registered, separately evaluated prompt.
* **Quantization**: q4 may degrade judgement relative to the same model at q8 or bf16. Where
  memory allows, run q8 as a sensitivity check.
* **Hardware dependence**: throughput and memory numbers hold only for the recorded hardware.
* **Estimated sizes**: `avd local doctor` sizes are estimates until a model is installed.
