# Real-data benchmark — real_pilot@v1 (dev split) — heuristic

> **Warning:** Evaluator is the HEURISTIC baseline (keyword/structure features), not an AI model. No live LLM was used in this run.
> **Warning:** No human labels for the dev split: model-human and human-human agreement are not evaluable. Nothing here measures accuracy against people.

## Reproducibility

| key | value |
|---|---|
| run_at | `2026-10-16T01:35:11+00:00` |
| git_commit | `2507f77d7fffc65b9bc17ecaa007e522942b78a7` |
| git_dirty | `False` |
| dataset | `real_pilot@v1` |
| split | `dev` |
| items | `10` |
| manifest_items_sha256 | `19039096dcc201ac06bc81b076a2a93d29a3137781371722eeb0ea5f88ae20c0` |
| labels_sha256 | `None` |
| human_label_rows | `0` |
| evaluator | `heuristic` |
| llm_provider | `heuristic` |
| pipeline_config_hash | `463df1cd5bb086318ad094c96de5df8274cfe5866d6af2df0c9b0411984c14df` |
| seed | `0` |
| n_boot | `2000` |
| runtime_seconds | `1.39` |
| gates_file_sha256 | `2d3f4651056975909d498fc0c22521b8b32169a813f083403e3f7e3a26c7016a` |
| gates_lock_valid | `True` |
| python | `3.12.3` |
| versions | `{'framework': '2026.10.1', 'rubric': 'rubric_v2', 'prompt': 'rubric_prompt_v2', 'aggregation': 'genre_weighted_mean_renormalized_v2', 'preprocessing': 'prep_v2', 'injection_detector': 'injection_v2'}` |

## Label-free measurements

* Items evaluated: 10/10; status {'complete': 10}; assessment state {'heuristic_baseline': 10}
* Overall score available: 1.0; needs review: 0.0
* Evidence: {'citations_claimed': 0, 'citations_verified': 0, 'invalid_citation_rate': None, 'ungrounded_scored_dimensions': 0}
* Injection flags (presumed-benign data): 0 items, 0 findings
* Predicted genre: {'interview_podcast': 8, 'technology': 2}
* Latency (ms): {'p50': 57.0, 'p95': 71.4}; cost (USD): {'total': 0.0, 'per_video_mean': 0.0, 'p50': 0.0, 'p95': 0.0}; tokens/video: {'input': {'p50': 0.0, 'p95': 0.0}, 'output': {'p50': 0.0, 'p95': 0.0}}; LLM calls: {'total': 0, 'errors': 0, 'cached': 0}
* Popularity invariance: PASSED ({'videos_checked': 10, 'fingerprint_changes': 0, 'max_score_delta': 0.0, 'passed': True})

### Dimension status (scored / abstained:reason)

* clarity: {'scored': 10}
* depth: {'scored': 10}
* factual_accuracy: {'insufficient_evidence:method_cannot_judge': 10}
* information_density: {'scored': 10}
* informational_value: {'scored': 10}
* originality: {'insufficient_evidence:method_cannot_judge': 10}
* pacing: {'scored': 10}
* storytelling: {'scored': 10}
* structure: {'scored': 10}
* title_consistency: {'insufficient_evidence:insufficient_material': 10}

## Human-grounded measurements

**Not available:** no human labels for this split. No model–human or human–human statistic is reported.

## Release gates (informational; release ready: False)

| gate | kind | status | detail |
|---|---|---|---|
| human_reliability_floor | precondition | not_evaluable | no human labels |
| content_spearman_lower_ci | validity | not_evaluable | no human labels |
| mae_noninferiority_vs_single_human | validity | not_evaluable | no human labels |
| pairwise_lower_ci | validity | not_evaluable | no human labels |
| calibration_ece | validity | not_evaluable | no human labels |
| selective_risk_informative | validity | not_evaluable | no human labels |
| abstention_respects_humans | validity | not_evaluable | no human labels |
| invalid_citation_rate | safety | not_evaluable | n=0 < min_n=20 |
| popularity_invariance | safety | pass | True == True |
| injection_benign_flag_rate | safety | pass | 0.0 <= 0.05 |
| cost_per_video | operational | pass | 0.0 <= 0.5 |
| latency_p95 | operational | pass | 71.4 <= 300000 |
