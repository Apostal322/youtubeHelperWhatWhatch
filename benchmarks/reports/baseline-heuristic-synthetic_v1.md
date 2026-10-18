# Benchmark report — synthetic_v1 — heuristic

> **Warning:** Labels are SYNTHETIC design-intent labels written by the fixture author, not human annotations. These numbers test the harness and catch regressions; they say nothing about real-world accuracy.
> **Warning:** Single rater: inter-rater agreement cannot be estimated.

## Reproducibility

| key | value |
|---|---|
| run_at | `2026-10-16T01:35:34.186487+00:00` |
| framework_version | `2026.10.1` |
| evaluator | `heuristic` |
| evaluation_mode | `synthetic_demo` |
| llm_provider | `heuristic` |
| models | `[]` |
| embedding_model | `hashing-uni-bigram-v1` |
| git_commit | `7a2ace4279b6d42310fcfb5f9c226daf1e50aa12` |
| dataset_sha256 | `68d437566906710c319081b0bea2bba8132e75dfca9750cf09ca2b3f0da3b8f4` |
| fixtures_sha256 | `3f2891d6cb954df5c96f25283bec9048c39c2b7518d7d416a71576ea952b2137` |
| label_sources | `['synthetic_design_intent']` |
| python | `3.12.3` |

## Per-dimension agreement with labels

| dimension | n | Spearman | Kendall τb | MAE | abstained (label exists) | scored (label null) |
|---|---:|---:|---:|---:|---:|---:|
| information_density | 18 | 0.7268 | 0.5637 | 1.2761 | 0 | 1 |
| depth | 20 | 0.452 | 0.3117 | 1.706 | 0 | 0 |
| factual_accuracy | 0 | None | None | None | 13 | 0 |
| originality | 0 | None | None | None | 20 | 0 |
| informational_value | 18 | 0.722 | 0.5507 | 1.1294 | 0 | 2 |
| clarity | 20 | 0.3028 | 0.2362 | 1.3685 | 0 | 0 |
| structure | 20 | 0.4849 | 0.3385 | 1.2725 | 0 | 0 |
| pacing | 20 | 0.2291 | 0.1878 | 1.5395 | 0 | 0 |
| storytelling | 20 | -0.0814 | -0.0705 | 1.4485 | 0 | 0 |
| title_consistency | 20 | 0.3089 | 0.2749 | 0.9715 | 0 | 0 |
| audio_quality | 2 | None | None | 0.5 | 0 | 0 |
| visual_quality | 0 | None | None | None | 0 | 0 |
| editing | 0 | None | None | None | 0 | 0 |
| visual_readability | 0 | None | None | None | 0 | 0 |

**Overall vs label reference** (labels aggregated with the same genre profiles): n=14, Spearman=0.5473, MAE=0.9529, overall abstentions=6

## Pairwise ranking accuracy

| criterion | accuracy | decided | abstained |
|---|---:|---:|---:|
| overall | 0.75 | 8 | 3 |
| factual_accuracy | None | 0 | 2 |
| depth | 0.5 | 2 | 0 |
| storytelling | 1.0 | 3 | 0 |
| title_consistency | 1.0 | 2 | 0 |
| originality | None | 0 | 5 |
| information_density | 1.0 | 2 | 0 |
| audio_quality | 1.0 | 1 | 0 |
| structure | 0.0 | 1 | 0 |
| **all** | 0.7895 | 19 | |

## Calibration

ECE (|model - label| <= 1): **0.0858**

| confidence bin | n | mean confidence | accuracy |
|---|---:|---:|---:|
| [0.2, 0.4] | 82 | 0.333 | 0.463 |
| [0.4, 0.6] | 76 | 0.436 | 0.474 |

**Inter-rater agreement:** raters=1, Krippendorff α (interval) = None

**Evaluation status counts:** {'complete': 14, 'partial': 6}

## Genre classification

accuracy 0.85 (n=20); errors: syn_doc_001→education (expected documentary), syn_doc_002→entertainment (expected documentary), syn_edu_002→howto (expected education)

## Popularity invariance

views/likes/subscribers ×1000 on 20 videos → fingerprint changes: 0, max score delta: 0.0 → **PASSED**

## Recommendation (persona profiles)

| persona | nDCG@5 | nDCG@10 | P@5 | R@10 | ILD@10 | low-quality in top5 |
|---|---:|---:|---:|---:|---:|---:|
| curious_science_generalist | 0.8855 | 0.8855 | 0.6 | 1.0 | 0.9844 | 0 |
| cs_math_student_advanced | 0.7654 | 0.9134 | 0.4 | 1.0 | 0.9973 | 1 |
| estudiante_hispanohablante | 0.906 | 0.906 | 0.6 | 1.0 | 1.0015 | 0 |
| creative_hobbyist | 0.8048 | 0.9439 | 0.6 | 1.0 | 0.9805 | 0 |
| history_and_civics_follower | 0.8855 | 0.8855 | 0.6 | 1.0 | 0.9841 | 0 |
| **mean** | 0.8494 | 0.9069 | 0.56 | 1.0 | 0.9896 | 1 |

Top-5 exposure by channel size: {'huge': 8, 'micro': 9, 'medium': 4, 'small': 4}

## Coverage and cost

Evaluated by genre: {'lecture': 1, 'art': 2, 'education': 1, 'entertainment': 3, 'howto': 3, 'music': 1, 'science': 3, 'gaming': 2, 'news_commentary': 1, 'product_review': 1, 'technology': 2}

Evaluated by channel size: {'micro': 5, 'small': 4, 'huge': 8, 'large': 1, 'medium': 2}

Evaluated by language: {'en': 18, 'es': 1, 'de': 1}

Cost: total $0.0, per video $0.0 (20 videos)
