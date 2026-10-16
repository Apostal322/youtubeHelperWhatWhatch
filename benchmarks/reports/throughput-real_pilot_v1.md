# Throughput and cost: measured call plan, projected scale

Commit `51d8424664a1a2180b12ca1b46c475015e97810a`; data: real_pilot@v1 (29 long-form episodes, median ~11k words).

> **No live LLM calls were made.** The LLM call plan is *measured* by running the production
> pipeline with a dry-run provider; the dollar and token figures are *projections* from
> the stated assumptions below.

## Measured (this environment)

* Heuristic pipeline per video (ms): {'p50': 75.95, 'p95': 107.76}
* Hashing embedding per video: 4.43 ms; MiniLM ONNX: 805 (retrieval experiment, 24 windows/doc, CPU) ms

| stage | model | LLM calls per video (mean; p50/p95) | call kinds | est. input tokens | assumed output tokens | est. $ per video (mean; p95) |
|---|---|---|---|---:|---:|---|
| B | claude-opus-5-5 | 1.0; {'p50': 1.0, 'p95': 1.0} | rubric_evaluation | 18,266 | 2,500 | 0.1231; 0.15 |
| C | claude-opus-5-5 | 3.0; {'p50': 3.0, 'p95': 3.0} | rubric_evaluation | 54,799 | 7,500 | 0.3692; 0.44 |

## Projected (assumptions, not measurements)

Assumptions: `{"chars_per_token": 4.0, "output_tokens_per_call": {"chunk_notes": 1200, "rubric_evaluation": 2500, "default": 1500}, "stage_c_escalation_rate": 0.2, "llm_latency_seconds_per_call": 60.0, "prices_usd_per_mtok": {"claude-opus-5-5": [4.0, 20.0]}, "representativeness": "pilot items are ~1 h podcasts; typical YouTube videos are shorter, so per-video cost is likely lower"}`

| videos/day | LLM $/day | $/day if output ×3 | LLM $/month | tokens/day (M) | LLM calls/min | concurrent requests | heuristic CPU s/day |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 197 | 357 | 5,908 | 33.2 | 1.1 | 1.1 | 79 |
| 10,000 | 1,969 | 3,569 | 59,082 | 332.3 | 11.1 | 11.1 | 786 |
| 100,000 | 19,694 | 35,694 | 590,820 | 3322.6 | 111.1 | 111.1 | 7,857 |
