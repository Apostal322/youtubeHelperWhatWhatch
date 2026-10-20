# Throughput and cost: measured call plan, projected scale

Commit `2093bcc848f9ffcc0da06cd548831f3660516b60`; data: real_pilot@v1 (29 long-form episodes, median ~11k words).

> **No live LLM calls were made.** The LLM call plan is *measured* by running the production
> pipeline with a dry-run provider; the dollar and token figures are *projections* from
> the stated assumptions below.

## Measured (this environment)

* Heuristic pipeline per video (ms): {'p50': 85.81, 'p95': 122.64}
* Hashing embedding per video: 5.55 ms; MiniLM ONNX: 805 (retrieval experiment, 24 windows/doc, CPU) ms

| stage | model | LLM calls per video (mean; p50/p95) | call kinds | est. input tokens | assumed output tokens | est. $ per video (mean; p95) |
|---|---|---|---|---:|---:|---|
| B | claude-opus-5-5 | 1.0; {'p50': 1.0, 'p95': 1.0} | rubric_evaluation | 18,266 | 2,500 | 0.1231; 0.15 |
| C | claude-opus-5-5 | 3.0; {'p50': 3.0, 'p95': 3.0} | rubric_evaluation | 54,799 | 7,500 | 0.3692; 0.44 |

## Projected (assumptions, not measurements)

Assumptions: `{"chars_per_token": 4.0, "output_tokens_per_call": {"chunk_notes": 1200, "rubric_evaluation": 2500, "default": 1500}, "stage_c_escalation_rate": 0.2, "llm_latency_seconds_per_call": 60.0, "prices_usd_per_mtok": {"claude-opus-5-5": [4.0, 20.0]}, "representativeness": "pilot items are ~1 h podcasts; typical YouTube videos are shorter, so per-video cost is likely lower", "workers": "4 concurrent LLM requests per worker (AVD_LLM_MAX_CONCURRENCY), 1.5x headroom", "completion_time": "~96 s per item at steady state (calls in sequence x latency); backlog grows without bound if workers < needed", "storage": "transcript + 3x heuristic evaluation JSON (LLM evaluations carry justifications and quotes) + 384-d float32 vector; excludes media and indexes", "youtube_quota": "search.list returns <= 50 results; units from the client's quota table; captions require the channel owner's OAuth, so owner captions are a per-creator grant, not a crawlable source", "failure_rates": "unmeasured: no live calls; the evaluate-real runner records them per run"}`

| videos/day | LLM $/day | $/day if output ×3 | LLM $/month | tokens/day (M) | tokens/min | LLM calls/min | concurrent requests | LLM workers | heuristic CPU s/day | storage GB/day (/yr) | YouTube units/day: discovery · owner captions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 100 | 20 | 36 | 591 | 3.3 | 2,307 | 0.1 | 0.1 | 1 | 9 | 0.012 (4.3) | 202 (ok) · 25,000 |
| 1,000 | 197 | 357 | 5,908 | 33.2 | 23,073 | 1.1 | 1.1 | 1 | 89 | 0.117 (42.8) | 2,020 (ok) · 250,000 |
| 10,000 | 1,969 | 3,569 | 59,082 | 332.3 | 230,735 | 11.1 | 11.1 | 5 | 894 | 1.172 (427.7) | 20,200 (**exceeds 10k default**) · 2,500,000 |
| 100,000 | 19,694 | 35,694 | 590,820 | 3322.6 | 2,307,347 | 111.1 | 111.1 | 42 | 8,944 | 11.717 (4276.8) | 202,000 (**exceeds 10k default**) · 25,000,000 |
