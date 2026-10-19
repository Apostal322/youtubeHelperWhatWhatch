# Configuring live AI evaluation safely

No live model has been run in this repository yet. This page is the checklist for the first
run, and every one after it.

**To run without an API key or any paid call**, use a local open-weight model through Ollama
(`AVD_LLM_PROVIDER=ollama`). See [`local-ai.md`](local-ai.md). Local runs need no budget, and
everything below about prices and budgets applies to the hosted providers.

## 1. Credentials (never in code)

| Provider | Settings |
|---|---|
| Anthropic (default) | `AVD_LLM_PROVIDER=anthropic`, `AVD_ANTHROPIC_API_KEY` |
| Ollama (local, no key) | `AVD_LLM_PROVIDER=ollama`, `AVD_OLLAMA_BASE_URL` (default `http://localhost:11434`, must be local), `AVD_OLLAMA_MODEL`; see [`local-ai.md`](local-ai.md) |
| OpenAI-compatible (incl. local servers) | `AVD_LLM_PROVIDER=openai_compatible`, `AVD_OPENAI_BASE_URL` (default `https://api.openai.com/v1`), `AVD_OPENAI_API_KEY` (required unless the endpoint is localhost) |

* Set keys as environment secrets: the deployment's secret store, or the cloud environment's
  secret settings. Never put them in `.env` files that are committed, or in screenshots or
  issues.
* Keys are `SecretStr`: never logged, never returned. `/v1/status` reports only *whether* a key
  is configured.
* The browser extension never holds model credentials.

## 2. Model, prompt and rubric

* **Model:** `AVD_LLM_MODEL_STAGE_B` (default `claude-opus-5-5`), `AVD_LLM_MODEL_STAGE_C`. A
  model without a known price is refused by `avd evaluate-real` unless
  `AVD_LLM_PRICE_OVERRIDES="model=in:out"` is set. The budget cannot be enforced without a
  price.
* **Prompt:** `AVD_LLM_PROMPT`, a registered key (`avd prompts` lists them with metadata):
  * `rubric_prompt_v2`, the default;
  * `rubric_prompt_v2-evidence_first` and `rubric_prompt_v2-strict_abstain`, sensitivity
    variants that each change one thing.

  The key is recorded on every evaluation, cache entry and ledger row, and is part of the config
  hash, so changing it re-evaluates instead of mixing results.
* **Rubric:** `rubric_v2` (`--rubric-version` makes the runner refuse a mismatch).
* **Prompt languages:** written for English; **no language has been verified live.**

## 3. Spending controls (all enforced before a call is sent)

| Control | Setting / flag | Scope |
|---|---|---|
| Approved run ceiling | `avd evaluate-real --budget-usd` | Each call reserves its worst-case cost (prompt at 3 chars/token, full `max_tokens` output) before it is sent, so concurrency and retries cannot cross the ceiling. Spend is read back from the run's ledger, so resumes count. |
| Expected-cost refusal | (automatic) | The run does not start when the projected *expected* cost exceeds the budget |
| Daily budget | `AVD_DAILY_LLM_BUDGET_USD` | Production workers |
| Per-job cap | `AVD_MAX_JOB_COST_USD` | Production workers |
| Retries | `AVD_LLM_MAX_RETRIES` / `--max-retries` | Transient errors only, exponential backoff; never refusals or invalid output |
| Timeout | `AVD_LLM_TIMEOUT_SECONDS` / `--timeout` | Per request |
| Concurrency | `AVD_LLM_MAX_CONCURRENCY` / `--concurrency` | In-flight requests |
| Stage C escalation | off unless `--stage-c` | 3 more calls per escalated item |

**Do not treat a projection as permission to spend.** Every live run needs an explicit budget
from the project owner.

## 4. The first live run (development split only)

```bash
cd backend
avd evaluate-real --dry-run --offline --run-id dev-plan    # no calls: plan + projected cost
#   real_pilot_v1 dev, Opus 5.5, Stage B: expected $1.21, worst case $4.14 (2026-10-19)
export AVD_LLM_PROVIDER=anthropic   # and AVD_ANTHROPIC_API_KEY from the secret store
avd evaluate-real --offline --run-id dev-opus-base --budget-usd 5 --concurrency 2
# interrupted? re-run the same command: completed items and calls are not repeated or re-billed
avd evaluate-real --offline --run-id dev-opus-evidence-first --budget-usd 5 \
    --prompt rubric_prompt_v2-evidence_first              # Experiment D (prompt sensitivity)
```

Outputs go to `benchmarks/runs/<run-id>/`: `plan.json`, `summary.json`, `report.md`,
`items/*.json` and the run database. The report records the following:
* calls (ok / cached / error by type), parse failures, refusals and budget refusals;
* evidence claimed vs verified, and ungrounded dimensions;
* abstentions by reason, tokens, cost and latency;
* model, prompt, rubric, config hashes, git commit and dirty flag.

A run's numbers are **operational evidence only**. Agreement with people needs human labels:
`avd bench-real --labels <export dir>`.

## 5. Held-out data

* `real_pilot_v1` test split: partly exposed (see its `exposure.json`).
* `real_holdout_v2`: sealed.

`avd evaluate-real` refuses either without `--confirm-test` and a valid gate lock. Use them only
for the single confirmatory run of a locked candidate.
