# Extending AI Video Discovery

The codebase already separates its parts behind small Python `Protocol`s. There is no plugin
framework: each extension point is an interface plus one place that chooses the implementation.
This keeps extensions reviewable and the official protocol intact.

| Extension point | Interface | Chosen in | Notes |
|---|---|---|---|
| Evaluator (judges dimensions) | `avd.evaluation.evaluators.base.Evaluator` | `avd.evaluation.pipeline.build_pipeline`; experimental ones via `AVD_EXPERIMENTAL_EVALUATORS` | must return one `DimensionScore` per dimension, abstaining explicitly |
| Model provider | `avd.llm.base.LLMProvider` (`complete_json`) | `avd.llm.providers.build_llm_provider` | wrapped by retries, rate limits, spend guards, cache and ledger automatically |
| Content source | `avd.content.providers.ContentProvider` | `avd.workers.context` | must record an authorization basis; never scrape |
| Fact-check retrieval | `avd.evaluation.evaluators.factual.ReferenceRetriever` | `load_reference_corpus` | passages need license metadata |
| Embeddings | `avd.embeddings.base.Embedder` (`embed`, `model_id`) | `get_embedder` / `AVD_EMBEDDING_PROVIDER` | the model id is stored with every vector; changing it triggers re-indexing |
| Ranking | `avd.recsys.ranker.Ranker` | `avd.recsys.recommender` | ranking never changes intrinsic quality scores |
| Export | `avd.portability` (format v1) | `avd evaluations export` | add fields as optional (1.x); a breaking change needs 2.0 |
| Benchmarks | `avd bench`, `avd bench-real`, `avd bench-local` | `avd.bench.*` | held-out data stays sealed; gates are not edited to pass |

## Example: a custom evaluator without touching the core

[`examples/word_count_evaluator.py`](examples/word_count_evaluator.py) is a complete, tested
evaluator. It judges `information_density` from a transparent word-rate rule and abstains when
it cannot. Run it:

```bash
PYTHONPATH=docs/examples AVD_EXPERIMENTAL_EVALUATORS='word_count_evaluator:WordRateEvaluator' avd start
```

What the platform guarantees for experimental evaluators (`avd/plugins.py`, tested in
`tests/test_local_first.py::TestExperimentalEvaluators`):

* **Explicit only.** Code runs only when the operator names it. There is no auto-discovery of
  entry points or directories.
* **Cannot override the official protocol.** Experimental evaluators have the lowest
  precedence, so they only fill dimensions the official evaluators abstain on.
* **Never silent.** `name@version` is recorded in every affected evaluation's
  `versions["experimental_evaluators"]` and in the config hash. Affected evaluations are
  therefore distinguishable, re-run when the configuration changes, and visible in exports.
* **Not automatically credible.** Benchmarks report what ran. An experimental evaluator becomes
  part of the validated configuration only through the research protocol: dev-split
  development, a locked candidate, and human-label comparison (`docs/evaluation.md`).
* **Trust.** An evaluator receives only the evaluation input and the genre profile, never
  settings or credentials. It is still Python running in the API process and could read the
  environment, so review it like any dependency.

## Adding a model provider

Implement `complete_json(LLMRequest) -> LLMResponse` (see `avd/llm/ollama.py` for a complete
local example). Then:

* return `served_model` with a version or digest, so results are reproducible;
* if answers depend on runtime settings, expose `revision(model)`, so that they enter the cache
  key and config hash;
* raise `LLMTransientError` for retryable failures, `LLMConfigurationError` for setup problems
  and `LLMOutputError` for unusable output;
* never fall back to another provider silently.

## Adding a rubric or prompt

Register prompts in `avd/evaluation/prompts.py` with a version, the rubric version they
implement and their compatible models. Prompts and rubrics enter the config hash. A new rubric
is a new `RUBRIC_VERSION`: scores under different rubrics are not comparable and are not mixed.
